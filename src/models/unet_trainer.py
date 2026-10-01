"""
Trainer and evaluator for U-Net Glomeruli Segmentation (NVIDIA A40 Optimized).

Features:
- Pre-trained backbone support (ResNet34, ResNet50, etc. via segmentation_models_pytorch)
- Mixed precision training (AMP fp16/bf16 with GradScaler)
- Learning rate warmup + Cosine Annealing scheduler
- Early stopping on validation Dice score
- CSV epoch-by-epoch logging (results.csv) & JSON train summary
- Pixel-level segmentation metrics (Dice, IoU, Precision, Recall, Specificity)
- Object-level detection metrics (AP50, Precision, Recall, F1 via contour bounding boxes)
- Publication-quality qualitative test visualization grid
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import csv
import json
import logging
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.models.losses import build_criterion, compute_metrics
from src.models.mask_to_bbox import extract_bboxes_from_mask, evaluate_detection_metrics
from src.models.unet import build_segmentation_model
from src.models.unet_dataset import UNetDataset

logger = logging.getLogger(__name__)


class UNetTrainer:
    """Manages the full lifecycle of U-Net: training, validation, and test evaluation."""

    def __init__(
        self,
        config_path: Optional[str] = "configs/unet_config.yaml",
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config_dict is not None:
            self.cfg = config_dict
            self.config_path = Path(config_path) if config_path else Path("configs/unet_config.yaml")
        else:
            self.config_path = Path(config_path or "configs/unet_config.yaml")
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)

        # Setup compute device
        device_cfg = self.cfg.get("device", 0)
        if torch.cuda.is_available():
            if isinstance(device_cfg, int) or (isinstance(device_cfg, str) and device_cfg.isdigit()):
                self.device = torch.device(f"cuda:{device_cfg}")
            else:
                self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            logger.info(f"Using GPU: {torch.cuda.get_device_name(self.device)} (Device: {self.device})")
        else:
            self.device = torch.device("cpu")
            logger.info(f"Using CPU device: {self.device}")

        # Setup output directory
        self.run_dir = Path(self.cfg.get("project", "runs/unet")) / self.cfg.get("name", "unet_20x_resnet34")
        self.weights_dir = self.run_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Architecture
        self.model = build_segmentation_model(self.cfg).to(self.device)

        # Optional PyTorch 2.x torch.compile for NVIDIA Ampere (A40)
        if self.cfg.get("compile", False) and hasattr(torch, "compile"):
            logger.info("Enabling PyTorch 2.0 graph compilation (torch.compile)...")
            self.model = torch.compile(self.model)

        # Initialize Criterion
        self.criterion = build_criterion(self.cfg)

        # Optimizer
        self.lr = float(self.cfg.get("lr", 0.0003))
        self.weight_decay = float(self.cfg.get("weight_decay", 0.0001))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.epochs = int(self.cfg.get("epochs", 50))
        self.min_lr = float(self.cfg.get("min_lr", 1e-6))
        self.warmup_epochs = int(self.cfg.get("warmup_epochs", 2))

        # Scheduler: Cosine Annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.epochs - self.warmup_epochs),
            eta_min=self.min_lr,
        )

        # Mixed precision setup (FP16 / BF16 for NVIDIA A40)
        self.use_amp = bool(self.cfg.get("amp", True)) and (self.device.type == "cuda")
        self.use_bfloat16 = bool(self.cfg.get("bfloat16", False)) and self.use_amp
        self.amp_dtype = torch.bfloat16 if self.use_bfloat16 else torch.float16

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and not self.use_bfloat16))
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.use_amp and not self.use_bfloat16))

        self.grad_accum_steps = max(1, int(self.cfg.get("gradient_accumulation_steps", 1)))
        self.clip_grad_norm = float(self.cfg.get("clip_grad_norm", 1.0))

        # CSV Logging path
        self.csv_log_path = self.run_dir / "results.csv"

    def _autocast_context(self):
        """Returns the appropriate autocast context manager."""
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype)
        return torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype)

    def get_dataloader(self, split: str) -> DataLoader:
        """Create DataLoader for a given dataset split."""
        dataset = UNetDataset(
            manifest_path=self.cfg.get("manifest_csv", "dataset/manifest.csv"),
            split=split,
            img_size=int(self.cfg.get("img_size", 1024)),
            augment=(split == "train"),
            color_jitter=self.cfg.get("color_jitter", None),
            positive_only=bool(self.cfg.get("positive_only", False) and split == "train"),
        )

        batch_size = int(self.cfg.get("batch_size", 16)) if split == "train" else max(1, int(self.cfg.get("batch_size", 16)) // 2)
        num_workers = int(self.cfg.get("num_workers", 8))

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=(split == "train"),
            persistent_workers=(num_workers > 0),
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Execute one training epoch with gradient accumulation and AMP."""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        self.optimizer.zero_grad()

        # Handle linear warmup
        if epoch <= self.warmup_epochs:
            warmup_factor = float(epoch) / float(max(1, self.warmup_epochs))
            current_lr = self.lr * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

        for step, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with self._autocast_context():
                logits = self.model(images)
                raw_loss = self.criterion(logits, masks)
                loss = raw_loss / float(self.grad_accum_steps)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step every grad_accum_steps or on final batch
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(dataloader):
                if self.scaler.is_enabled():
                    if self.clip_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            batch_metrics = compute_metrics(logits, masks, threshold=float(self.cfg.get("threshold", 0.5)))

            total_loss += raw_loss.item()
            total_dice += batch_metrics["dice"]
            total_iou += batch_metrics["iou"]

            pbar.set_postfix({"loss": f"{raw_loss.item():.4f}", "dice": f"{batch_metrics['dice']:.4f}"})

        n = len(dataloader)
        return {
            "loss": total_loss / n,
            "dice": total_dice / n,
            "iou": total_iou / n,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation split."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_specificity = 0.0

        for images, masks, _ in dataloader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with self._autocast_context():
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            metrics = compute_metrics(logits, masks, threshold=float(self.cfg.get("threshold", 0.5)))

            total_loss += loss.item()
            total_dice += metrics["dice"]
            total_iou += metrics["iou"]
            total_precision += metrics["precision"]
            total_recall += metrics["recall"]
            total_specificity += metrics["specificity"]

        n = len(dataloader)
        return {
            "loss": total_loss / n,
            "dice": total_dice / n,
            "iou": total_iou / n,
            "precision": total_precision / n,
            "recall": total_recall / n,
            "specificity": total_specificity / n,
        }

    def fit(self) -> Path:
        """Run full training routine with validation, logging, and early stopping."""
        train_loader = self.get_dataloader("train")
        val_loader = self.get_dataloader("val")

        logger.info(f"Starting U-Net training for {self.epochs} epochs...")
        logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

        # Initialize CSV logging
        with open(self.csv_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "time_sec", "train/loss", "train/dice", "train/iou",
                "val/loss", "val/dice", "val/iou", "val/precision", "val/recall", "val/specificity", "lr"
            ])

        best_val_dice = -1.0
        best_epoch = 0
        patience_counter = 0
        patience = int(self.cfg.get("patience", 15))

        best_model_path = self.weights_dir / "best.pt"
        start_time_all = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            train_res = self.train_epoch(train_loader, epoch=epoch)
            val_res = self.validate(val_loader)

            if epoch > self.warmup_epochs:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            # Write to results.csv
            with open(self.csv_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, round(epoch_time, 2),
                    round(train_res["loss"], 4), round(train_res["dice"], 4), round(train_res["iou"], 4),
                    round(val_res["loss"], 4), round(val_res["dice"], 4), round(val_res["iou"], 4),
                    round(val_res["precision"], 4), round(val_res["recall"], 4), round(val_res["specificity"], 4),
                    f"{current_lr:.6e}"
                ])

            logger.info(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] ({epoch_time:.1f}s) "
                f"Train Loss: {train_res['loss']:.4f} Dice: {train_res['dice']:.4f} | "
                f"Val Loss: {val_res['loss']:.4f} Dice: {val_res['dice']:.4f} IoU: {val_res['iou']:.4f} "
                f"Prec: {val_res['precision']:.4f} Rec: {val_res['recall']:.4f}"
            )

            # Check if validation metric improved
            if val_res["dice"] > best_val_dice:
                best_val_dice = val_res["dice"]
                best_epoch = epoch
                patience_counter = 0

                # Save best checkpoint
                raw_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_res,
                    "config": self.cfg,
                }, best_model_path)
                logger.info(f"   >>> Saved new best model checkpoint (Val Dice: {best_val_dice:.4f} at epoch {best_epoch})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs (Best epoch: {best_epoch} with Val Dice: {best_val_dice:.4f}).")
                    break

        # Save last checkpoint
        raw_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        torch.save(raw_model.state_dict(), self.weights_dir / "last.pt")

        # Save train summary
        total_time = time.time() - start_time_all
        summary = {
            "best_epoch": best_epoch,
            "best_val_dice": round(best_val_dice, 4),
            "total_time_sec": round(total_time, 2),
            "architecture": self.cfg.get("arch", "unet"),
            "encoder": self.cfg.get("encoder_name", "resnet34"),
        }
        with open(self.run_dir / "train_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training completed in {total_time/60.0:.2f} min. Best model saved to: {best_model_path}")
        return best_model_path

    @torch.no_grad()
    def evaluate_test(
        self,
        checkpoint_path: Optional[str] = None,
        save_visualizations: bool = True,
        max_vis: int = 6
    ) -> Dict[str, Any]:
        """
        Evaluate checkpoint on Test set with both segmentation and detection metrics.
        """
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            raw_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            raw_model.load_state_dict(state_dict)

        self.model.eval()
        test_loader = self.get_dataloader("test")

        logger.info(f"Evaluating on Test set ({len(test_loader.dataset)} samples)...")

        all_pred_boxes = []
        all_gt_boxes = []

        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_pixel_p = 0.0
        total_pixel_r = 0.0
        total_pixel_sp = 0.0

        vis_samples = []

        threshold = float(self.cfg.get("threshold", 0.5))
        min_blob_area = int(self.cfg.get("min_blob_area", 400))

        for images, masks, metas in tqdm(test_loader, desc="Evaluating Test"):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with self._autocast_context():
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            total_loss += loss.item()
            seg_metrics = compute_metrics(logits, masks, threshold=threshold)
            total_dice += seg_metrics["dice"]
            total_iou += seg_metrics["iou"]
            total_pixel_p += seg_metrics["precision"]
            total_pixel_r += seg_metrics["recall"]
            total_pixel_sp += seg_metrics["specificity"]

            # Convert batch predictions to numpy for detection evaluation
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0, :, :]
            pred_masks = (probs > threshold).astype(np.uint8) * 255
            gt_masks = (masks.cpu().numpy()[:, 0, :, :] > 0.5).astype(np.uint8) * 255

            for b_idx in range(len(images)):
                p_mask = pred_masks[b_idx]
                g_mask = gt_masks[b_idx]
                p_prob = probs[b_idx]

                # Extract bounding boxes from predicted mask
                pred_detections = extract_bboxes_from_mask(p_mask, p_prob, min_area=min_blob_area)
                # Extract ground truth bounding boxes from GT mask
                gt_detections = extract_bboxes_from_mask(g_mask, g_mask.astype(float)/255.0, min_area=100)
                gt_boxes_b = [d["bbox"] for d in gt_detections]

                all_pred_boxes.append(pred_detections)
                all_gt_boxes.append(gt_boxes_b)

                # Collect visual samples
                if len(vis_samples) < max_vis and (len(gt_boxes_b) > 0 or len(pred_detections) > 0):
                    img_np = images[b_idx].cpu().permute(1, 2, 0).numpy()
                    img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255.0
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

                    vis_samples.append({
                        "patch_id": metas["patch_id"][b_idx],
                        "image": img_np,
                        "gt_mask": g_mask,
                        "pred_mask": p_mask,
                        "pred_detections": pred_detections,
                        "gt_boxes": gt_boxes_b,
                    })

        n = len(test_loader)
        pixel_results = {
            "test_loss": round(total_loss / n, 4),
            "dice": round(total_dice / n, 4),
            "iou": round(total_iou / n, 4),
            "pixel_precision": round(total_pixel_p / n, 4),
            "pixel_recall": round(total_pixel_r / n, 4),
            "pixel_specificity": round(total_pixel_sp / n, 4),
        }

        # Compute object detection metrics (AP50, Precision, Recall, F1)
        det_results = evaluate_detection_metrics(all_pred_boxes, all_gt_boxes, iou_threshold=0.5)

        full_results = {
            "segmentation_metrics": pixel_results,
            "detection_metrics": det_results,
        }

        # Save test metrics JSON
        json_path = self.run_dir / "test_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)
        logger.info(f"Saved test evaluation metrics to {json_path}")

        # Save visual comparison figure
        if save_visualizations and len(vis_samples) > 0:
            self._save_test_visualizations(vis_samples)

        return full_results

    def _save_test_visualizations(self, samples: List[Dict[str, Any]]) -> None:
        """Render publication-quality qualitative comparison figure."""
        n_samples = len(samples)
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
        if n_samples == 1:
            axes = np.expand_dims(axes, 0)

        for i, s in enumerate(samples):
            img = s["image"]
            gt_mask = s["gt_mask"]
            pred_mask = s["pred_mask"]

            # Panel 1: Original Image with GT Mask Overlay (Green)
            ax1 = axes[i, 0]
            ov1 = img.copy()
            ov1[gt_mask > 0] = (ov1[gt_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            ax1.imshow(ov1)
            ax1.set_title(f"{s['patch_id']}\nGround Truth Mask (Green)", fontsize=10)
            ax1.axis("off")

            # Panel 2: Predicted Segmentation Mask (Cyan)
            ax2 = axes[i, 1]
            ov2 = img.copy()
            ov2[pred_mask > 0] = (ov2[pred_mask > 0] * 0.5 + np.array([0, 255, 255]) * 0.5).astype(np.uint8)
            ax2.imshow(ov2)
            ax2.set_title("U-Net Predicted Mask (Cyan)", fontsize=10)
            ax2.axis("off")

            # Panel 3: Extracted Bounding Boxes (Red) vs GT Boxes (Green)
            ax3 = axes[i, 2]
            ov3 = img.copy()
            # Draw GT boxes in green
            for gx1, gy1, gx2, gy2 in s["gt_boxes"]:
                cv2.rectangle(ov3, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            # Draw Predicted boxes in red
            for d in s["pred_detections"]:
                px1, py1, px2, py2 = d["bbox"]
                cv2.rectangle(ov3, (px1, py1), (px2, py2), (255, 0, 0), 2)
                cv2.putText(ov3, f"{d['score']:.2f}", (px1, max(15, py1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            ax3.imshow(ov3)
            ax3.set_title("Extracted Boxes (Red: Pred, Green: GT)", fontsize=10)
            ax3.axis("off")

        plt.tight_layout()
        vis_path = self.run_dir / "test_predictions_grid.png"
        plt.savefig(vis_path, dpi=150)
        plt.close()
        logger.info(f"Saved qualitative test visualizations to: {vis_path}")

