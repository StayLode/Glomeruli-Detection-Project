#!/usr/bin/env python3
"""
CLI Script to train U-Net on the extracted glomeruli segmentation dataset.

Usage:
    python scripts/train_unet.py --config configs/unet_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet_trainer import UNetTrainer


def setup_logger() -> logging.Logger:
    """Configure structured console logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net on Glomeruli Segmentation Dataset (NVIDIA A40 Optimized).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/unet_config.yaml",
        help="Path to U-Net training configuration YAML file."
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Override architecture: unet, unetplusplus, deeplabv3plus."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Override encoder backbone: resnet34, resnet50, efficientnet-b3."
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=None,
        help="Override loss function: combo, bce_dice, focal_tversky."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size (e.g. 16 for NVIDIA A40 46GB)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override initial learning rate."
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override compute device (e.g. 0, 'cuda:0', 'cpu')."
    )
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("Initializing U-Net Trainer...")

    # Load config and apply CLI overrides before initializing trainer
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.arch is not None:
        cfg["arch"] = args.arch
    if args.encoder is not None:
        cfg["encoder_name"] = args.encoder
    if args.loss is not None:
        cfg["loss_type"] = args.loss
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        cfg["gradient_accumulation_steps"] = args.grad_accum_steps
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.device is not None:
        cfg["device"] = args.device

    trainer = UNetTrainer(config_path=args.config, config_dict=cfg)

    # 1. Train model
    best_checkpoint = trainer.fit()
    logger.info(f"Training completed. Best checkpoint: {best_checkpoint}")

    # 2. Evaluate on Test set
    test_results = trainer.evaluate_test(checkpoint_path=str(best_checkpoint))

    # Print summary table
    seg_m = test_results["segmentation_metrics"]
    det_m = test_results["detection_metrics"]

    print("\n" + "=" * 65)
    print("U-NET EVALUATION ON TEST SET (RECHERCHE-015, RECHERCHE-017)")
    print("=" * 65)
    print("PIXEL-LEVEL SEGMENTATION METRICS:")
    print(f"   Dice Score (F1)     : {seg_m['dice']:.4f}")
    print(f"   IoU (Jaccard Index) : {seg_m['iou']:.4f}")
    print(f"   Pixel Precision     : {seg_m['pixel_precision']:.4f}")
    print(f"   Pixel Recall        : {seg_m['pixel_recall']:.4f}")
    print(f"   Pixel Specificity   : {seg_m.get('pixel_specificity', 0.0):.4f}")
    print("-" * 65)
    print("OBJECT-LEVEL DETECTION METRICS (Extracted BBoxes vs GT):")
    print(f"   mAP@50 (AP50)       : {det_m['ap50']:.4f}")
    print(f"   BBox Precision      : {det_m['precision']:.4f}")
    print(f"   BBox Recall         : {det_m['recall']:.4f}")
    print(f"   BBox F1-Score       : {det_m['f1']:.4f}")
    print(f"   Total GT Glomeruli  : {det_m['total_gt']}")
    print(f"   Total Predicted     : {det_m['total_preds']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
