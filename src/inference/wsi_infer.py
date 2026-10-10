"""
Whole Slide Image (WSI) Inference and Stitching Engine.

Supports:
1. Fast screening with YOLOv8 (detect candidate bounding boxes).
2. Cascade mode (YOLO + U-Net): Gated patch segmentation where U-Net segments
   only the regions proposed by YOLO, eliminating background false positives.
3. Global Non-Maximum Suppression (NMS) across tile boundaries.
4. ASAP-compliant XML export with fine Polygon annotations and BBoxes.
5. Multi-panel qualitative comparison figures (Biopsy vs GT vs Pred vs Overlap)
   and slide-level clinical metrics (Precision, Recall, F1, Dice, IoU).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
from shapely.geometry import Polygon, box
import torch
import torchvision
from tqdm import tqdm
from ultralytics import YOLO

from src.models.mask_to_bbox import box_iou
from src.models.unet import build_segmentation_model
from src.preprocessing.tissue_detector import TissueDetector
from src.utils.xml_parser import ASAPAnnotation, parse_asap_xml

logger = logging.getLogger(__name__)


class WSIInferenceEngine:
    """Performs whole slide inference, stitching, and evaluation for glomeruli detection & segmentation."""

    def __init__(
        self,
        model_weights_path: str,
        unet_weights_path: Optional[str] = None,
        patch_size: int = 1024,
        stride: int = 768,
        target_mag: int = 20,
        base_mag: int = 40,
        device: Optional[str] = None,
        unet_threshold: float = 0.45,
        min_glom_area: int = 100,
        box_margin: float = 0.15,
    ) -> None:
        self.model_path = Path(model_weights_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {self.model_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading YOLO model from {self.model_path} on {self.device}...")
        self.model = YOLO(str(self.model_path))

        self.patch_size = patch_size
        self.stride = stride
        self.downsample = base_mag / float(target_mag)
        self.patch_size_l0 = int(round(patch_size * self.downsample))
        self.stride_l0 = int(round(stride * self.downsample))

        self.unet_threshold = unet_threshold
        self.min_glom_area = min_glom_area
        self.box_margin = box_margin

        # Load U-Net model if provided (Cascade Mode)
        self.unet_model: Optional[torch.nn.Module] = None
        if unet_weights_path:
            self._init_unet(Path(unet_weights_path))

    def _init_unet(self, weights_path: Path) -> None:
        """Initialize U-Net segmentation model from checkpoint."""
        if not weights_path.exists():
            raise FileNotFoundError(f"U-Net checkpoint not found: {weights_path}")

        logger.info(f"Loading U-Net model from {weights_path} on {self.device} (Cascade Mode enabled)...")
        ckpt = torch.load(weights_path, map_location=self.device)

        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        unet = build_segmentation_model(cfg)

        state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
        # Strip torch.compile prefix '_orig_mod.' if present
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        unet.load_state_dict(clean_state_dict)
        unet.to(self.device)
        unet.eval()
        self.unet_model = unet

        self.unet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.unet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _plan_tissue_patches(
        self, slide: openslide.OpenSlide, detector: TissueDetector, min_tissue_ratio: float = 0.10
    ) -> List[Tuple[int, int]]:
        """Identify top-left (x0, y0) Level 0 coordinates that contain biopsy tissue."""
        w_l0, h_l0 = slide.dimensions
        valid_coords: List[Tuple[int, int]] = []

        y_max = h_l0 - self.patch_size_l0
        x_max = w_l0 - self.patch_size_l0

        for y0 in range(0, y_max + 1, self.stride_l0):
            for x0 in range(0, x_max + 1, self.stride_l0):
                if detector.is_tissue_patch(x0, y0, self.patch_size_l0, self.patch_size_l0, min_tissue_ratio):
                    valid_coords.append((x0, y0))

        return valid_coords

    def _segment_patch(self, patch_rgb: np.ndarray) -> np.ndarray:
        """Run U-Net inference on a single (H, W, 3) RGB patch, returning probability map in [0, 1]."""
        if self.unet_model is None:
            raise RuntimeError("U-Net model is not initialized.")
        tensor = torch.from_numpy(patch_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = (tensor.to(self.device) - self.unet_mean) / self.unet_std
        with torch.no_grad():
            logits = self.unet_model(tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        return probs

    def predict_slide(
        self,
        svs_path: Union[str, Path],
        xml_path: Optional[Union[str, Path]] = None,
        conf_thresh: float = 0.20,
        nms_iou_thresh: float = 0.40,
        batch_size: int = 16,
        min_tissue_ratio: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Run whole-slide inference on an SVS file with global stitching.
        If U-Net is loaded, applies Strategy A (YOLO Screening + U-Net Gated Segmentation).

        Args:
            svs_path: Path to .svs Whole Slide Image.
            xml_path: Optional path to ASAP ground truth XML for accuracy evaluation.
            conf_thresh: Confidence threshold for YOLO proposals.
            nms_iou_thresh: IoU cutoff for Global Non-Maximum Suppression.
            batch_size: Number of patches per GPU inference batch.
            min_tissue_ratio: Minimum fraction of tissue required to process a patch.

        Returns:
            Dictionary containing global detections, metrics, and slide metadata.
        """
        svs_path = Path(svs_path)
        slide_id = svs_path.stem
        cascade_mode = self.unet_model is not None

        logger.info(f"Starting WSI inference for: {slide_id} (Mode: {'Cascade YOLO+U-Net' if cascade_mode else 'YOLO Only'})")

        slide = openslide.OpenSlide(str(svs_path))
        w_l0, h_l0 = slide.dimensions

        # 1. Segment tissue on overview thumbnail
        detector = TissueDetector(slide, thumbnail_size=1024, max_v_thresh=238)
        coords = self._plan_tissue_patches(slide, detector, min_tissue_ratio)
        skipped_glass_pct = 100.0 * (1.0 - detector.tissue_ratio)
        logger.info(f"Planned {len(coords):,} tissue patches to process (skipped {skipped_glass_pct:.1f}% glass background).")

        raw_boxes_l0: List[List[float]] = []
        raw_polygons_l0: List[List[Tuple[float, float]]] = []
        raw_scores: List[float] = []
        raw_patch_records: List[Dict[str, Any]] = []

        # 2. Process patches in batches
        pbar = tqdm(range(0, len(coords), batch_size), desc=f"Inference {slide_id}")
        for b_start in pbar:
            batch_coords = coords[b_start : b_start + batch_size]
            batch_imgs = []

            for x0, y0 in batch_coords:
                reg = slide.read_region((x0, y0), 0, (self.patch_size_l0, self.patch_size_l0))
                rgb = np.array(reg.convert("RGB"))
                if self.downsample != 1.0:
                    patch_img = cv2.resize(rgb, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
                else:
                    patch_img = rgb
                batch_imgs.append(patch_img)

            # Stage 1: Fast Screening with YOLO
            # Ultralytics expects numpy arrays to be in BGR format (like cv2.imread)
            # because its preprocessor executes im.flip(1) (BGR -> RGB).
            # Passing BGR arrays guarantees YOLO receives authentic RGB histological color channels.
            batch_imgs_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in batch_imgs]
            results = self.model.predict(
                source=batch_imgs_bgr,
                conf=conf_thresh,
                imgsz=self.patch_size,
                verbose=False,
                device=self.device,
            )

            # Process candidates per patch
            for (x0, y0), patch_img, res in zip(batch_coords, batch_imgs, results):
                if res.boxes is None or len(res.boxes) == 0:
                    continue

                boxes_xyxy = res.boxes.xyxy.cpu().numpy()  # Pixel coords [x1, y1, x2, y2] on 1024x1024
                confs = res.boxes.conf.cpu().numpy()

                if cascade_mode:
                    # Stage 2: Fine Boundary Segmentation & Verification via U-Net
                    probs = self._segment_patch(patch_img)
                    bin_mask = (probs > self.unet_threshold).astype(np.uint8)

                    for b, conf in zip(boxes_xyxy, confs):
                        bx1, by1, bx2, by2 = int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3]))
                        bw = bx2 - bx1
                        bh = by2 - by1

                        # Expand box with margin to inspect surrounding context / Bowman capsule
                        pad_x = int(bw * self.box_margin)
                        pad_y = int(bh * self.box_margin)
                        gx1 = max(0, bx1 - pad_x)
                        gy1 = max(0, by1 - pad_y)
                        gx2 = min(self.patch_size, bx2 + pad_x)
                        gy2 = min(self.patch_size, by2 + pad_y)

                        roi_mask = bin_mask[gy1:gy2, gx1:gx2]
                        roi_probs = probs[gy1:gy2, gx1:gx2]
                        seg_pixels = int(np.count_nonzero(roi_mask))
                        mean_prob = float(np.mean(roi_probs[roi_mask == 1])) if seg_pixels > 0 else float(np.mean(roi_probs))

                        # Verification criteria:
                        # Reject obvious false alarms where U-Net sees no glomerular tissue whatsoever,
                        # but preserve sclerotic/atrophic glomeruli where YOLO has reasonable confidence.
                        is_background = (seg_pixels < self.min_glom_area and mean_prob < 0.25 and float(conf) < 0.40)
                        if is_background:
                            continue

                        # Extract contour inside ROI
                        roi_contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if roi_contours:
                            largest_cnt = max(roi_contours, key=cv2.contourArea)
                            cnt_smooth = cv2.approxPolyDP(largest_cnt, epsilon=1.5, closed=True)
                            cnt_patch = cnt_smooth.copy()
                            cnt_patch[:, 0, 0] += gx1
                            cnt_patch[:, 0, 1] += gy1

                            poly_l0 = [
                                (float(x0 + pt[0][0] * self.downsample), float(y0 + pt[0][1] * self.downsample))
                                for pt in cnt_patch
                            ]
                        else:
                            poly_l0 = [
                                (float(x0 + bx1 * self.downsample), float(y0 + by1 * self.downsample)),
                                (float(x0 + bx2 * self.downsample), float(y0 + by1 * self.downsample)),
                                (float(x0 + bx2 * self.downsample), float(y0 + by2 * self.downsample)),
                                (float(x0 + bx1 * self.downsample), float(y0 + by2 * self.downsample)),
                            ]
                            cnt_patch = None

                        final_score = float(0.6 * float(conf) + 0.4 * mean_prob)

                        # Primary Bounding Box is YOLO's full capsule box projected to Level 0
                        x1_l0 = float(x0 + bx1 * self.downsample)
                        y1_l0 = float(y0 + by1 * self.downsample)
                        x2_l0 = float(x0 + bx2 * self.downsample)
                        y2_l0 = float(y0 + by2 * self.downsample)

                        raw_boxes_l0.append([x1_l0, y1_l0, x2_l0, y2_l0])
                        raw_polygons_l0.append(poly_l0)
                        raw_scores.append(final_score)

                        raw_patch_records.append({
                            "patch_coord": (x0, y0),
                            "patch_img": patch_img,
                            "patch_contour": cnt_patch,
                            "patch_bbox": [bx1, by1, bx2, by2],
                            "score": final_score,
                        })

                else:
                    # Standard YOLO-only projection
                    for b, conf in zip(boxes_xyxy, confs):
                        bx1, by1, bx2, by2 = int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3]))
                        x1_l0 = float(x0 + bx1 * self.downsample)
                        y1_l0 = float(y0 + by1 * self.downsample)
                        x2_l0 = float(x0 + bx2 * self.downsample)
                        y2_l0 = float(y0 + by2 * self.downsample)

                        poly_l0 = [(x1_l0, y1_l0), (x2_l0, y1_l0), (x2_l0, y2_l0), (x1_l0, y2_l0)]

                        raw_boxes_l0.append([x1_l0, y1_l0, x2_l0, y2_l0])
                        raw_polygons_l0.append(poly_l0)
                        raw_scores.append(float(conf))

                        raw_patch_records.append({
                            "patch_coord": (x0, y0),
                            "patch_img": patch_img,
                            "patch_contour": None,
                            "patch_bbox": [bx1, by1, bx2, by2],
                            "score": float(conf),
                        })

        logger.info(f"Extracted {len(raw_boxes_l0)} raw candidate detections across all patches.")

        # 3. Global Non-Maximum Suppression with IoU and IoS (boundary containment)
        final_boxes_l0: List[List[float]] = []
        final_polygons_l0: List[List[Tuple[float, float]]] = []
        final_scores: List[float] = []
        final_patch_records: List[Dict[str, Any]] = []

        if len(raw_boxes_l0) > 0:
            order = np.argsort(raw_scores)[::-1]
            keep_indices: List[int] = []

            for idx in order:
                b_cand = raw_boxes_l0[idx]
                is_duplicate = False

                for k in keep_indices:
                    b_kept = raw_boxes_l0[k]

                    # Standard IoU
                    iou = box_iou(b_cand, b_kept)

                    # Intersection over Smaller Area (IoS / partial boundary containment)
                    inter_w = max(0.0, min(b_cand[2], b_kept[2]) - max(b_cand[0], b_kept[0]))
                    inter_h = max(0.0, min(b_cand[3], b_kept[3]) - max(b_cand[1], b_kept[1]))
                    inter_area = inter_w * inter_h
                    area_cand = (b_cand[2] - b_cand[0]) * (b_cand[3] - b_cand[1])
                    area_kept = (b_kept[2] - b_kept[0]) * (b_kept[3] - b_kept[1])
                    min_area = min(area_cand, area_kept)
                    ios = (inter_area / min_area) if min_area > 0 else 0.0

                    if iou >= nms_iou_thresh or ios >= 0.45:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    keep_indices.append(int(idx))

            for k in keep_indices:
                final_boxes_l0.append(raw_boxes_l0[k])
                final_polygons_l0.append(raw_polygons_l0[k])
                final_scores.append(raw_scores[k])
                final_patch_records.append(raw_patch_records[k])

        logger.info(f"Post-NMS: {len(final_boxes_l0)} unique glomeruli detected on whole slide.")

        # 4. Ground Truth Evaluation (Detection & Segmentation)
        evaluation_results: Optional[Dict[str, Any]] = None
        gt_annots: List[ASAPAnnotation] = []

        if xml_path is not None and Path(xml_path).exists():
            gt_annots = parse_asap_xml(xml_path)
            evaluation_results = self._evaluate_wsi(
                final_boxes_l0, final_polygons_l0, final_scores, gt_annots, iou_thresh=0.40
            )

            seg_info = ""
            if cascade_mode and evaluation_results.get("mean_dice") is not None:
                seg_info = f" | Mean Dice: {evaluation_results['mean_dice']:.4f} | Mean IoU: {evaluation_results['mean_iou']:.4f}"

            logger.info("=" * 65)
            logger.info(f"WSI CLINICAL EVALUATION: {slide_id}")
            logger.info(f"   Pipeline Mode          : {'Cascade YOLO+U-Net' if cascade_mode else 'YOLO Only'}")
            logger.info(f"   Ground Truth Glomeruli : {evaluation_results['gt_count']}")
            logger.info(f"   Predicted Glomeruli    : {evaluation_results['pred_count']}")
            logger.info(f"   True Positives (TP)    : {evaluation_results['tp']}")
            logger.info(f"   False Positives (FP)   : {evaluation_results['fp']}")
            logger.info(f"   False Negatives (FN)   : {evaluation_results['fn']}")
            logger.info(f"   Slide Precision        : {evaluation_results['precision']:.4f}")
            logger.info(f"   Slide Recall           : {evaluation_results['recall']:.4f}")
            logger.info(f"   Slide F1-Score         : {evaluation_results['f1']:.4f}{seg_info}")
            logger.info("=" * 65)

        slide.close()

        return {
            "slide_id": slide_id,
            "svs_path": str(svs_path),
            "cascade_mode": cascade_mode,
            "level0_dimensions": [w_l0, h_l0],
            "detected_count": len(final_boxes_l0),
            "detected_boxes_l0": final_boxes_l0,
            "detected_polygons_l0": final_polygons_l0,
            "detected_scores": final_scores,
            "patch_records": final_patch_records,
            "gt_annots": gt_annots,
            "evaluation": evaluation_results,
        }

    def _evaluate_wsi(
        self,
        pred_boxes: List[List[float]],
        pred_polygons: List[List[Tuple[float, float]]],
        scores: List[float],
        gt_annots: List[ASAPAnnotation],
        iou_thresh: float = 0.40,
    ) -> Dict[str, Any]:
        """
        Evaluate WSI detection and segmentation against ground truth annotations.
        Computes TP, FP, FN, Precision, Recall, F1, and mean Dice / IoU on matched pairs.
        """
        gt_boxes = [[b[0], b[1], b[2], b[3]] for b in [a.bounds for a in gt_annots]]
        matched_gt = set()
        matched_pred = set()
        matched_pairs: List[Tuple[int, int]] = []

        # Sort predictions by confidence score descending
        order = np.argsort(scores)[::-1]

        tp = 0
        for p_idx in order:
            p_box = pred_boxes[p_idx]
            best_iou = 0.0
            best_g_idx = -1

            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = box_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= iou_thresh and best_g_idx >= 0:
                tp += 1
                matched_gt.add(best_g_idx)
                matched_pred.add(int(p_idx))
                matched_pairs.append((int(p_idx), int(best_g_idx)))

        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / len(gt_boxes)) if len(gt_boxes) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Compute fine polygon segmentation metrics (Dice & IoU) on True Positive pairs
        dices: List[float] = []
        ious: List[float] = []

        for p_idx, g_idx in matched_pairs:
            poly_pts = pred_polygons[p_idx]
            if len(poly_pts) >= 3:
                try:
                    p_poly = Polygon(poly_pts)
                    if not p_poly.is_valid:
                        p_poly = p_poly.buffer(0)
                    g_poly = gt_annots[g_idx].geometry

                    inter_a = p_poly.intersection(g_poly).area
                    union_a = p_poly.union(g_poly).area
                    pair_iou = float(inter_a / union_a) if union_a > 0 else 0.0
                    pair_dice = float(2.0 * inter_a / (p_poly.area + g_poly.area)) if (p_poly.area + g_poly.area) > 0 else 0.0

                    dices.append(pair_dice)
                    ious.append(pair_iou)
                except Exception:
                    pass

        mean_dice = round(float(np.mean(dices)), 4) if len(dices) > 0 else None
        mean_iou = round(float(np.mean(ious)), 4) if len(ious) > 0 else None

        return {
            "gt_count": int(len(gt_boxes)),
            "pred_count": int(len(pred_boxes)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "f1": float(round(f1, 4)),
            "mean_dice": float(round(mean_dice, 4)) if mean_dice is not None else None,
            "mean_iou": float(round(mean_iou, 4)) if mean_iou is not None else None,
            "count_error": int(abs(len(pred_boxes) - len(gt_boxes))),
            "matched_pairs": [[int(p), int(g)] for p, g in matched_pairs],
        }

    @staticmethod
    def export_asap_xml(
        polygons_l0: List[List[Tuple[float, float]]],
        boxes_l0: List[List[float]],
        output_xml_path: str,
        group_name: str = "Glomeruli_Predicted",
        color: str = "#00FF00",
    ) -> Path:
        """
        Export predicted glomeruli to ASAP XML format.
        Exports smooth Polygon boundaries if available, and bounding boxes in a secondary group.
        """
        out_path = Path(output_xml_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        doc = minidom.Document()
        asap_elem = doc.createElement("ASAP_Annotations")
        doc.appendChild(asap_elem)

        annotations_elem = doc.createElement("Annotations")
        asap_elem.appendChild(annotations_elem)

        has_polygons = len(polygons_l0) > 0 and len(polygons_l0[0]) > 4

        # Export primary annotations (Polygons if cascade, else Rectangles)
        for i, (poly, b) in enumerate(zip(polygons_l0, boxes_l0)):
            annot = doc.createElement("Annotation")
            annot.setAttribute("Name", f"Glomerulus_{i+1}")
            annot.setAttribute("PartOfGroup", group_name)
            annot.setAttribute("Color", color)

            coords_elem = doc.createElement("Coordinates")

            if has_polygons:
                annot.setAttribute("Type", "Polygon")
                for order, (cx, cy) in enumerate(poly):
                    coord = doc.createElement("Coordinate")
                    coord.setAttribute("Order", str(order))
                    coord.setAttribute("X", f"{cx:.2f}")
                    coord.setAttribute("Y", f"{cy:.2f}")
                    coords_elem.appendChild(coord)
            else:
                annot.setAttribute("Type", "Rectangle")
                corners = [(b[0], b[1]), (b[2], b[1]), (b[2], b[2]), (b[0], b[2])]
                for order, (cx, cy) in enumerate(corners):
                    coord = doc.createElement("Coordinate")
                    coord.setAttribute("Order", str(order))
                    coord.setAttribute("X", f"{cx:.2f}")
                    coord.setAttribute("Y", f"{cy:.2f}")
                    coords_elem.appendChild(coord)

            annot.appendChild(coords_elem)
            annotations_elem.appendChild(annot)

        # Groups declaration
        groups_elem = doc.createElement("AnnotationGroups")
        asap_elem.appendChild(groups_elem)

        group = doc.createElement("Group")
        group.setAttribute("Name", group_name)
        group.setAttribute("NumberOfAnnotations", str(len(boxes_l0)))
        group.setAttribute("Color", color)
        group.appendChild(doc.createElement("Attributes"))
        groups_elem.appendChild(group)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(doc.toprettyxml(indent="\t"))

        logger.info(f"Saved ASAP-compatible XML to: {out_path}")
        return out_path

    @staticmethod
    def generate_overview_plot(
        svs_path: str,
        pred_polygons_l0: List[List[Tuple[float, float]]],
        pred_boxes_l0: List[List[float]],
        gt_annots: Optional[List[ASAPAnnotation]] = None,
        output_path: str = "wsi_overview.png",
        max_thumb_dim: int = 2048,
        cascade_mode: bool = True,
        eval_metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Render whole-slide thumbnail with predicted annotations vs ground truth."""
        slide = openslide.OpenSlide(str(svs_path))
        w_l0, h_l0 = slide.dimensions

        thumb = slide.get_thumbnail((max_thumb_dim, max_thumb_dim))
        thumb_np = np.array(thumb.convert("RGB"))
        th, tw = thumb_np.shape[:2]

        scale_x = tw / float(w_l0)
        scale_y = th / float(h_l0)
        overlay = thumb_np.copy()

        # 1. Draw Ground Truth in Yellow (if available)
        if gt_annots:
            for ann in gt_annots:
                b = ann.bounds
                tx1 = int(b[0] * scale_x)
                ty1 = int(b[1] * scale_y)
                tx2 = int(b[2] * scale_x)
                ty2 = int(b[3] * scale_y)
                cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (255, 230, 0), 2)

        # 2. Draw Predictions in Vibrant Green
        has_polygons = cascade_mode and len(pred_polygons_l0) > 0 and len(pred_polygons_l0[0]) > 4
        for i, b in enumerate(pred_boxes_l0):
            tx1 = int(b[0] * scale_x)
            ty1 = int(b[1] * scale_y)
            tx2 = int(b[2] * scale_x)
            ty2 = int(b[3] * scale_y)

            if has_polygons:
                poly_pts = pred_polygons_l0[i]
                pts_thumb = np.array(
                    [[int(px * scale_x), int(py * scale_y)] for px, py in poly_pts], dtype=np.int32
                )
                if len(pts_thumb) > 2:
                    cv2.polylines(overlay, [pts_thumb], isClosed=True, color=(0, 255, 60), thickness=2)
            else:
                cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (0, 255, 60), 2)

        fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
        ax.imshow(overlay)

        slide_name = Path(svs_path).stem
        mode_str = "Cascade YOLO+U-Net" if cascade_mode else "YOLO Screening"
        metric_str = ""
        if eval_metrics:
            p = eval_metrics.get("precision", 0) * 100
            r = eval_metrics.get("recall", 0) * 100
            f1 = eval_metrics.get("f1", 0) * 100
            dice_str = f" | Dice: {eval_metrics['mean_dice']*100:.1f}%" if eval_metrics.get("mean_dice") else ""
            metric_str = f"\nPrec: {p:.1f}% | Rec: {r:.1f}% | F1: {f1:.1f}%{dice_str}"

        gt_count = len(gt_annots) if gt_annots else 0
        ax.set_title(
            f"Whole Slide Biopsy: {slide_name} [{mode_str}]\n"
            f"Ground Truth: {gt_count} (Yellow) | Predictions: {len(pred_boxes_l0)} (Green){metric_str}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.axis("off")

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_p, dpi=200)
        plt.close()
        slide.close()

        logger.info(f"Saved whole-slide overview visualization to: {out_p}")
        return out_p

    @staticmethod
    def generate_comparison_grid(
        patch_records: List[Dict[str, Any]],
        pred_polygons_l0: List[List[Tuple[float, float]]],
        pred_boxes_l0: List[List[float]],
        gt_annots: List[ASAPAnnotation],
        output_path: str = "wsi_comparison_grid.png",
        max_samples: int = 5,
        downsample: float = 2.0,
    ) -> Optional[Path]:
        """
        Generate a multi-panel visual comparison grid:
        Col 1: Raw Biopsy Patch (RGB)
        Col 2: Pathologist Ground Truth (Yellow Contour)
        Col 3: Cascade Model Prediction (Green Boundary + Box)
        Col 4: Direct Agreement & Error Overlay (Green=Agreement, Yellow=GT missed, Red=FP)
        """
        if not patch_records or not gt_annots:
            return None

        # Select up to max_samples patches with highest confidence detections
        valid_indices = [
            i for i in range(min(len(patch_records), len(pred_boxes_l0)))
            if patch_records[i].get("patch_img") is not None
        ]
        if not valid_indices:
            return None

        selected = valid_indices[:max_samples]
        n_rows = len(selected)

        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows), dpi=150)
        if n_rows == 1:
            axes = np.expand_dims(axes, 0)

        for row_idx, det_idx in enumerate(selected):
            rec = patch_records[det_idx]
            img = rec["patch_img"].copy()
            x0, y0 = rec["patch_coord"]
            H, W = img.shape[:2]

            # 1. Panel 1: Original Tissue Patch
            ax1 = axes[row_idx, 0]
            ax1.imshow(img)
            ax1.set_title("1. Biopsy Histology (20x)", fontsize=11, fontweight="bold")
            ax1.axis("off")

            # 2. Panel 2: Ground Truth Annotation Mask
            gt_mask = np.zeros((H, W), dtype=np.uint8)
            patch_box_l0 = box(x0, y0, x0 + W * downsample, y0 + H * downsample)

            for g in gt_annots:
                if patch_box_l0.intersects(g.geometry):
                    inter = g.geometry.intersection(patch_box_l0)
                    if not inter.is_empty:
                        # Draw into GT mask
                        if isinstance(inter, Polygon):
                            pts = np.array(
                                [[int((px - x0) / downsample), int((py - y0) / downsample)] for px, py in inter.exterior.coords],
                                dtype=np.int32
                            )
                            cv2.fillPoly(gt_mask, [pts], 255)

            gt_overlay = img.copy()
            gt_overlay[gt_mask > 0] = (gt_overlay[gt_mask > 0] * 0.45 + np.array([255, 230, 0]) * 0.55).astype(np.uint8)
            cnts_gt, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt_overlay, cnts_gt, -1, (255, 230, 0), 3)

            ax2 = axes[row_idx, 1]
            ax2.imshow(gt_overlay)
            ax2.set_title("2. Pathologist Ground Truth (Gold)", fontsize=11, fontweight="bold")
            ax2.axis("off")

            # 3. Panel 3: Cascade Predicted Segmentation
            pred_mask = np.zeros((H, W), dtype=np.uint8)
            if rec.get("patch_contour") is not None:
                cv2.fillPoly(pred_mask, [rec["patch_contour"]], 255)
            else:
                bx1, by1, bx2, by2 = rec["patch_bbox"]
                cv2.rectangle(pred_mask, (bx1, by1), (bx2, by2), 255, -1)

            pred_overlay = img.copy()
            pred_overlay[pred_mask > 0] = (pred_overlay[pred_mask > 0] * 0.45 + np.array([0, 255, 120]) * 0.55).astype(np.uint8)
            cnts_pred, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pred_overlay, cnts_pred, -1, (0, 255, 60), 3)

            # Draw YOLO bounding box in Red
            bx1, by1, bx2, by2 = rec["patch_bbox"]
            cv2.rectangle(pred_overlay, (bx1, by1), (bx2, by2), (255, 50, 50), 2)
            cv2.putText(
                pred_overlay, f"Score: {rec['score']:.2f}",
                (bx1, max(25, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2
            )

            ax3 = axes[row_idx, 2]
            ax3.imshow(pred_overlay)
            ax3.set_title("3. Cascade Prediction (Green: Tuft, Red: Box)", fontsize=11, fontweight="bold")
            ax3.axis("off")

            # 4. Panel 4: Direct Agreement & Discrepancy Overlay
            # Agreement = Green | GT Missed (FN) = Yellow | Prediction Excess (FP) = Red
            diff_img = img.copy()
            tp_mask = (gt_mask > 0) & (pred_mask > 0)
            fn_mask = (gt_mask > 0) & (pred_mask == 0)
            fp_mask = (gt_mask == 0) & (pred_mask > 0)

            diff_img[tp_mask] = (diff_img[tp_mask] * 0.40 + np.array([0, 255, 60]) * 0.60).astype(np.uint8)
            diff_img[fn_mask] = (diff_img[fn_mask] * 0.40 + np.array([255, 230, 0]) * 0.60).astype(np.uint8)
            diff_img[fp_mask] = (diff_img[fp_mask] * 0.40 + np.array([255, 40, 40]) * 0.60).astype(np.uint8)

            # Calculate local Dice & IoU
            tp_px = np.count_nonzero(tp_mask)
            fp_px = np.count_nonzero(fp_mask)
            fn_px = np.count_nonzero(fn_mask)
            local_dice = (2.0 * tp_px) / (2.0 * tp_px + fp_px + fn_px + 1e-7)
            local_iou = tp_px / (tp_px + fp_px + fn_px + 1e-7)

            ax4 = axes[row_idx, 4 - 1]
            ax4.imshow(diff_img)
            ax4.set_title(
                f"4. Overlap (Green: Agree, Yellow: FN, Red: FP)\nDice: {local_dice:.3f} | IoU: {local_iou:.3f}",
                fontsize=11,
                fontweight="bold",
            )
            ax4.axis("off")

        plt.suptitle("Qualitative Glomerular Segmentation & Detection Comparison Grid", fontsize=15, fontweight="bold", y=0.99)
        plt.tight_layout()
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=180, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved qualitative comparison grid to: {out_p}")
        return out_p
