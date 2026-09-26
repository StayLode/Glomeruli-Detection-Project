"""
Whole Slide Image (WSI) Inference and Stitching Engine.

Features:
- Automated tissue detection to process only tissue areas (~5% of the slide).
- Batched inference across overlapping 20x patches.
- Projection of local detections back to native Level 0 coordinates.
- Global Non-Maximum Suppression (NMS) across tile boundaries.
- Evaluation against ground truth XML (True Positives, False Positives, False Negatives).
- Export to ASAP-compliant XML for pathologist review.
- Publication-quality whole-slide visualization.
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
from shapely.geometry import box
import torch
import torchvision
from tqdm import tqdm
from ultralytics import YOLO

from src.models.mask_to_bbox import box_iou
from src.preprocessing.tissue_detector import TissueDetector
from src.utils.xml_parser import ASAPAnnotation, parse_asap_xml

logger = logging.getLogger(__name__)


class WSIInferenceEngine:
    """Performs whole slide inference, stitching, and evaluation for glomeruli detection."""

    def __init__(
        self,
        model_weights_path: str,
        patch_size: int = 1024,
        stride: int = 768,
        target_mag: int = 20,
        base_mag: int = 40,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_weights_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading YOLO model from {self.model_path} on {self.device}...")
        self.model = YOLO(str(self.model_path))

        self.patch_size = patch_size
        self.stride = stride
        self.downsample = base_mag / float(target_mag)
        self.patch_size_l0 = int(round(patch_size * self.downsample))
        self.stride_l0 = int(round(stride * self.downsample))

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

    def predict_slide(
        self,
        svs_path: Union[str, Path],
        xml_path: Optional[Union[str, Path]] = None,
        conf_thresh: float = 0.25,
        nms_iou_thresh: float = 0.40,
        batch_size: int = 8,
        min_tissue_ratio: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Run whole-slide inference on an SVS file with global stitching.

        Args:
            svs_path: Path to .svs Whole Slide Image.
            xml_path: Optional path to ASAP ground truth XML for accuracy evaluation.
            conf_thresh: Confidence threshold for YOLO predictions.
            nms_iou_thresh: IoU cutoff for Global Non-Maximum Suppression.
            batch_size: Number of patches per GPU inference batch.
            min_tissue_ratio: Minimum fraction of tissue required to process a patch.

        Returns:
            Dictionary containing global detections, metrics, and slide metadata.
        """
        svs_path = Path(svs_path)
        slide_id = svs_path.stem
        logger.info(f"Starting WSI inference for: {slide_id}")

        slide = openslide.OpenSlide(str(svs_path))
        w_l0, h_l0 = slide.dimensions

        # 1. Segment tissue on overview
        detector = TissueDetector(slide, thumbnail_size=1024, max_v_thresh=238)
        coords = self._plan_tissue_patches(slide, detector, min_tissue_ratio)
        logger.info(f"Planned {len(coords):,} tissue patches to process (skipped {100*(1-detector.tissue_ratio):.1f}% empty glass).")

        raw_boxes_l0: List[List[float]] = []
        raw_scores: List[float] = []

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

            # Predict on batch
            results = self.model.predict(
                source=batch_imgs,
                conf=conf_thresh,
                imgsz=self.patch_size,
                verbose=False,
                device=self.device,
            )

            # Project bounding boxes back to Level 0 coordinates
            for (x0, y0), res in zip(batch_coords, results):
                if res.boxes is not None and len(res.boxes) > 0:
                    boxes_xyxy_norm = res.boxes.xyxyn.cpu().numpy()  # [x1, y1, x2, y2] in [0, 1]
                    confs = res.boxes.conf.cpu().numpy()

                    for b_norm, conf in zip(boxes_xyxy_norm, confs):
                        x1_l0 = x0 + b_norm[0] * self.patch_size_l0
                        y1_l0 = y0 + b_norm[1] * self.patch_size_l0
                        x2_l0 = x0 + b_norm[2] * self.patch_size_l0
                        y2_l0 = y0 + b_norm[3] * self.patch_size_l0

                        raw_boxes_l0.append([float(x1_l0), float(y1_l0), float(x2_l0), float(y2_l0)])
                        raw_scores.append(float(conf))

        logger.info(f"Extracted {len(raw_boxes_l0)} raw candidate detections across all patches.")

        # 3. Global Non-Maximum Suppression across overlapping boundaries
        final_boxes_l0: List[List[float]] = []
        final_scores: List[float] = []

        if len(raw_boxes_l0) > 0:
            boxes_t = torch.tensor(raw_boxes_l0, dtype=torch.float32)
            scores_t = torch.tensor(raw_scores, dtype=torch.float32)
            keep_indices = torchvision.ops.nms(boxes_t, scores_t, nms_iou_thresh).tolist()

            for k in keep_indices:
                final_boxes_l0.append(raw_boxes_l0[k])
                final_scores.append(raw_scores[k])

        logger.info(f"Post-NMS: {len(final_boxes_l0)} unique glomeruli detected on whole slide.")

        # 4. Optional Ground Truth Evaluation
        evaluation_results: Optional[Dict[str, Any]] = None
        gt_boxes_l0: List[List[float]] = []

        if xml_path is not None and Path(xml_path).exists():
            gt_annots = parse_asap_xml(xml_path)
            gt_boxes_l0 = [[b[0], b[1], b[2], b[3]] for b in [a.bounds for a in gt_annots]]
            evaluation_results = self._evaluate_wsi(final_boxes_l0, final_scores, gt_boxes_l0, iou_thresh=0.40)

            logger.info("=" * 55)
            logger.info(f"WSI CLINICAL EVALUATION: {slide_id}")
            logger.info(f"   Ground Truth Glomeruli : {evaluation_results['gt_count']}")
            logger.info(f"   Predicted Glomeruli    : {evaluation_results['pred_count']}")
            logger.info(f"   True Positives (TP)    : {evaluation_results['tp']}")
            logger.info(f"   False Positives (FP)   : {evaluation_results['fp']}")
            logger.info(f"   False Negatives (FN)   : {evaluation_results['fn']}")
            logger.info(f"   Slide Precision        : {evaluation_results['precision']:.4f}")
            logger.info(f"   Slide Recall           : {evaluation_results['recall']:.4f}")
            logger.info(f"   Slide F1-Score         : {evaluation_results['f1']:.4f}")
            logger.info(f"   Absolute Count Error   : {evaluation_results['count_error']}")
            logger.info("=" * 55)

        slide.close()

        return {
            "slide_id": slide_id,
            "svs_path": str(svs_path),
            "level0_dimensions": [w_l0, h_l0],
            "detected_count": len(final_boxes_l0),
            "detected_boxes_l0": final_boxes_l0,
            "detected_scores": final_scores,
            "gt_boxes_l0": gt_boxes_l0,
            "evaluation": evaluation_results,
        }

    def _evaluate_wsi(
        self,
        pred_boxes: List[List[float]],
        scores: List[float],
        gt_boxes: List[List[float]],
        iou_thresh: float = 0.40,
    ) -> Dict[str, Any]:
        """Match predicted WSI boxes to ground truth boxes using IoU thresholding."""
        matched_gt = set()
        matched_pred = set()

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
                matched_pred.add(p_idx)

        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / len(gt_boxes)) if len(gt_boxes) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "gt_count": len(gt_boxes),
            "pred_count": len(pred_boxes),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "count_error": abs(len(pred_boxes) - len(gt_boxes)),
        }

    @staticmethod
    def export_asap_xml(
        predictions: List[List[float]],
        output_xml_path: str,
        group_name: str = "Glomeruli_Predicted",
        color: str = "#00FF00"
    ) -> Path:
        """
        Export predicted bounding boxes to ASAP XML format for clinical inspection in ASAP viewer.
        """
        out_path = Path(output_xml_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        doc = minidom.Document()
        asap_elem = doc.createElement("ASAP_Annotations")
        doc.appendChild(asap_elem)

        annotations_elem = doc.createElement("Annotations")
        asap_elem.appendChild(annotations_elem)

        for i, (x1, y1, x2, y2) in enumerate(predictions):
            annot = doc.createElement("Annotation")
            annot.setAttribute("Name", f"Glomerulus_{i+1}")
            annot.setAttribute("Type", "Rectangle")
            annot.setAttribute("PartOfGroup", group_name)
            annot.setAttribute("Color", color)

            coords_elem = doc.createElement("Coordinates")
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            for order, (cx, cy) in enumerate(corners):
                coord = doc.createElement("Coordinate")
                coord.setAttribute("Order", str(order))
                coord.setAttribute("X", f"{cx:.2f}")
                coord.setAttribute("Y", f"{cy:.2f}")
                coords_elem.appendChild(coord)

            annot.appendChild(coords_elem)
            annotations_elem.appendChild(annot)

        # Groups element
        groups_elem = doc.createElement("AnnotationGroups")
        asap_elem.appendChild(groups_elem)

        group = doc.createElement("Group")
        group.setAttribute("Name", group_name)
        group.setAttribute("NumberOfAnnotations", str(len(predictions)))
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
        pred_boxes_l0: List[List[float]],
        gt_boxes_l0: Optional[List[List[float]]] = None,
        output_path: str = "wsi_overview.png",
        max_thumb_dim: int = 2048,
    ) -> Path:
        """
        Render a high-resolution overview image showing all detected glomeruli on the whole biopsy.
        """
        slide = openslide.OpenSlide(str(svs_path))
        w_l0, h_l0 = slide.dimensions

        thumb = slide.get_thumbnail((max_thumb_dim, max_thumb_dim))
        thumb_np = np.array(thumb.convert("RGB"))
        th, tw = thumb_np.shape[:2]

        scale_x = tw / float(w_l0)
        scale_y = th / float(h_l0)

        overlay = thumb_np.copy()

        # 1. Draw Ground Truth boxes in Yellow (if available)
        if gt_boxes_l0 is not None:
            for gx1, gy1, gx2, gy2 in gt_boxes_l0:
                tx1 = int(gx1 * scale_x)
                ty1 = int(gy1 * scale_y)
                tx2 = int(gx2 * scale_x)
                ty2 = int(gy2 * scale_y)
                cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (255, 255, 0), 2)

        # 2. Draw Predicted boxes in Green
        for px1, py1, px2, py2 in pred_boxes_l0:
            tx1 = int(px1 * scale_x)
            ty1 = int(py1 * scale_y)
            tx2 = int(px2 * scale_x)
            ty2 = int(py2 * scale_y)
            cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)

        # Title & legend
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(overlay)
        slide_name = Path(svs_path).stem
        gt_info = f" | GT Count: {len(gt_boxes_l0)} (Yellow)" if gt_boxes_l0 else ""
        ax.set_title(
            f"Whole Slide Glomeruli Detection: {slide_name}\n"
            f"Predicted Detections: {len(pred_boxes_l0)} (Green){gt_info}",
            fontsize=12,
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
