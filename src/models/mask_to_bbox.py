"""
Module for extracting bounding box detections from semantic segmentation masks.

Enables direct, apples-to-apples comparison between U-Net segmentation
and YOLO object detection metrics.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


def extract_bboxes_from_mask(
    mask_binary: np.ndarray,
    prob_map: np.ndarray,
    min_area: int = 400
) -> List[Dict[str, Any]]:
    """
    Extract bounding boxes and confidence scores from a predicted binary mask.

    Args:
        mask_binary: 2D uint8 array (0 or 255) representing binary prediction.
        prob_map: 2D float array in [0, 1] containing raw sigmoid probabilities.
        min_area: Minimum pixel area to consider a valid glomerular detection.

    Returns:
        List of dictionaries with keys:
            'bbox': [x1, y1, x2, y2] in pixel coordinates
            'score': float confidence score (mean probability inside blob)
            'area': pixel area of the blob
    """
    if mask_binary.dtype != np.uint8:
        mask_binary = (mask_binary > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Compute average probability within the detected contour
        blob_mask = np.zeros(mask_binary.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
        mean_score = float(np.mean(prob_map[blob_mask == 255])) if np.any(blob_mask == 255) else 0.5

        detections.append({
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "score": round(mean_score, 4),
            "area": float(area),
        })

    # Sort detections by confidence score descending
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


def box_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def evaluate_detection_metrics(
    all_pred_boxes: List[List[Dict[str, Any]]],
    all_gt_boxes: List[List[List[float]]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate object detection metrics (P, R, F1, AP50) across a dataset.

    Args:
        all_pred_boxes: List of predicted detections per image.
        all_gt_boxes: List of ground truth [x1, y1, x2, y2] boxes per image.
        iou_threshold: IoU cutoff for true positive assignment.

    Returns:
        Dictionary containing 'precision', 'recall', 'f1', and 'ap50'.
    """
    total_gt = sum(len(gts) for gts in all_gt_boxes)
    total_preds = sum(len(preds) for preds in all_pred_boxes)

    if total_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap50": 0.0}

    # Flatten predictions with image index for global PR curve calculation
    flat_preds = []
    for img_idx, preds in enumerate(all_pred_boxes):
        for p in preds:
            flat_preds.append((p["score"], img_idx, p["bbox"]))

    # Sort all detections globally by confidence descending
    flat_preds.sort(key=lambda x: x[0], reverse=True)

    # Track matched GT boxes per image
    matched_gt = [set() for _ in range(len(all_gt_boxes))]

    tp = np.zeros(len(flat_preds))
    fp = np.zeros(len(flat_preds))

    for det_idx, (score, img_idx, p_box) in enumerate(flat_preds):
        gt_boxes = all_gt_boxes[img_idx]
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx in matched_gt[img_idx]:
                continue
            iou = box_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[det_idx] = 1.0
            matched_gt[img_idx].add(best_gt_idx)
        else:
            fp[det_idx] = 1.0

    # Compute cumulative precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / float(total_gt)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(float).eps)

    # Compute 11-point interpolated Average Precision (AP50)
    ap50 = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        prec_at_rec = precisions[recalls >= t]
        p_val = np.max(prec_at_rec) if len(prec_at_rec) > 0 else 0.0
        ap50 += p_val / 11.0

    final_tp = int(tp.sum())
    final_fp = int(fp.sum())
    final_fn = total_gt - final_tp

    final_precision = float(final_tp / (final_tp + final_fp)) if (final_tp + final_fp) > 0 else 0.0
    final_recall = float(final_tp / total_gt) if total_gt > 0 else 0.0
    final_f1 = (
        float(2.0 * final_precision * final_recall / (final_precision + final_recall))
        if (final_precision + final_recall) > 0
        else 0.0
    )

    return {
        "precision": round(final_precision, 4),
        "recall": round(final_recall, 4),
        "f1": round(final_f1, 4),
        "ap50": round(float(ap50), 4),
        "total_gt": total_gt,
        "total_preds": total_preds,
    }
