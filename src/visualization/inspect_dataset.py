"""
Dataset inspection and visual quality control utilities.
"""

from pathlib import Path
from typing import Optional, List
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_single_patch(
    image_path: Path,
    mask_path: Optional[Path] = None,
    label_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Render a single patch with mask overlay (green) and YOLO bounding box (red).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    h, w = img.shape[:2]
    overlay = img.copy()

    # 1. Overlay segmentation mask
    if mask_path is not None and Path(mask_path).exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Colorize mask with green tint
            green_layer = np.zeros_like(img)
            green_layer[:, :] = [0, 255, 0]
            mask_bool = mask > 0
            overlay[mask_bool] = (overlay[mask_bool] * 0.5 + green_layer[mask_bool] * 0.5).astype(np.uint8)

            # Draw contour boundary in yellow
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    # 2. Draw YOLO bounding boxes
    if label_path is not None and Path(label_path).exists():
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, cx_f, cy_f, bw_f, bh_f = [float(x) for x in parts[:5]]
                bx1 = int((cx_f - bw_f / 2.0) * w)
                by1 = int((cy_f - bh_f / 2.0) * h)
                bx2 = int((cx_f + bw_f / 2.0) * w)
                by2 = int((cy_f + bh_f / 2.0) * h)
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

    return overlay


def create_inspection_grid(
    manifest_csv: str = "dataset/manifest.csv",
    output_png: str = "dataset/inspection_grid.png",
    num_samples: int = 6,
) -> Path:
    """
    Generate a visual verification grid of extracted patches with labels and masks.
    """
    manifest_path = Path(manifest_csv)
    dataset_root = manifest_path.parent

    df = pd.read_csv(manifest_path)

    pos_df = df[df["is_positive"] == True]
    neg_df = df[df["is_positive"] == False]

    # Sample equal positives and negatives
    n_pos = min(num_samples // 2, len(pos_df))
    n_neg = min(num_samples - n_pos, len(neg_df))

    sample_pos = pos_df.sample(n=n_pos, random_state=42)
    sample_neg = neg_df.sample(n=n_neg, random_state=42)
    sample_df = pd.concat([sample_pos, sample_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        if idx >= len(axes):
            break

        img_p = dataset_root / row["image_path"]
        mask_p = dataset_root / row["mask_path"]
        lbl_p = dataset_root / row["label_path"]

        vis_bgr = visualize_single_patch(img_p, mask_p, lbl_p)
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

        ax = axes[idx]
        ax.imshow(vis_rgb)
        status = f"POSITIVE (Gloms: {row['num_glomeruli']})" if row["is_positive"] else "NEGATIVE (Tissue only)"
        ax.set_title(
            f"{row['patch_id']}\nSlide: {row['slide_id']} | Split: {row['split']}\n{status}",
            fontsize=10
        )
        ax.axis("off")

    plt.tight_layout()
    out_path = Path(output_png)
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path
