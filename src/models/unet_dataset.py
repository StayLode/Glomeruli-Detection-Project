"""
PyTorch Dataset for U-Net Glomeruli Segmentation.

Handles image-mask pairs with:
- Synchronized geometric augmentations (flips, 90-deg rotations, scaling)
- Photometric stain augmentations (brightness, contrast, saturation, hue)
- Support for both full patches (1024x1024) and focused glomeruli crops
- Fast OpenCV I/O with tensor normalization
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class UNetDataset(Dataset):
    """
    Dataset loading histology patches and binary glomerular segmentation masks.
    """

    def __init__(
        self,
        manifest_path: str = "dataset/manifest.csv",
        split: str = "train",
        img_size: int = 1024,
        augment: bool = True,
        color_jitter: Optional[Dict[str, float]] = None,
        positive_only: bool = False,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            manifest_path: Path to dataset manifest CSV.
            split: Dataset split ('train', 'val', 'test').
            img_size: Target square image dimension (e.g. 1024 or 512).
            augment: Whether to apply data augmentation (active only for train).
            color_jitter: Dict with brightness, contrast, saturation, hue limits.
            positive_only: If True, filters only patches containing glomeruli.
        """
        self.manifest_path = Path(manifest_path)
        self.dataset_root = self.manifest_path.parent
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.color_jitter = color_jitter

        df = pd.read_csv(self.manifest_path)
        split_data = df[df["split"] == split].reset_index(drop=True)

        if positive_only:
            split_data = split_data[split_data["is_positive"] == True].reset_index(drop=True)

        self.data = split_data

        # Standard ImageNet normalization tensors
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.data.iloc[idx]

        img_path = self.dataset_root / row["image_path"]
        mask_path = self.dataset_root / row["mask_path"]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            mask_gray = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

        # Resize if dimensions differ from target img_size
        h, w = img_rgb.shape[:2]
        if h != self.img_size or w != self.img_size:
            img_rgb = cv2.resize(img_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            mask_gray = cv2.resize(mask_gray, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Convert to float PyTorch tensors
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy((mask_gray > 127).astype(np.float32)).unsqueeze(0)

        # Apply synchronized data augmentations for training
        if self.augment:
            img_tensor, mask_tensor = self._apply_augmentations(img_tensor, mask_tensor)

        # Apply ImageNet normalization to image
        normalized_img = (img_tensor - self.mean) / self.std

        meta = {
            "patch_id": row["patch_id"],
            "slide_id": row["slide_id"],
            "num_glomeruli": int(row["num_glomeruli"]),
            "is_positive": bool(row["is_positive"]),
        }

        return normalized_img, mask_tensor, meta

    def _apply_augmentations(
        self, img: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply joint geometric and color augmentations."""
        # 1. Random Horizontal Flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 2. Random Vertical Flip
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # 3. Random Orthogonal Rotations (0, 90, 180, 270 deg)
        rot_k = random.randint(0, 3)
        if rot_k > 0:
            img = torch.rot90(img, rot_k, [1, 2])
            mask = torch.rot90(mask, rot_k, [1, 2])

        # 4. Color Jitter on image only (Stain Variation Simulation)
        if self.color_jitter:
            b = self.color_jitter.get("brightness", 0.0)
            c = self.color_jitter.get("contrast", 0.0)
            s = self.color_jitter.get("saturation", 0.0)
            h = self.color_jitter.get("hue", 0.0)

            # Brightness
            if b > 0:
                factor = 1.0 + random.uniform(-b, b)
                img = torch.clamp(img * factor, 0.0, 1.0)
            # Contrast
            if c > 0:
                factor = 1.0 + random.uniform(-c, c)
                mean_val = img.mean()
                img = torch.clamp((img - mean_val) * factor + mean_val, 0.0, 1.0)
            # Saturation / Hue via torchvision if available
            if s > 0:
                sat_factor = 1.0 + random.uniform(-s, s)
                img = TF.adjust_saturation(img, sat_factor)
            if h > 0:
                hue_factor = random.uniform(-h, h)
                img = TF.adjust_hue(img, hue_factor)

        return img, mask


def extract_glomerulus_crop(
    image: np.ndarray,
    bbox: List[int],
    margin_ratio: float = 0.25,
    target_size: int = 512,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract a square context-preserving crop centered on a detected glomerulus.
    Bridges detection (YOLO) and fine segmentation / clustering (Step 2 & 3).

    Args:
        image: Full image or patch (H, W, 3).
        bbox: [x1, y1, x2, y2] bounding box.
        margin_ratio: Context padding ratio around the glomerulus.
        target_size: Target square output resolution (e.g. 512).

    Returns:
        cropped_image: Resized square image (target_size, target_size, 3).
        crop_coords: (crop_x1, crop_y1, crop_x2, crop_y2) in original image coordinates.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    side = max(w, h) * (1.0 + margin_ratio)
    half = side / 2.0

    img_h, img_w = image.shape[:2]
    crop_x1 = max(0, int(round(cx - half)))
    crop_y1 = max(0, int(round(cy - half)))
    crop_x2 = min(img_w, int(round(cx + half)))
    crop_y2 = min(img_h, int(round(cy + half)))

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        crop = np.zeros((target_size, target_size, 3), dtype=image.dtype)
    else:
        crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

    return crop, (crop_x1, crop_y1, crop_x2, crop_y2)
