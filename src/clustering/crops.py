"""Glomerulus crop extraction with explicit legacy and padded-square behavior."""

from typing import Tuple

import cv2
import numpy as np


def square_padded_crop(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    target_size: int,
    margin_ratio: float = 0.25,
    padding_mode: str = "reflect",
    is_mask: bool = False,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
    """Extract a centered square without anisotropic scaling.

    Returns the resized square, requested source bounds (which may leave the
    image), and source-pixel side length for physical-unit conversion.
    """
    if padding_mode not in {"reflect", "background"}:
        raise ValueError("padding_mode must be 'reflect' or 'background'")
    x1, y1, x2, y2 = map(int, bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive width and height")
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + margin_ratio)
    side = max(1, int(round(side)))
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right, bottom = left + side, top + side

    height, width = image.shape[:2]
    crop_left, crop_top = max(0, left), max(0, top)
    crop_right, crop_bottom = min(width, right), min(height, bottom)
    crop = image[crop_top:crop_bottom, crop_left:crop_right]
    pad_left, pad_top = crop_left - left, crop_top - top
    pad_right, pad_bottom = right - crop_right, bottom - crop_bottom
    if is_mask:
        border_type, border_value = cv2.BORDER_CONSTANT, 0
    elif padding_mode == "reflect" and min(crop.shape[:2], default=0) > 1:
        border_type, border_value = cv2.BORDER_REFLECT_101, 0
    else:
        border_type, border_value = cv2.BORDER_CONSTANT, (255, 255, 255)
    crop = cv2.copyMakeBorder(
        crop, pad_top, pad_bottom, pad_left, pad_right, border_type, value=border_value
    )
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    crop = cv2.resize(crop, (target_size, target_size), interpolation=interpolation)
    return crop, (left, top, right, bottom), float(side)
