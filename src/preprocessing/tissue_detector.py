"""
Tissue detector module for Whole Slide Images (WSI).

Computes a binary tissue mask from low-resolution thumbnails to filter out
the ~95% empty glass background before high-resolution patch extraction.
"""

from typing import Tuple, Union
import cv2
import numpy as np
import openslide


class TissueDetector:
    """Detects tissue regions in a Whole Slide Image using color thresholding."""

    def __init__(
        self,
        slide: openslide.OpenSlide,
        thumbnail_size: int = 1024,
        max_v_thresh: int = 238,
        morph_kernel_size: int = 5,
    ) -> None:
        """
        Initialize the tissue detector.

        Args:
            slide: OpenSlide object.
            thumbnail_size: Maximum dimension for thumbnail image.
            max_v_thresh: Maximum Value (brightness) threshold in HSV to exclude white glass.
            morph_kernel_size: Size of kernel for morphological operations.
        """
        self.slide = slide
        self.w_l0, self.h_l0 = slide.dimensions

        # Generate thumbnail
        thumb = slide.get_thumbnail((thumbnail_size, thumbnail_size))
        self.thumb_rgb = np.array(thumb.convert("RGB"))
        self.th, self.tw = self.thumb_rgb.shape[:2]

        # Scale factors to convert Level 0 coords -> thumbnail coords
        self.scale_x = self.tw / float(self.w_l0)
        self.scale_y = self.th / float(self.h_l0)

        # Compute tissue mask
        self.mask = self._segment_tissue(max_v_thresh, morph_kernel_size)
        self.tissue_ratio = float(np.count_nonzero(self.mask)) / float(self.mask.size)

    def _segment_tissue(self, max_v_thresh: int, kernel_size: int) -> np.ndarray:
        """Compute morphological binary tissue mask from thumbnail."""
        hsv = cv2.cvtColor(self.thumb_rgb, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        # Otsu threshold on saturation channel
        _, thresh_s = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Threshold on brightness (exclude pure white / near-white glass)
        mask_v = (v_channel < max_v_thresh).astype(np.uint8) * 255

        # Intersect
        combined = cv2.bitwise_and(thresh_s, mask_v)

        # Morphological closing to fill small holes inside tubules/tissue
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Morphological opening to eliminate dust specks on glass
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        return cleaned

    def get_tissue_ratio(self, x0: int, y0: int, w: int, h: int) -> float:
        """
        Calculate the tissue coverage ratio inside a Level 0 bounding box.

        Args:
            x0: Top-left X coordinate at Level 0.
            y0: Top-left Y coordinate at Level 0.
            w: Width at Level 0.
            h: Height at Level 0.

        Returns:
            Float in [0.0, 1.0] representing the fraction of tissue.
        """
        tx0 = int(x0 * self.scale_x)
        ty0 = int(y0 * self.scale_y)
        tx1 = int((x0 + w) * self.scale_x)
        ty1 = int((y0 + h) * self.scale_y)

        # Clamp to thumbnail bounds
        tx0 = max(0, min(tx0, self.tw - 1))
        tx1 = max(0, min(tx1, self.tw))
        ty0 = max(0, min(ty0, self.th - 1))
        ty1 = max(0, min(ty1, self.th))

        submask = self.mask[ty0:ty1, tx0:tx1]
        if submask.size == 0:
            return 0.0

        return float(np.count_nonzero(submask)) / float(submask.size)

    def is_tissue_patch(self, x0: int, y0: int, w: int, h: int, min_ratio: float = 0.15) -> bool:
        """Return True if the patch contains at least `min_ratio` tissue."""
        return self.get_tissue_ratio(x0, y0, w, h) >= min_ratio
