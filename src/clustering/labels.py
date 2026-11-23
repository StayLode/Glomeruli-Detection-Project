"""Fraction of bright, low-saturation pixels in an inner peripheral mask band.

Used strictly as an experimental post-hoc proxy to order cluster letters (A, B, ...)
on the training split; it is not a clustering feature and not a clinical measurement.
"""


import numpy as np
import cv2

RIM_PARAMETERS = {"relative_band_width": 0.15, "min_value": 0.80, "max_saturation": 0.20}
LABEL_LIMITATION = (
    "Experimental ordering only: A has the highest train-cluster median peripheral clear fraction and "
    "the final letter has the lowest. The letters are shown as a low-to-high necrosis-compatibility "
    "proxy for visualization; this is not a measurement of Bowman's space or a validated clinical grade."
)


def peripheral_clear_fraction(image, mask, relative_band_width=.15, min_value=.80, max_saturation=.20):
    """Fraction of bright, low-saturation pixels in an inner peripheral mask band.

    Thresholds are fixed before the new training cycle; color/stain/artifact effects
    remain possible. No new feature is added to the PCA20 representation.
    """
    image = np.asarray(image)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if image.shape[:2] != binary.shape or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image and mask must align")
    area = int(binary.sum())
    if area == 0:
        return float("nan")
    radius = np.sqrt(area / np.pi)
    # Pad so distanceTransform also has a background for masks touching all edges.
    distance = cv2.distanceTransform(np.pad(binary, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    band = (binary > 0) & (distance <= max(1., relative_band_width * radius))
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    clear = (hsv[..., 2] >= min_value * 255) & (hsv[..., 1] <= max_saturation * 255)
    return float(clear[band].mean()) if band.any() else float("nan")


def fit_label_mapping(labels, scores, k):
    """Order by train median clear fraction, from A (highest) to last (lowest)."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    medians = {}
    for cluster in range(k):
        values = scores[(labels == cluster) & np.isfinite(scores)]
        if not len(values):
            raise ValueError(f"Cannot order cluster {cluster}: no measurable train rim scores")
        medians[cluster] = float(np.median(values))
    ordered = sorted(medians, key=lambda cluster: (-medians[cluster], cluster))
    mapping = {cluster: chr(ord("A") + rank) for rank, cluster in enumerate(ordered)}
    return mapping, medians
