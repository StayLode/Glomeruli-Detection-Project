"""
Feature extraction module for glomeruli crops.

Supports:
1. Stain-Invariant Morphological & Structural Features (default final model):
   7 physical/geometric and internal texture descriptors immune to slide-level
   staining and illumination shifts.
2. Full 10-feature Native Morphological & Photometric Dataset (for shared
   evaluation space and descriptive profiling).
3. Deep / Hybrid Representations via pre-trained ResNet backbones (retained for
   baseline comparison).
"""

from typing import Dict, Any, List, Optional, Sequence, Tuple
import hashlib
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

CANONICAL_MORPH_FEATURES: List[str] = [
    "glom_area_um2",
    "glom_perimeter_um",
    "circularity",
    "solidity",
    "extent",
    "aspect_ratio",
    "mean_intensity",
    "mean_optical_density",
    "stain_heterogeneity",
    "red_green_ratio",
]

STAIN_INVARIANT_MORPH_FEATURES: List[str] = [
    "glom_area_um2",
    "glom_perimeter_um",
    "circularity",
    "solidity",
    "extent",
    "aspect_ratio",
    "stain_heterogeneity",
]


def select_morphological_columns(
    morph: np.ndarray,
    morph_names: Sequence[str],
    selected_features: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Select an ordered subset of morphological feature columns."""
    selected = list(selected_features or STAIN_INVARIANT_MORPH_FEATURES)
    if not selected:
        raise ValueError("selected_features cannot be empty")
    matrix = np.asarray(morph, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("morph feature matrix must be 2D")
    name_to_idx = {str(name): idx for idx, name in enumerate(morph_names)}
    missing = [name for name in selected if name not in name_to_idx]
    if missing:
        raise ValueError(f"Missing required morphological features: {missing}")
    indices = [name_to_idx[name] for name in selected]
    return matrix[:, indices].astype(np.float32), selected


def extract_single_morphological_features(
    crop_rgb: np.ndarray,
    mask_binary: np.ndarray,
    morph_units: str = "legacy_resized",
    pixel_size_um: float = 1.0,
) -> Dict[str, float]:
    """Extract histopathological morphological and photometric features for a single glomerulus."""
    if morph_units not in {"legacy_resized", "native"}:
        raise ValueError(f"Unsupported morph_units: {morph_units}")
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3 or mask_binary.shape != crop_rgb.shape[:2]:
        raise ValueError("crop_rgb must be HxWx3 and align with the HxW mask")
    mask_binary = (mask_binary > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        area_name = "glom_area_um2" if morph_units == "native" else "glom_area"
        perimeter_name = "glom_perimeter_um" if morph_units == "native" else "glom_perimeter"
        return {
            area_name: 0.0,
            perimeter_name: 0.0,
            "circularity": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "aspect_ratio": 1.0,
            "mean_intensity": 0.0,
            "mean_optical_density": 0.0,
            "stain_heterogeneity": 0.0,
            "red_green_ratio": 1.0,
        }

    cnt = max(contours, key=cv2.contourArea)
    area_px = float(cv2.contourArea(cnt))
    perimeter_px = float(cv2.arcLength(cnt, True))
    area = area_px
    perimeter = perimeter_px
    if morph_units == "native":
        area *= pixel_size_um ** 2
        perimeter *= pixel_size_um

    circularity = float(4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    circularity = min(1.0, max(0.0, circularity))

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area_px / hull_area) if hull_area > 0 else 0.0

    _, _, bw, bh = cv2.boundingRect(cnt)
    aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0
    extent = float(area_px) / float(bw * bh) if (bw * bh) > 0 else 0.0

    glom_pixels = crop_rgb[mask_binary == 255]
    if len(glom_pixels) > 0:
        mean_rgb = np.mean(glom_pixels, axis=0)
        std_rgb = np.std(glom_pixels, axis=0)
        mean_intensity = float(np.mean(mean_rgb))
        stain_heterogeneity = float(np.mean(std_rgb))
        od = -np.log10((glom_pixels.astype(np.float32) + 1.0) / 256.0)
        mean_od = float(np.mean(od))
        rg_ratio = float((mean_rgb[0] + 1e-3) / (mean_rgb[1] + 1e-3))
    else:
        mean_intensity = float(np.mean(crop_rgb))
        stain_heterogeneity = float(np.std(crop_rgb))
        mean_od = 0.5
        rg_ratio = 1.0

    area_name = "glom_area_um2" if morph_units == "native" else "glom_area"
    perimeter_name = "glom_perimeter_um" if morph_units == "native" else "glom_perimeter"
    return {
        area_name: round(area, 4 if morph_units == "native" else 2),
        perimeter_name: round(perimeter, 4 if morph_units == "native" else 2),
        "circularity": round(circularity, 4),
        "solidity": round(solidity, 4),
        "extent": round(extent, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "mean_intensity": round(mean_intensity, 2),
        "mean_optical_density": round(mean_od, 4),
        "stain_heterogeneity": round(stain_heterogeneity, 2),
        "red_green_ratio": round(rg_ratio, 4),
    }


def extract_morphological_dataset(
    crops_rgb: List[np.ndarray],
    masks_binary: List[np.ndarray],
    morph_units: str = "legacy_resized",
    pixel_sizes_um: Optional[List[float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract aligned morphology without constructing a deep backbone."""
    if len(crops_rgb) != len(masks_binary):
        raise ValueError("crops_rgb and masks_binary must have equal lengths")
    if pixel_sizes_um is None:
        pixel_sizes_um = [1.0] * len(crops_rgb)
    if len(pixel_sizes_um) != len(crops_rgb):
        raise ValueError("pixel_sizes_um must have one value per crop")
    records = [
        extract_single_morphological_features(crop, mask, morph_units, pixel_size)
        for crop, mask, pixel_size in zip(crops_rgb, masks_binary, pixel_sizes_um)
    ]
    if not records:
        return np.empty((0, 0), dtype=np.float32), []
    feature_names = list(records[0])
    matrix = np.asarray([[record[name] for name in feature_names] for record in records], dtype=np.float32)
    return matrix, feature_names


class StainInvariantMorphTransformer:
    """Validate and pass through selected stain-invariant morphological features."""

    def __init__(self, selected_features: Optional[Sequence[str]] = None) -> None:
        self.selected_features = list(selected_features or STAIN_INVARIANT_MORPH_FEATURES)
        self.n_features = len(self.selected_features)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.transform(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.n_features:
            raise ValueError(f"Expected 2D matrix with {self.n_features} stain-invariant features")
        if not np.isfinite(arr).all():
            raise ValueError("Stain-invariant features contain NaN or infinite values")
        return arr


class FeatureBlockTransformer:
    """Train-fitted scaling and optional equal-variance weighting for hybrid blocks."""

    def __init__(self, weighting: str = "none") -> None:
        if weighting not in {"none", "equal"}:
            raise ValueError("weighting must be 'none' or 'equal'")
        self.weighting = weighting
        self.deep_scaler: Optional[StandardScaler] = None
        self.morph_scaler: Optional[StandardScaler] = None
        self.deep_width = 0
        self.morph_width = 0
        self.deep_active_width = 0
        self.morph_active_width = 0
        self.block_variance: Dict[str, float] = {}
        self.block_variance_fraction: Dict[str, float] = {}

    def fit_transform(self, deep: np.ndarray, morph: np.ndarray) -> np.ndarray:
        deep, morph = self._validate(deep, morph)
        self.deep_width, self.morph_width = deep.shape[1], morph.shape[1]
        self.deep_scaler = StandardScaler().fit(deep)
        self.morph_scaler = StandardScaler().fit(morph)
        self.deep_active_width = int(np.count_nonzero(self.deep_scaler.var_ > 1e-12))
        self.morph_active_width = int(np.count_nonzero(self.morph_scaler.var_ > 1e-12))
        return self.transform(deep, morph)

    def transform(self, deep: np.ndarray, morph: np.ndarray) -> np.ndarray:
        if self.deep_scaler is None or self.morph_scaler is None:
            raise RuntimeError("FeatureBlockTransformer must be fitted first")
        deep, morph = self._validate(deep, morph)
        if deep.shape[1] != self.deep_width or morph.shape[1] != self.morph_width:
            raise ValueError("Feature block widths differ from the fitted blocks")
        if self.weighting == "equal":
            deep_block = self.deep_scaler.transform(deep) / np.sqrt(max(self.deep_active_width, 1))
            morph_block = self.morph_scaler.transform(morph) / np.sqrt(max(self.morph_active_width, 1))
        else:
            deep_block = deep
            morph_block = self.morph_scaler.transform(morph)
        self.block_variance = {
            "deep": float(np.var(deep_block, axis=0, ddof=0).sum()),
            "morphological": float(np.var(morph_block, axis=0, ddof=0).sum()),
        }
        total_variance = sum(self.block_variance.values())
        self.block_variance_fraction = {
            name: value / total_variance if total_variance else 0.0
            for name, value in self.block_variance.items()
        }
        return np.hstack([deep_block, morph_block]).astype(np.float32)

    @staticmethod
    def _validate(deep: np.ndarray, morph: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        deep = np.asarray(deep, dtype=np.float64)
        morph = np.asarray(morph, dtype=np.float64)
        if deep.ndim != 2 or morph.ndim != 2 or len(deep) != len(morph):
            raise ValueError("Feature blocks must be 2D and have matching row counts")
        if not np.isfinite(deep).all() or not np.isfinite(morph).all():
            raise ValueError("Feature blocks contain NaN or infinite values")
        if deep.shape[1] == 0 or morph.shape[1] == 0:
            raise ValueError("Feature blocks cannot be empty")
        return deep, morph


class GlomeruliFeatureExtractor:
    """Extracts deep CNN embeddings (ResNet) and morphological features from glomeruli crops."""

    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        device: str = "cpu",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.model, self.embed_dim = self._build_backbone(backbone_name, pretrained)
        self.model.to(self.device)
        self.model.eval()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _build_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        if name == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            net = tv_models.resnet18(weights=weights)
            dim = 512
        elif name == "resnet34":
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            net = tv_models.resnet34(weights=weights)
            dim = 512
        elif name == "resnet50":
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            net = tv_models.resnet50(weights=weights)
            dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        net.fc = nn.Identity()
        return net, dim

    def weights_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def extract_deep_embeddings(self, crops_rgb: List[np.ndarray]) -> np.ndarray:
        if not crops_rgb:
            return np.empty((0, self.embed_dim), dtype=np.float32)
        embeddings = []
        for start in range(0, len(crops_rgb), self.batch_size):
            batch = crops_rgb[start : start + self.batch_size]
            tensors = []
            for img in batch:
                resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA) if img.shape[:2] != (224, 224) else img
                arr = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
                tensors.append(torch.from_numpy(arr.transpose(2, 0, 1)))
            x = torch.stack(tensors, dim=0).to(self.device)
            feat = self.model(x).cpu().numpy().astype(np.float32)
            if self.normalize_embeddings:
                norms = np.linalg.norm(feat, axis=1, keepdims=True)
                feat = feat / np.maximum(norms, 1e-12)
            embeddings.append(feat)
        return np.vstack(embeddings)

    def extract_hybrid(
        self,
        crops_rgb: List[np.ndarray],
        masks_binary: List[np.ndarray],
        morph_units: str = "native",
        pixel_sizes_um: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        deep = self.extract_deep_embeddings(crops_rgb)
        morph, morph_names = extract_morphological_dataset(crops_rgb, masks_binary, morph_units, pixel_sizes_um)
        return deep, morph, morph_names
