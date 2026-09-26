"""
Feature extraction module for glomeruli crops.

Extracts:
1. Deep Representations: CNN embeddings via pre-trained ResNet backbones (Torchvision).
2. Morphological & Histological Features: Area, circularity, solidity, optical density,
   color channel statistics, and texture indicators.
3. Hybrid Representations: Fused multimodal feature vectors for manifold learning.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


def extract_single_morphological_features(crop_rgb: np.ndarray, mask_binary: np.ndarray) -> Dict[str, float]:
    """
    Extract histopathological morphological and photometric features for a single glomerulus.

    Args:
        crop_rgb: (H, W, 3) uint8 image.
        mask_binary: (H, W) uint8 binary mask (0 or 255).

    Returns:
        Dictionary of numerical morphological features.
    """
    if mask_binary.dtype != np.uint8:
        mask_binary = (mask_binary > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Default fallback for empty mask
        return {
            "glom_area": 0.0,
            "glom_perimeter": 0.0,
            "circularity": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "aspect_ratio": 1.0,
            "mean_intensity": 0.0,
            "mean_optical_density": 0.0,
            "stain_heterogeneity": 0.0,
            "red_green_ratio": 1.0,
        }

    # Use largest connected contour
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))

    # Circularity: 4 * pi * Area / (Perimeter^2)
    circularity = float(4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    circularity = min(1.0, max(0.0, circularity))

    # Convex hull & Solidity
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 0 else 0.0

    # Bounding rectangle & Aspect ratio
    bx, by, bw, bh = cv2.boundingRect(cnt)
    aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0
    extent = float(area) / float(bw * bh) if (bw * bh) > 0 else 0.0

    # Photometric / Stain features inside the glomerular mask
    glom_pixels = crop_rgb[mask_binary == 255]
    if len(glom_pixels) > 0:
        mean_rgb = np.mean(glom_pixels, axis=0)  # [R, G, B]
        std_rgb = np.std(glom_pixels, axis=0)
        mean_intensity = float(np.mean(mean_rgb))
        stain_heterogeneity = float(np.mean(std_rgb))

        # Optical density: -log10((I + 1) / 256)
        od = -np.log10((glom_pixels.astype(np.float32) + 1.0) / 256.0)
        mean_od = float(np.mean(od))

        # Red/Green ratio (PAS / eosinophilic collagen indicator)
        rg_ratio = float((mean_rgb[0] + 1e-3) / (mean_rgb[1] + 1e-3))
    else:
        mean_intensity = float(np.mean(crop_rgb))
        stain_heterogeneity = float(np.std(crop_rgb))
        mean_od = 0.5
        rg_ratio = 1.0

    return {
        "glom_area": round(area, 2),
        "glom_perimeter": round(perimeter, 2),
        "circularity": round(circularity, 4),
        "solidity": round(solidity, 4),
        "extent": round(extent, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "mean_intensity": round(mean_intensity, 2),
        "mean_optical_density": round(mean_od, 4),
        "stain_heterogeneity": round(stain_heterogeneity, 2),
        "red_green_ratio": round(rg_ratio, 4),
    }


class GlomeruliFeatureExtractor:
    """
    Extracts deep representation embeddings and morphological features from glomeruli crops.
    """

    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        device: Optional[Any] = 0,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        self.device = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() and device not in (None, "cpu") else "cpu"
        )
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.backbone_name = backbone_name.lower()

        logger.info(f"Initializing Feature Extractor (Backbone: {self.backbone_name}) on device: {self.device}")

        # Build feature extractor backbone up to Global Average Pooling
        if self.backbone_name == "resnet50":
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            base = tv_models.resnet50(weights=weights)
            self.embedding_dim = 2048
        elif self.backbone_name == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            base = tv_models.resnet18(weights=weights)
            self.embedding_dim = 512
        else:  # resnet34 (default)
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            base = tv_models.resnet34(weights=weights)
            self.embedding_dim = 512

        # Extract all layers except the final classification FC
        self.encoder = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        ).to(self.device)

        self.encoder.eval()

        # ImageNet normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract_deep_embeddings(self, crops_rgb: List[np.ndarray]) -> np.ndarray:
        """
        Extract deep CNN embeddings for a collection of (H, W, 3) RGB crops.

        Args:
            crops_rgb: List of numpy uint8 images.

        Returns:
            (N, embedding_dim) float32 numpy array.
        """
        if not crops_rgb:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        all_embeddings = []
        n_crops = len(crops_rgb)

        for i in range(0, n_crops, self.batch_size):
            batch_crops = crops_rgb[i : i + self.batch_size]

            # Convert to float tensor [B, 3, H, W] normalized to [0, 1]
            batch_tensors = []
            for crop in batch_crops:
                if crop.shape[0] != 224 or crop.shape[1] != 224:
                    crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
                else:
                    crop_resized = crop
                t = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
                batch_tensors.append(t)

            batch_stack = torch.stack(batch_tensors).to(self.device)
            # Apply ImageNet normalization
            batch_norm = (batch_stack - self.mean) / self.std

            # Forward pass through encoder
            features = self.encoder(batch_norm)  # (B, embedding_dim, 1, 1)
            features = torch.flatten(features, 1)  # (B, embedding_dim)

            if self.normalize_embeddings:
                features = nn.functional.normalize(features, p=2, dim=1)

            all_embeddings.append(features.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def extract_morphological_dataset(
        self, crops_rgb: List[np.ndarray], masks_binary: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract tabular morphological features for all crops and masks.

        Returns:
            features_matrix: (N, n_features) float32 array.
            feature_names: List of column names.
        """
        records = []
        for crop, mask in zip(crops_rgb, masks_binary):
            feat_dict = extract_single_morphological_features(crop, mask)
            records.append(feat_dict)

        if not records:
            return np.empty((0, 0), dtype=np.float32), []

        feature_names = list(records[0].keys())
        features_matrix = np.array(
            [[r[col] for col in feature_names] for r in records], dtype=np.float32
        )
        return features_matrix, feature_names
