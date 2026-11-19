"""Serializable clustering assignment pipeline and its frozen experimental label mapping."""

from typing import Optional, Sequence
import numpy as np
from src.clustering.feature_extractor import (
    STAIN_INVARIANT_MORPH_FEATURES,
    extract_morphological_dataset,
    select_morphological_columns,
)
from src.clustering.labels import peripheral_clear_fraction, LABEL_LIMITATION


class ClusterAssignmentPipeline:
    def __init__(
        self,
        extractor,
        clusterer,
        feature_type: str = "stain_invariant_morph",
        morph_units: str = "native",
        block_transformer=None,
        selected_features: Optional[Sequence[str]] = None,
    ) -> None:
        self.extractor = extractor
        self.clusterer = clusterer
        self.feature_type = feature_type
        self.morph_units = morph_units
        self.block_transformer = block_transformer
        self.selected_features = list(selected_features or STAIN_INVARIANT_MORPH_FEATURES)
        self.label_mapping = {i: chr(65 + i) for i in range(clusterer.n_clusters)}
        self.label_limitation = LABEL_LIMITATION

    def _features_for_new(self, crops, masks, pixels):
        morph, names = extract_morphological_dataset(crops, masks, self.morph_units, pixels)
        if self.feature_type == "stain_invariant_morph":
            selected_matrix, _ = select_morphological_columns(morph, names, self.selected_features)
            return selected_matrix
        if self.feature_type == "hybrid":
            if self.extractor is None or self.block_transformer is None:
                raise RuntimeError("Hybrid pipeline requires both extractor and block_transformer")
            deep = self.extractor.extract_deep_embeddings(crops)
            return self.block_transformer.transform(deep, morph)
        raise ValueError(f"Unsupported pipeline feature_type: {self.feature_type}")

    def class_labels(self, cluster_ids):
        return [self.label_mapping[int(cluster_id)] for cluster_id in cluster_ids]
