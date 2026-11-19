"""Refit feature transforms and K-Means clusterers on explicit train subsets."""

from typing import Optional, Sequence
import numpy as np
from src.clustering.feature_extractor import (
    FeatureBlockTransformer,
    StainInvariantMorphTransformer,
)
from src.clustering.manifold_clustering import GlomeruliClusterer


class FittedFeatureModel:
    def __init__(self, transformer, clusterer, feature_type: str = "stain_invariant_morph"):
        self.block_transformer = transformer
        self.clusterer = clusterer
        self.feature_type = feature_type

    def predict(self, features=None, deep_features=None, morph_features=None):
        if self.feature_type == "hybrid":
            transformed = self.block_transformer.transform(deep_features, morph_features)
        else:
            transformed = self.block_transformer.transform(features)
        return self.clusterer.predict(transformed)


def fit_feature_model(
    n_clusters: int,
    algorithm: str,
    seed: int,
    pca_variance: Optional[int],
    feature_type: str,
    block_weighting: str,
    input_scaling: str,
    features: Optional[np.ndarray] = None,
    deep_features: Optional[np.ndarray] = None,
    morph_features: Optional[np.ndarray] = None,
    manifold_method: Optional[str] = None,
    selected_features: Optional[Sequence[str]] = None,
) -> FittedFeatureModel:
    if feature_type == "stain_invariant_morph":
        if block_weighting not in {"none", None}:
            raise ValueError("stain_invariant_morph requires block_weighting='none'")
        method = manifold_method or "none"
        transformer = StainInvariantMorphTransformer(selected_features=selected_features)
        values = transformer.fit_transform(features)
        clusterer = GlomeruliClusterer(
            n_clusters=n_clusters,
            pca_variance=pca_variance if method == "pca" else None,
            manifold_method=method,
            random_state=seed,
            input_scaling=input_scaling,
        )
        return FittedFeatureModel(transformer, clusterer.fit(values, algorithm), feature_type=feature_type)

    if feature_type == "hybrid" and block_weighting == "equal":
        method = manifold_method or "pca"
        transformer = FeatureBlockTransformer("equal")
        values = transformer.fit_transform(deep_features, morph_features)
        clusterer = GlomeruliClusterer(
            n_clusters=n_clusters,
            pca_variance=pca_variance if pca_variance is not None else 20,
            manifold_method=method,
            random_state=seed,
            input_scaling=input_scaling,
        )
        return FittedFeatureModel(transformer, clusterer.fit(values, algorithm), feature_type=feature_type)

    raise ValueError(
        f"Unsupported feature configuration: feature_type={feature_type!r}, block_weighting={block_weighting!r}"
    )
