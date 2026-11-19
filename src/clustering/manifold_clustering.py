"""Train-fitted scaling / optional manifold reduction and K-Means with deterministic held-out assignment."""

from typing import Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class _IdentityScaler:
    def fit(self, features):
        self.mean_ = np.zeros(features.shape[1])
        self.scale_ = np.ones(features.shape[1])
        self.var_ = np.ones(features.shape[1])
        return self

    def transform(self, features):
        return np.asarray(features, dtype=np.float64)

    def fit_transform(self, features):
        return self.fit(features).transform(features)


class GlomeruliClusterer:
    """Train-fitted clusterer supporting standardized direct space (default) and PCA20 baseline."""

    def __init__(
        self,
        n_clusters: int = 4,
        pca_variance: Optional[int] = None,
        manifold_method: str = "none",
        random_state: int = 42,
        input_scaling: str = "standard",
    ) -> None:
        if n_clusters not in (2, 4, 6):
            raise ValueError("Supported K values are 2, 4, 6")
        if manifold_method not in {"none", "pca"}:
            raise ValueError(f"Unsupported manifold_method: {manifold_method}")
        if input_scaling not in {"standard", "none"}:
            raise ValueError(f"Unsupported input_scaling: {input_scaling}")
        if manifold_method == "pca" and pca_variance != 20:
            raise ValueError("When manifold_method='pca', pca_variance must be 20")

        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.pca_variance = 20 if manifold_method == "pca" else None
        self.manifold_method = manifold_method
        self.input_scaling = input_scaling
        self.algorithm = "kmeans"
        self.scaler = StandardScaler() if input_scaling == "standard" else _IdentityScaler()
        self.pca: Optional[PCA] = PCA(n_components=20, random_state=random_state) if manifold_method == "pca" else None
        self.cluster_model: Optional[KMeans] = None

    def fit(self, features: np.ndarray, algorithm: str = "kmeans") -> "GlomeruliClusterer":
        if algorithm != "kmeans":
            raise ValueError("Only K-Means is supported")
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError("Features must be a finite 2D array")
        if len(features) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} samples to fit {self.n_clusters} clusters")
        if self.manifold_method == "pca" and min(features.shape) < 20:
            raise ValueError("PCA20 needs at least 20 finite samples and 20 coordinates")

        scaled = self.scaler.fit_transform(features)
        reduced = self.pca.fit_transform(scaled) if self.pca is not None else scaled
        self.cluster_model = KMeans(
            n_clusters=self.n_clusters,
            n_init=20,
            random_state=self.random_state,
        ).fit(reduced)
        counts = np.bincount(self.cluster_model.predict(reduced), minlength=self.n_clusters)
        self.fit_diagnostics_ = {
            "algorithm": "kmeans",
            "manifold_method": self.manifold_method,
            "input_scaling": self.input_scaling,
            "requested_clusters": self.n_clusters,
            "occupied_clusters": int(np.count_nonzero(counts)),
            "counts": counts.tolist(),
            "valid": bool(np.count_nonzero(counts) == self.n_clusters),
            "n_iter": int(self.cluster_model.n_iter_),
        }
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.cluster_model is None:
            raise RuntimeError("The model is not fitted")
        scaled = self.scaler.transform(np.asarray(features, dtype=np.float64))
        return self.pca.transform(scaled) if self.pca is not None else scaled

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.cluster_model.predict(self.transform(features))

    def distances_to_centers(self, features: np.ndarray) -> np.ndarray:
        reduced = self.transform(features)
        labels = self.cluster_model.predict(reduced)
        return np.linalg.norm(reduced - self.cluster_model.cluster_centers_[labels], axis=1)

    @staticmethod
    def _metrics(features: np.ndarray, labels: np.ndarray) -> dict:
        if not 2 <= len(np.unique(labels)) < len(labels):
            return {name: None for name in ("silhouette_score", "davies_bouldin_score", "calinski_harabasz_score")}
        return {
            "silhouette_score": float(silhouette_score(features, labels)),
            "davies_bouldin_score": float(davies_bouldin_score(features, labels)),
            "calinski_harabasz_score": float(calinski_harabasz_score(features, labels)),
        }
