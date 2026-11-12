"""Train-fitted PCA20 and K-Means with deterministic held-out assignment."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class _IdentityScaler:
    def fit(self, features):
        self.mean_ = np.zeros(features.shape[1]); self.scale_ = np.ones(features.shape[1])
        self.var_ = np.ones(features.shape[1])
        return self

    def transform(self, features):
        return np.asarray(features, dtype=np.float64)

    def fit_transform(self, features):
        return self.fit(features).transform(features)


class GlomeruliClusterer:
    def __init__(self, n_clusters=4, pca_variance=20, manifold_method="pca", random_state=42,
                 input_scaling="none"):
        if pca_variance != 20 or manifold_method != "pca" or input_scaling != "none":
            raise ValueError("The supported solution is global PCA20 with explicit block weighting")
        if n_clusters not in (2, 4, 6):
            raise ValueError("Supported K values are 2, 4, 6")
        self.n_clusters, self.random_state = n_clusters, random_state
        self.pca_variance, self.manifold_method, self.input_scaling = 20, "pca", "none"
        self.algorithm = "kmeans"
        self.scaler = _IdentityScaler()
        self.pca = PCA(n_components=20, random_state=random_state)
        self.cluster_model = None

    def fit(self, features, algorithm="kmeans"):
        if algorithm != "kmeans":
            raise ValueError("Only K-Means is supported")
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or not np.isfinite(features).all() or min(features.shape) < 20:
            raise ValueError("PCA20 needs at least 20 finite samples and 20 coordinates")
        reduced = self.pca.fit_transform(self.scaler.fit_transform(features))
        # Fit using transform, matching the original PCA20 implementation exactly.
        reduced = self.pca.transform(features)
        self.cluster_model = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state).fit(reduced)
        counts = np.bincount(self.cluster_model.predict(reduced), minlength=self.n_clusters)
        self.fit_diagnostics_ = {"algorithm":"kmeans", "requested_clusters":self.n_clusters,
            "occupied_clusters":int(np.count_nonzero(counts)), "counts":counts.tolist(),
            "valid":bool(np.count_nonzero(counts)==self.n_clusters), "n_iter":int(self.cluster_model.n_iter_)}
        return self

    def transform(self, features):
        if self.cluster_model is None:
            raise RuntimeError("The model is not fitted")
        return self.pca.transform(self.scaler.transform(np.asarray(features, dtype=np.float64)))

    def predict(self, features):
        return self.cluster_model.predict(self.transform(features))

    def distances_to_centers(self, features):
        reduced = self.transform(features)
        labels = self.cluster_model.predict(reduced)
        return np.linalg.norm(reduced - self.cluster_model.cluster_centers_[labels], axis=1)

    @staticmethod
    def _metrics(features, labels):
        if not 2 <= len(np.unique(labels)) < len(labels):
            return {name:None for name in ("silhouette_score","davies_bouldin_score","calinski_harabasz_score")}
        return {"silhouette_score":float(silhouette_score(features,labels)),
            "davies_bouldin_score":float(davies_bouldin_score(features,labels)),
            "calinski_harabasz_score":float(calinski_harabasz_score(features,labels))}
