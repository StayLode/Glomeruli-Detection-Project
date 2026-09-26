"""
Unsupervised Manifold Learning and Glomerular Disease Grading module.

Implements:
1. Dimensionality Reduction (PCA + t-SNE / UMAP)
2. Unsupervised Clustering (K-Means, GMM)
3. Cluster Validation Metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, Inertia)
4. Optimal K Analysis (Elbow Method & Silhouette Profiling)
5. Clinically Aligned Trajectory Ordering (Healthy -> Segmental Sclerosis -> Global Sclerosis)
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

try:
    import umap
    HAS_UMAP = True
except ImportError:
    umap = None  # type: ignore
    HAS_UMAP = False

logger = logging.getLogger(__name__)


class GlomeruliClusterer:
    """
    Manifold learning and clustering pipeline for unsupervised glomerular subtyping.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        pca_variance: float = 0.95,
        manifold_method: str = "tsne",
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.pca_variance = pca_variance
        self.manifold_method = manifold_method.lower()
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_variance, random_state=random_state)
        self.cluster_model: Optional[Union[KMeans, GaussianMixture]] = None
        self.manifold_embedding: Optional[np.ndarray] = None
        self.pca_features: Optional[np.ndarray] = None

    def fit_transform(
        self,
        features: np.ndarray,
        algorithm: str = "kmeans",
        order_by_progression: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full clustering and manifold projection pipeline.

        Args:
            features: (N, D) feature matrix (deep embeddings, morphological, or hybrid).
            algorithm: 'kmeans' or 'gmm'.
            order_by_progression: If True, re-indexes clusters along the disease progression axis.

        Returns:
            Dictionary containing labels, manifold coordinates, and validation metrics.
        """
        N, D = features.shape
        logger.info(f"Clustering {N} glomeruli with {D} features using {algorithm.upper()} (k={self.n_clusters})...")

        # 1. Feature Standardization
        X_scaled = self.scaler.fit_transform(features)

        # 2. PCA Dimensionality Reduction
        pca_feats = self.pca.fit_transform(X_scaled)
        if pca_feats.shape[1] < 2 and D >= 2:
            self.pca = PCA(n_components=min(2, D, N - 1), random_state=self.random_state)
            pca_feats = self.pca.fit_transform(X_scaled)

        self.pca_features = pca_feats
        n_pcs = pca_feats.shape[1]
        var_explained = float(np.sum(self.pca.explained_variance_ratio_))
        logger.info(f"PCA reduced {D} dimensions to {n_pcs} components (Variance Explained: {var_explained:.2%})")

        # 3. 2D Manifold Embedding Projection (t-SNE or UMAP)
        if self.manifold_method == "umap" and HAS_UMAP and umap is not None:
            logger.info("Computing UMAP 2D embedding...")
            reducer = umap.UMAP(n_components=2, random_state=self.random_state, n_neighbors=15, min_dist=0.1)
            self.manifold_embedding = reducer.fit_transform(pca_feats)
        else:
            if self.manifold_method == "umap" and not HAS_UMAP:
                logger.warning("UMAP not installed; falling back to t-SNE projection.")
            logger.info("Computing t-SNE 2D embedding...")
            perplexity = min(30.0, max(5.0, float(N - 1) / 3.0))
            tsne_init = "pca" if pca_feats.shape[1] >= 2 else "random"
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=self.random_state,
                learning_rate="auto",
                init=tsne_init,
            )
            self.manifold_embedding = tsne.fit_transform(pca_feats)

        # 4. Clustering in PCA Space
        if algorithm.lower() == "gmm":
            gmm = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
            raw_labels = gmm.fit_predict(pca_feats)
            self.cluster_model = gmm
        else:
            km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state)
            raw_labels = km.fit_predict(pca_feats)
            self.cluster_model = km

        # 5. Clinical Progression Alignment
        # Order clusters along PC1 so that 0=Normal, 1=Segmental Sclerosis, 2=Global Sclerosis
        if order_by_progression:
            labels = self._align_clusters_by_progression(raw_labels, pca_feats)
        else:
            labels = raw_labels

        # 6. Cluster Quality Validation Metrics
        sil_score = float(silhouette_score(pca_feats, labels)) if self.n_clusters > 1 else 0.0
        db_score = float(davies_bouldin_score(pca_feats, labels)) if self.n_clusters > 1 else 0.0
        ch_score = float(calinski_harabasz_score(pca_feats, labels)) if self.n_clusters > 1 else 0.0

        logger.info(f"Clustering Quality Metrics:")
        logger.info(f"   Silhouette Score : {sil_score:.4f} (higher is better)")
        logger.info(f"   Davies-Bouldin   : {db_score:.4f} (lower is better)")
        logger.info(f"   Calinski-Harabasz: {ch_score:.2f} (higher is better)")

        # Cluster distribution breakdown
        unique, counts = np.unique(labels, return_counts=True)
        dist_dict = {f"Class {u}": int(c) for u, c in zip(unique, counts)}
        logger.info(f"Discovered Cluster Distribution: {dist_dict}")

        return {
            "labels": labels,
            "embedding_2d": self.manifold_embedding,
            "pca_features": self.pca_features,
            "silhouette_score": round(sil_score, 4),
            "davies_bouldin_score": round(db_score, 4),
            "calinski_harabasz_score": round(ch_score, 2),
            "distribution": dist_dict,
            "n_components_pca": int(n_pcs),
            "variance_explained": round(var_explained, 4),
        }

    def _align_clusters_by_progression(self, labels: np.ndarray, pca_feats: np.ndarray) -> np.ndarray:
        """
        Sort cluster IDs monotonically along the first principal component (PC1).
        Provides consistent clinical semantics across different runs and patient cohorts.
        """
        unique_labels = np.unique(labels)
        cluster_means = []

        for cid in unique_labels:
            mean_pc1 = float(np.mean(pca_feats[labels == cid, 0]))
            cluster_means.append((cid, mean_pc1))

        # Sort by mean PC1 ascending
        cluster_means.sort(key=lambda x: x[1])

        # Map to ordered integers [0, 1, ..., K-1]
        label_map = {old_cid: new_cid for new_cid, (old_cid, _) in enumerate(cluster_means)}
        ordered_labels = np.array([label_map[l] for l in labels], dtype=int)
        return ordered_labels

    def evaluate_optimal_k(self, features: np.ndarray, k_range: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Evaluate Silhouette and Inertia across a range of cluster counts (Elbow analysis).
        """
        if k_range is None:
            k_range = [2, 3, 4, 5, 6]

        X_scaled = self.scaler.fit_transform(features)
        X_pca = self.pca.fit_transform(X_scaled)

        inertias = []
        silhouettes = []
        db_scores = []

        for k in k_range:
            km = KMeans(n_clusters=k, n_init=15, random_state=self.random_state)
            km_labels = km.fit_predict(X_pca)
            inertias.append(float(km.inertia_))
            silhouettes.append(float(silhouette_score(X_pca, km_labels)))
            db_scores.append(float(davies_bouldin_score(X_pca, km_labels)))

        best_k_idx = int(np.argmax(silhouettes))
        best_k = k_range[best_k_idx]

        return {
            "k_range": k_range,
            "inertias": inertias,
            "silhouettes": silhouettes,
            "davies_bouldin": db_scores,
            "optimal_k_by_silhouette": best_k,
        }
