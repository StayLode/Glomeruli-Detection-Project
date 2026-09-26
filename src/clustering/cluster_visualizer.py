"""
Publication-quality visualization module for unsupervised glomeruli clustering and manifold learning.

Produces:
1. 2D Manifold Scatter Plot (t-SNE / UMAP colored by cluster with clinical annotations)
2. Optimal K Profiling Plot (Elbow Inertia + Silhouette Score vs number of clusters)
3. Morphological Feature Distribution Boxplots (Area, Circularity, Optical Density)
4. Exemplar Cluster Gallery (representative glomeruli images for each discovered class)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Clinical palette: Class 0 (Teal/Green: Healthy), Class 1 (Amber: Segmental), Class 2 (Crimson: Sclerotic)
CLINICAL_COLORS = ["#2a9d8f", "#e76f51", "#d62828", "#457b9d", "#6a4c93", "#f4a261"]
CLINICAL_NAMES = [
    "Class 0 (Normal / Preserved)",
    "Class 1 (Segmental Sclerosis)",
    "Class 2 (Global Sclerosis / Necrotic)",
    "Class 3 (Hypertrophic / Atypical)",
]


class ClusterVisualizer:
    """Generates analytical and qualitative visualizations for glomerular clustering."""

    def __init__(self, output_dir: Union[str, Path] = "runs/clustering/baseline") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_manifold_scatter(
        self,
        embedding_2d: np.ndarray,
        labels: np.ndarray,
        method_name: str = "t-SNE",
        silhouette: float = 0.0,
        save_name: str = "manifold_clusters.png",
    ) -> Path:
        """
        Plot 2D manifold scatter with cluster coloring and density contours.
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        unique_labels = np.unique(labels)
        total_samples = len(labels)

        for i, cid in enumerate(unique_labels):
            mask = labels == cid
            color = CLINICAL_COLORS[i % len(CLINICAL_COLORS)]
            label_text = (
                CLINICAL_NAMES[i] if i < len(CLINICAL_NAMES) else f"Cluster {cid}"
            )
            count = int(np.sum(mask))
            pct = (count / total_samples) * 100.0

            ax.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=color,
                label=f"{label_text} (n={count}, {pct:.1f}%)",
                alpha=0.80,
                edgecolors="none",
                s=40,
            )

            # Plot cluster centroid in manifold space
            centroid = np.mean(embedding_2d[mask], axis=0)
            ax.scatter(
                centroid[0],
                centroid[1],
                c="black",
                marker="X",
                s=120,
                edgecolors="white",
                linewidth=1.5,
                zorder=10,
            )

        ax.set_title(
            f"Unsupervised Glomerular Subtyping ({method_name.upper()} Projection)\n"
            f"Total Glomeruli: {total_samples} | Silhouette Score: {silhouette:.3f}",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel(f"{method_name.upper()} Dimension 1", fontsize=11)
        ax.set_ylabel(f"{method_name.upper()} Dimension 2", fontsize=11)
        ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=10, loc="best")
        ax.grid(True, linestyle="--", alpha=0.3)

        out_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    def plot_optimal_k_curves(
        self,
        k_results: Dict[str, Any],
        save_name: str = "optimal_k_analysis.png",
    ) -> Path:
        """
        Plot Elbow (Inertia) and Silhouette score curves across cluster counts.
        """
        k_range = k_results["k_range"]
        inertias = k_results["inertias"]
        silhouettes = k_results["silhouettes"]
        best_k = k_results.get("optimal_k_by_silhouette", 3)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Plot 1: Elbow Method (Inertia)
        ax1.plot(k_range, inertias, marker="o", color="#1d3557", linewidth=2, markersize=7)
        ax1.axvline(best_k, color="#e63946", linestyle="--", alpha=0.7, label=f"Selected K={best_k}")
        ax1.set_title("Elbow Method (Cluster Inertia)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Number of Clusters (K)", fontsize=11)
        ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend()

        # Plot 2: Silhouette Score
        ax2.plot(k_range, silhouettes, marker="s", color="#2a9d8f", linewidth=2, markersize=7)
        ax2.axvline(best_k, color="#e63946", linestyle="--", alpha=0.7, label=f"Optimal K={best_k}")
        ax2.set_title("Silhouette Score vs K", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Number of Clusters (K)", fontsize=11)
        ax2.set_ylabel("Mean Silhouette Score", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend()

        out_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    def plot_morphological_distributions(
        self,
        features_matrix: np.ndarray,
        feature_names: List[str],
        labels: np.ndarray,
        save_name: str = "morphological_distributions.png",
    ) -> Path:
        """
        Boxplots comparing key histological biomarkers across discovered clusters.
        """
        target_features = ["glom_area", "circularity", "mean_optical_density", "red_green_ratio"]
        present_features = [f for f in target_features if f in feature_names]

        if not present_features:
            present_features = feature_names[:4]

        n_feats = len(present_features)
        fig, axes = plt.subplots(1, n_feats, figsize=(4.5 * n_feats, 5), dpi=150)
        if n_feats == 1:
            axes = [axes]

        unique_labels = np.unique(labels)
        k = len(unique_labels)

        for idx, feat_name in enumerate(present_features):
            col_idx = feature_names.index(feat_name)
            data_by_class = [features_matrix[labels == cid, col_idx] for cid in unique_labels]

            ax = axes[idx]
            bplot = ax.boxplot(
                data_by_class,
                patch_artist=True,
                tick_labels=[f"Class {cid}" for cid in unique_labels],
                medianprops=dict(color="black", linewidth=1.5),
            )

            for p_idx, patch in enumerate(bplot["boxes"]):
                patch.set_facecolor(CLINICAL_COLORS[p_idx % len(CLINICAL_COLORS)])
                patch.set_alpha(0.75)

            clean_title = feat_name.replace("_", " ").title()
            ax.set_title(clean_title, fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3)

        plt.suptitle("Clinical Biomarker Distributions across Discovered Subtypes", fontsize=14, y=1.03)
        out_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_cluster_gallery(
        self,
        crops_rgb: List[np.ndarray],
        masks_binary: List[np.ndarray],
        labels: np.ndarray,
        pca_features: np.ndarray,
        samples_per_cluster: int = 6,
        save_name: str = "cluster_gallery.png",
    ) -> Path:
        """
        Render a gallery of representative glomeruli closest to cluster centroids.
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        fig, axes = plt.subplots(
            n_clusters,
            samples_per_cluster,
            figsize=(3.0 * samples_per_cluster, 3.2 * n_clusters),
            dpi=150,
        )

        if n_clusters == 1:
            axes = np.expand_dims(axes, 0)
        if samples_per_cluster == 1:
            axes = np.expand_dims(axes, 1)

        for row, cid in enumerate(unique_labels):
            cluster_indices = np.where(labels == cid)[0]
            if len(cluster_indices) == 0:
                continue

            # Find samples closest to centroid in PCA space
            centroid = np.mean(pca_features[cluster_indices], axis=0)
            dists = np.linalg.norm(pca_features[cluster_indices] - centroid, axis=1)
            sorted_local_idx = np.argsort(dists)

            selected_indices = cluster_indices[sorted_local_idx[:samples_per_cluster]]

            class_name = CLINICAL_NAMES[row] if row < len(CLINICAL_NAMES) else f"Cluster {cid}"

            for col in range(samples_per_cluster):
                ax = axes[row, col]
                if col < len(selected_indices):
                    idx = selected_indices[col]
                    crop = crops_rgb[idx].copy()
                    mask = masks_binary[idx]

                    # Overlay mask contours in lime green
                    if mask is not None and np.any(mask > 0):
                        cnts, _ = cv2.findContours(
                            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(crop, cnts, -1, (0, 255, 0), 2)

                    ax.imshow(crop)
                ax.axis("off")
                if col == 0:
                    ax.set_title(f"{class_name}\n(Exemplar 1)", fontsize=10, fontweight="bold")
                else:
                    ax.set_title(f"Exemplar {col + 1}", fontsize=9)

        plt.tight_layout()
        out_path = self.output_dir / save_name
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return out_path
