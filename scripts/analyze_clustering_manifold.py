#!/usr/bin/env python3
"""
Diagnostic analysis of clustering feature spaces, modality ablation,
slide-level domain shift audit, and manifold learning comparison
on the frozen train/val development cohort (zero test leakage).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, SpectralEmbedding, trustworthiness
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

TEST_SLIDES = {"RECHERCHE-015", "RECHERCHE-017"}


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Format a DataFrame as a GitHub-flavored Markdown table without extra dependencies."""
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells: List[str] = []
        for val in row:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cells.append("N/A")
            elif isinstance(val, (float, np.floating)):
                cells.append(format(float(val), floatfmt))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def check_lfs_pointer(path: Path) -> None:
    """Verify that the artifact is materialized locally and not an unresolved Git LFS pointer."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with path.open("rb") as fh:
        header = fh.read(64)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"File {path} is an unresolved Git LFS pointer.\n"
            "Run 'git lfs pull' from the repository root and retry."
        )


def load_run_artifacts(run_root: Path) -> Tuple[Dict[str, np.ndarray], Dict[int, pd.DataFrame]]:
    """Load shared features.npz and per-K assignments.csv aligned by annotation_id."""
    k2_dir = run_root / "clustering_k2"
    npz_path = k2_dir / "features.npz"
    check_lfs_pointer(npz_path)

    with np.load(npz_path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}

    splits = set(data["split"].astype(str))
    slides = set(data["slide_id"].astype(str))
    if not splits.issubset({"train", "val"}) or (slides & TEST_SLIDES):
        raise ValueError("Test samples detected in features.npz; aborting to prevent data leakage.")

    assignments_by_k: Dict[int, pd.DataFrame] = {}
    base_ids = data["annotation_id"].astype(str)
    id_to_idx = {aid: i for i, aid in enumerate(base_ids)}

    for k in (2, 4, 6):
        csv_path = run_root / f"clustering_k{k}" / "assignments.csv"
        df = pd.read_csv(csv_path)
        df["annotation_id"] = df["annotation_id"].astype(str)
        df["_npz_idx"] = [id_to_idx[aid] for aid in df["annotation_id"]]
        df = df.sort_values("_npz_idx").drop(columns=["_npz_idx"]).reset_index(drop=True)
        assignments_by_k[k] = df

    return data, assignments_by_k


def safe_internal_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Compute internal clustering metrics safely when fewer than 2 clusters are populated."""
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.min(counts) < 2:
        return {
            "n_occupied": int(len(unique)),
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }
    return {
        "n_occupied": int(len(unique)),
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def run_modality_ablation(
    data: Dict[str, np.ndarray],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    k_values: Tuple[int, ...] = (2, 3, 4, 5, 6),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compare cluster separability and validation collapse across three train-fitted spaces:
      1. morph_only (10D): train-standardized morphological/histological features
      2. deep_only (PCA20): L2-normalized ResNet34 embeddings -> PCA(20)
      3. hybrid (PCA20): equal-variance balanced hybrid vector -> PCA(20)
    """
    morph_raw = data["morphological_features_raw"].astype(np.float64)
    morph_scaler = StandardScaler().fit(morph_raw[train_mask])
    X_morph = morph_scaler.transform(morph_raw)

    deep_raw = data["deep_features_raw"].astype(np.float64)
    deep_norm = deep_raw / np.maximum(np.linalg.norm(deep_raw, axis=1, keepdims=True), 1e-12)
    pca_deep = PCA(n_components=20, random_state=seed).fit(deep_norm[train_mask])
    X_deep = pca_deep.transform(deep_norm)

    hybrid_raw = data["model_features"].astype(np.float64)
    pca_hybrid = PCA(n_components=20, random_state=seed).fit(hybrid_raw[train_mask])
    X_hybrid = pca_hybrid.transform(hybrid_raw)

    common_morph = data["common_morph_features"].astype(np.float64)

    spaces = {
        "morph_only (10D)": X_morph,
        "deep_only (PCA20)": X_deep,
        "hybrid (PCA20)": X_hybrid,
    }

    records: List[Dict[str, Any]] = []
    for space_name, X in spaces.items():
        for k in k_values:
            km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X[train_mask])
            tr_labels = km.predict(X[train_mask])
            va_labels = km.predict(X[val_mask])

            m_tr_space = safe_internal_metrics(X[train_mask], tr_labels)
            m_tr_common = safe_internal_metrics(common_morph[train_mask], tr_labels)
            m_va_space = safe_internal_metrics(X[val_mask], va_labels)

            _, va_counts = np.unique(va_labels, return_counts=True)
            val_max_share = float(np.max(va_counts) / np.sum(va_counts))

            records.append(
                {
                    "modality": space_name,
                    "k": k,
                    "train_sil_model_space": m_tr_space["silhouette"],
                    "train_sil_common_morph": m_tr_common["silhouette"],
                    "train_db": m_tr_space["davies_bouldin"],
                    "train_ch": m_tr_space["calinski_harabasz"],
                    "val_occupied_clusters": m_va_space["n_occupied"],
                    "val_max_cluster_share": val_max_share,
                    "val_sil_model_space": m_va_space["silhouette"],
                }
            )
    return pd.DataFrame(records)


def run_slide_shift_audit(
    data: Dict[str, np.ndarray],
    df_k2: pd.DataFrame,
    train_mask: np.ndarray,
) -> pd.DataFrame:
    """Quantify per-slide staining/morphological shift and distance to train centroids."""
    morph_raw = data["common_morph_raw"].astype(np.float64)
    feat_names = [str(x) for x in data["common_feature_names"]]
    df_feat = pd.DataFrame(morph_raw, columns=feat_names)
    df_feat["slide_id"] = data["slide_id"].astype(str)
    df_feat["split"] = data["split"].astype(str)
    df_feat["k2_label"] = df_k2["class_label"].astype(str)
    df_feat["k2_dist"] = df_k2["distance_to_center"].astype(float)
    if "peripheral_clear_fraction" in df_k2.columns:
        df_feat["peripheral_clear_fraction"] = df_k2["peripheral_clear_fraction"].astype(float)

    scaler = StandardScaler().fit(morph_raw[train_mask])
    Z_morph = scaler.transform(morph_raw)
    df_feat["morph_z_norm_from_train_mean"] = np.linalg.norm(Z_morph, axis=1)

    rows = []
    for (split, slide_id), grp in df_feat.groupby(["split", "slide_id"], sort=True):
        row: Dict[str, Any] = {
            "split": split,
            "slide_id": slide_id,
            "n_glom": int(len(grp)),
            "k2_A_pct": float((grp["k2_label"] == "A").mean() * 100.0),
            "k2_B_pct": float((grp["k2_label"] == "B").mean() * 100.0),
            "k2_dist_median": float(grp["k2_dist"].median()),
            "morph_z_dist_median": float(grp["morph_z_norm_from_train_mean"].median()),
            "mean_intensity_med": float(grp["mean_intensity"].median()),
            "mean_od_med": float(grp["mean_optical_density"].median()),
            "rg_ratio_med": float(grp["red_green_ratio"].median()),
            "stain_het_med": float(grp["stain_heterogeneity"].median()),
            "glom_area_um2_med": float(grp["glom_area_um2"].median()),
            "circularity_med": float(grp["circularity"].median()),
        }
        if "peripheral_clear_fraction" in grp.columns:
            row["rim_clear_med"] = float(grp["peripheral_clear_fraction"].median())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["split", "slide_id"]).reset_index(drop=True)


def run_feature_associations(
    data: Dict[str, np.ndarray],
    assignments_by_k: Dict[int, pd.DataFrame],
    train_mask: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute morphological PCA loadings and Kruskal-Wallis eta-squared across K=2,4,6 on train."""
    morph_raw = data["common_morph_raw"].astype(np.float64)
    feat_names = [str(x) for x in data["common_feature_names"]]
    df_m = pd.DataFrame(morph_raw, columns=feat_names)
    if "peripheral_clear_fraction" in assignments_by_k[2].columns:
        df_m["peripheral_clear_fraction"] = assignments_by_k[2]["peripheral_clear_fraction"].astype(float)

    scaler = StandardScaler().fit(df_m.iloc[train_mask])
    Z_tr = scaler.transform(df_m.iloc[train_mask])
    pca_m = PCA(n_components=min(5, Z_tr.shape[1]), random_state=42).fit(Z_tr)
    loadings = pd.DataFrame(
        pca_m.components_.T,
        index=df_m.columns,
        columns=[f"PC{i+1} ({v*100:.1f}%)" for i, v in enumerate(pca_m.explained_variance_ratio_)],
    )

    assoc_rows = []
    n_tr = int(np.sum(train_mask))
    for feat in df_m.columns:
        vals = df_m.iloc[train_mask][feat].to_numpy()
        entry: Dict[str, Any] = {"feature": feat}
        for k in (2, 4, 6):
            labels = assignments_by_k[k].iloc[train_mask]["class_label"].astype(str).to_numpy()
            groups = [vals[labels == g] for g in np.unique(labels) if np.sum(labels == g) > 0]
            if len(groups) >= 2:
                stat, pval = kruskal(*groups)
                eta2 = max(0.0, float((stat - len(groups) + 1) / max(1, n_tr - len(groups))))
                entry[f"k{k}_eta2"] = eta2
                entry[f"k{k}_pval"] = float(pval)
            else:
                entry[f"k{k}_eta2"] = None
                entry[f"k{k}_pval"] = None
        assoc_rows.append(entry)

    df_assoc = pd.DataFrame(assoc_rows).sort_values("k4_eta2", ascending=False).reset_index(drop=True)
    return loadings, df_assoc


def run_manifold_comparison(
    data: Dict[str, np.ndarray],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Compare linear and non-linear manifold embeddings (PCA, Isomap, SpectralEmbedding,
    and optional UMAP) on trustworthiness and K-Means / GMM separability.
    """
    hybrid_raw = data["model_features"].astype(np.float64)
    common_morph = data["common_morph_features"].astype(np.float64)

    embeddings_2d: Dict[str, np.ndarray] = {}
    summary_rows: List[Dict[str, Any]] = []

    pca2 = PCA(n_components=2, random_state=seed).fit(hybrid_raw[train_mask])
    embeddings_2d["PCA (2D, train-fit)"] = pca2.transform(hybrid_raw)

    iso2 = Isomap(n_neighbors=15, n_components=2).fit(hybrid_raw[train_mask])
    embeddings_2d["Isomap (k=15, train-fit)"] = iso2.transform(hybrid_raw)

    spec2 = SpectralEmbedding(n_components=2, n_neighbors=15, random_state=seed)
    embeddings_2d["Spectral (k=15, explorative)"] = spec2.fit_transform(hybrid_raw)

    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=seed)
        reducer.fit(hybrid_raw[train_mask])
        embeddings_2d["UMAP (k=15, train-fit)"] = reducer.transform(hybrid_raw)
    except Exception:
        pass

    for name, Z in embeddings_2d.items():
        tw_train = float(trustworthiness(hybrid_raw[train_mask], Z[train_mask], n_neighbors=12))
        tw_all = float(trustworthiness(hybrid_raw, Z, n_neighbors=12))

        km2 = KMeans(n_clusters=2, n_init=20, random_state=seed).fit(Z[train_mask])
        km4 = KMeans(n_clusters=4, n_init=20, random_state=seed).fit(Z[train_mask])
        gmm4 = GaussianMixture(n_components=4, covariance_type="full", random_state=seed).fit(Z[train_mask])

        tr_l2 = km2.predict(Z[train_mask])
        va_l2 = km2.predict(Z[val_mask])
        tr_l4 = km4.predict(Z[train_mask])
        va_l4 = km4.predict(Z[val_mask])
        tr_g4 = gmm4.predict(Z[train_mask])
        va_g4 = gmm4.predict(Z[val_mask])

        summary_rows.append(
            {
                "manifold": name,
                "trustworthiness_train": tw_train,
                "trustworthiness_all": tw_all,
                "km2_train_sil_2d": safe_internal_metrics(Z[train_mask], tr_l2)["silhouette"],
                "km2_val_occupied": safe_internal_metrics(Z[val_mask], va_l2)["n_occupied"],
                "km4_train_sil_2d": safe_internal_metrics(Z[train_mask], tr_l4)["silhouette"],
                "km4_train_sil_morph": safe_internal_metrics(common_morph[train_mask], tr_l4)["silhouette"],
                "km4_val_occupied": safe_internal_metrics(Z[val_mask], va_l4)["n_occupied"],
                "gmm4_train_sil_2d": safe_internal_metrics(Z[train_mask], tr_g4)["silhouette"],
                "gmm4_val_occupied": safe_internal_metrics(Z[val_mask], va_g4)["n_occupied"],
            }
        )

    pca20 = PCA(n_components=20, random_state=seed).fit(hybrid_raw[train_mask])
    Z20_tr = pca20.transform(hybrid_raw[train_mask])
    bic_scores: Dict[int, float] = {}
    for k in range(1, 8):
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed).fit(Z20_tr)
        bic_scores[k] = float(gmm.bic(Z20_tr))
    best_bic_k = min(bic_scores, key=bic_scores.get)

    df_manifold = pd.DataFrame(summary_rows)
    df_manifold.attrs["gmm_pca20_bic_by_k"] = bic_scores
    df_manifold.attrs["gmm_pca20_best_bic_k"] = best_bic_k
    return df_manifold, embeddings_2d


def save_diagnostic_plots(
    out_dir: Path,
    data: Dict[str, np.ndarray],
    assignments_by_k: Dict[int, pd.DataFrame],
    df_ablation: pd.DataFrame,
    df_slides: pd.DataFrame,
    embeddings_2d: Dict[str, np.ndarray],
) -> None:
    """Save four diagnostic figures for technical review and reporting."""
    splits = data["split"].astype(str)
    train_mask = splits == "train"
    val_mask = splits == "val"

    # 1. Modality ablation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)
    for mod, grp in df_ablation.groupby("modality"):
        axes[0].plot(grp["k"], grp["train_sil_model_space"], marker="o", label=mod)
        axes[1].plot(grp["k"], grp["val_max_cluster_share"] * 100.0, marker="s", label=mod)
    axes[0].set_title("Train Silhouette by Modality and K")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Silhouette Score (Model Space)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Validation Cluster Collapse (RECHERCHE-016)")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Max Validation Share in Single Cluster (%)")
    axes[1].axhline(80, color="red", linestyle="--", alpha=0.5, label="80% Dominance Threshold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(out_dir / "01_modality_ablation.png")
    plt.close(fig)

    # 2. Slide domain shift
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)
    morph_raw = data["common_morph_raw"].astype(np.float64)
    feat_names = [str(x) for x in data["common_feature_names"]]
    od_idx = feat_names.index("mean_optical_density")
    rg_idx = feat_names.index("red_green_ratio")

    axes[0].scatter(
        morph_raw[train_mask, od_idx],
        morph_raw[train_mask, rg_idx],
        c="#1f77b4",
        alpha=0.55,
        s=22,
        label="Train (6 slides, N=411)",
    )
    axes[0].scatter(
        morph_raw[val_mask, od_idx],
        morph_raw[val_mask, rg_idx],
        c="#d62728",
        marker="^",
        s=42,
        alpha=0.85,
        label="Val RECHERCHE-016 (N=57)",
    )
    axes[0].set_title("Staining Shift: Optical Density vs Red/Green Ratio")
    axes[0].set_xlabel("Mean Optical Density (PAS)")
    axes[0].set_ylabel("Red / Green Ratio")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    colors_bar = ["#1f77b4" if s == "train" else "#d62728" for s in df_slides["split"]]
    axes[1].bar(df_slides["slide_id"], df_slides["k2_dist_median"], color=colors_bar)
    axes[1].set_title("Median Distance to Train Centroids (K=2)")
    axes[1].set_ylabel("Euclidean Distance in PCA20 Space")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "02_slide_domain_shift.png")
    plt.close(fig)

    # 3. Manifold comparison
    n_m = len(embeddings_2d)
    fig, axes = plt.subplots(1, n_m, figsize=(5.2 * n_m, 4.6), dpi=200)
    axes_list = np.atleast_1d(axes).ravel().tolist()
    k4_labels = assignments_by_k[4]["class_label"].astype(str).to_numpy()
    palette = {"A": "#2ca02c", "B": "#17becf", "C": "#ff7f0e", "D": "#d62728"}

    for ax, (m_name, Z) in zip(axes_list, embeddings_2d.items()):
        for lbl in ("A", "B", "C", "D"):
            m_tr = train_mask & (k4_labels == lbl)
            m_va = val_mask & (k4_labels == lbl)
            if np.any(m_tr):
                ax.scatter(Z[m_tr, 0], Z[m_tr, 1], c=palette[lbl], s=18, alpha=0.55, label=f"{lbl} (train)")
            if np.any(m_va):
                ax.scatter(
                    Z[m_va, 0],
                    Z[m_va, 1],
                    c=palette[lbl],
                    marker="^",
                    edgecolors="black",
                    linewidths=0.6,
                    s=46,
                    label=f"{lbl} (val)",
                )
        ax.set_title(m_name)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(True, alpha=0.25)
    axes_list[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "03_manifold_comparison.png")
    plt.close(fig)

    # 4. Key morphological profiles across K=4 clusters
    df_k4 = pd.DataFrame(morph_raw, columns=feat_names)
    df_k4["split"] = splits
    df_k4["cluster_k4"] = k4_labels
    if "peripheral_clear_fraction" in assignments_by_k[4].columns:
        df_k4["peripheral_clear_fraction"] = assignments_by_k[4]["peripheral_clear_fraction"].astype(float)

    key_feats = [
        f
        for f in ("peripheral_clear_fraction", "mean_optical_density", "stain_heterogeneity", "glom_area_um2")
        if f in df_k4.columns
    ]
    fig, axes = plt.subplots(1, len(key_feats), figsize=(4.2 * len(key_feats), 4.2), dpi=200)
    axes_k4 = np.atleast_1d(axes).ravel().tolist()
    for ax, feat in zip(axes_k4, key_feats):
        groups = [df_k4.loc[train_mask & (df_k4["cluster_k4"] == lbl), feat].values for lbl in ("A", "B", "C", "D")]
        ax.boxplot(groups, showfliers=False)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["A", "B", "C", "D"])
        val_sub = df_k4.loc[val_mask]
        x_pos = [{"A": 1, "B": 2, "C": 3, "D": 4}[lbl] for lbl in val_sub["cluster_k4"]]
        ax.scatter(x_pos, val_sub[feat], color="#d62728", alpha=0.7, s=22, zorder=4, label="Val (016)")
        ax.set_title(feat)
        ax.grid(True, axis="y", alpha=0.3)
    axes_k4[0].legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "04_morphological_profiles_k4.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clustering and manifold learning diagnostics (train/val).")
    parser.add_argument("--run-root", type=Path, default=Path("runs/clustering"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/clustering_diagnostics"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, assignments_by_k = load_run_artifacts(args.run_root)

    splits = data["split"].astype(str)
    train_mask = splits == "train"
    val_mask = splits == "val"

    df_ablation = run_modality_ablation(data, train_mask, val_mask, seed=args.seed)
    df_slides = run_slide_shift_audit(data, assignments_by_k[2], train_mask)
    df_loadings, df_assoc = run_feature_associations(data, assignments_by_k, train_mask)
    df_manifold, embeddings_2d = run_manifold_comparison(data, train_mask, val_mask, seed=args.seed)

    save_diagnostic_plots(args.output_dir, data, assignments_by_k, df_ablation, df_slides, embeddings_2d)

    df_ablation.to_csv(args.output_dir / "modality_ablation.csv", index=False)
    df_slides.to_csv(args.output_dir / "slide_shift_audit.csv", index=False)
    df_loadings.to_csv(args.output_dir / "morph_pca_loadings.csv")
    df_assoc.to_csv(args.output_dir / "feature_cluster_associations.csv", index=False)
    df_manifold.to_csv(args.output_dir / "manifold_comparison.csv", index=False)

    summary_payload = {
        "n_train": int(np.sum(train_mask)),
        "n_val": int(np.sum(val_mask)),
        "gmm_pca20_bic_by_k": df_manifold.attrs["gmm_pca20_bic_by_k"],
        "gmm_pca20_best_bic_k": df_manifold.attrs["gmm_pca20_best_bic_k"],
        "ablation": df_ablation.to_dict(orient="records"),
        "slides": df_slides.to_dict(orient="records"),
        "feature_associations": df_assoc.to_dict(orient="records"),
        "manifold_comparison": df_manifold.to_dict(orient="records"),
    }
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Clustering & Manifold Learning Diagnostics (Train/Val Cohort)",
        "",
        f"- **Samples analyzed**: {int(np.sum(train_mask))} train + {int(np.sum(val_mask))} val (`RECHERCHE-016`).",
        f"- **Optimal GMM BIC on PCA20 (train)**: K = {df_manifold.attrs['gmm_pca20_best_bic_k']}.",
        "",
        "## 1. Modality Ablation (Morph-only vs Deep-only vs Hybrid)",
        df_to_markdown(df_ablation, floatfmt=".4f"),
        "",
        "## 2. Slide-Level Domain Shift Audit",
        df_to_markdown(df_slides, floatfmt=".3f"),
        "",
        "## 3. Morphological Feature <-> Cluster Associations (Kruskal-Wallis eta^2 on Train)",
        df_to_markdown(df_assoc, floatfmt=".4f"),
        "",
        "## 4. Manifold Learning Comparison (PCA vs Isomap vs Spectral / UMAP)",
        df_to_markdown(df_manifold, floatfmt=".4f"),
        "",
    ]
    (args.output_dir / "diagnostic_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n=== DIAGNOSTICS COMPLETED ===")
    print(f"Report saved to: {args.output_dir / 'diagnostic_report.md'}")
    print("\n--- Modality Ablation ---")
    print(df_ablation.to_string(index=False))
    print("\n--- Slide Shift Audit ---")
    print(df_slides.to_string(index=False))
    print("\n--- Feature Associations ---")
    print(df_assoc.to_string(index=False))
    print("\n--- Manifold Comparison ---")
    print(df_manifold.to_string(index=False))


if __name__ == "__main__":
    main()
