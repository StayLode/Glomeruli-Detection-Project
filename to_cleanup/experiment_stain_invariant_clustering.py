#!/usr/bin/env python3
"""Controlled Step-2 experiment: Stain-Invariant Clustering benchmark on train/val."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering.labels import fit_label_mapping

MORPH_10_COLS = [
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

SHAPE_7_COLS = [
    "glom_area_um2",
    "glom_perimeter_um",
    "circularity",
    "solidity",
    "extent",
    "aspect_ratio",
    "peripheral_clear_fraction",
]

STRUCT_8_COLS = SHAPE_7_COLS + ["normalized_texture_cv"]


def normalized_entropy(counts: np.ndarray, k: int) -> float:
    total = float(np.sum(counts))
    if total <= 0 or k <= 1:
        return 0.0
    probs = counts[counts > 0] / total
    ent = -float(np.sum(probs * np.log(probs)))
    return ent / float(np.log(k))


def compute_cramers_v(slide_ids: np.ndarray, cluster_labels: np.ndarray) -> float:
    table = pd.crosstab(pd.Series(cluster_labels), pd.Series(slide_ids)).to_numpy()
    if min(table.shape) < 2:
        return 0.0
    chi2 = float(chi2_contingency(table, correction=False)[0])
    denom = float(table.sum()) * float(min(table.shape[0] - 1, table.shape[1] - 1))
    return float(np.sqrt(chi2 / denom)) if denom > 0.0 else 0.0


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        formatted = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                formatted.append(f"{val:.4f}" if np.isfinite(val) else "nan")
            else:
                formatted.append(str(val))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def load_aligned_train_val_data(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    assignments_path = run_dir / "assignments.csv"
    features_path = run_dir / "features.npz"
    if not assignments_path.is_file():
        raise FileNotFoundError(f"Missing assignments file: {assignments_path}")
    if not features_path.is_file():
        raise FileNotFoundError(f"Missing features archive: {features_path}")

    df = pd.read_csv(assignments_path)
    allowed_splits = {"train", "val"}
    if not set(df["split"].astype(str)).issubset(allowed_splits):
        raise ValueError("Unexpected split found; test slides must remain excluded.")

    with np.load(features_path, allow_pickle=False) as archive:
        npz_ids = archive["annotation_id"].astype(str)
        deep_raw = np.asarray(archive["deep_features_raw"], dtype=np.float64)
        morph_raw = np.asarray(archive["morphological_features_raw"], dtype=np.float64)
        if "common_feature_names" in archive.files:
            morph_names = archive["common_feature_names"].astype(str).tolist()
        else:
            morph_names = list(MORPH_10_COLS)

    id_to_idx = {aid: idx for idx, aid in enumerate(npz_ids)}
    if len(id_to_idx) != len(npz_ids):
        raise ValueError("Duplicate annotation_id values detected in features.npz")

    df_ids = df["annotation_id"].astype(str).tolist()
    if set(df_ids) != set(id_to_idx):
        raise ValueError("Mismatch between annotation_id sets in assignments.csv and features.npz")

    order = np.asarray([id_to_idx[aid] for aid in df_ids], dtype=int)
    deep_aligned = deep_raw[order]
    morph_aligned = morph_raw[order]

    for j, col_name in enumerate(morph_names):
        if col_name not in df.columns:
            df[col_name] = morph_aligned[:, j]

    missing_cols = [c for c in MORPH_10_COLS + ["peripheral_clear_fraction"] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required morphological columns: {missing_cols}")

    df["normalized_texture_cv"] = df["stain_heterogeneity"] / (df["mean_intensity"] + 1e-6)
    return df, deep_aligned


def try_add_bw_crop_features(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if "image_path" not in df.columns or "mask_path" not in df.columns:
        return df, False

    sample_rel = str(df.iloc[0]["image_path"])
    sample_path = Path(sample_rel) if Path(sample_rel).is_absolute() else (PROJECT_ROOT / sample_rel)
    if not sample_path.is_file():
        return df, False

    try:
        import cv2
        from src.clustering.crops import square_padded_crop
    except Exception:
        return df, False

    bw_grad_mean = []
    bw_grad_std = []
    bw_lap_std = []
    bw_iqr = []

    for _, row in df.iterrows():
        img_rel = str(row["image_path"])
        msk_rel = str(row["mask_path"])
        img_path = Path(img_rel) if Path(img_rel).is_absolute() else (PROJECT_ROOT / img_rel)
        msk_path = Path(msk_rel) if Path(msk_rel).is_absolute() else (PROJECT_ROOT / msk_rel)
        if not img_path.is_file() or not msk_path.is_file():
            return df, False

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or msk is None:
            return df, False

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bbox_raw = json.loads(str(row["bbox_patch"]))
        bbox = (int(round(bbox_raw[0])), int(round(bbox_raw[1])), int(round(bbox_raw[2])), int(round(bbox_raw[3])))
        crop_rgb, _, _ = square_padded_crop(rgb, bbox, 512, margin_ratio=0.25, padding_mode="reflect", is_mask=False)
        crop_msk, _, _ = square_padded_crop((msk > 0).astype(np.uint8), bbox, 512, margin_ratio=0.25, padding_mode="reflect", is_mask=True)

        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        fg = gray[crop_msk > 0]
        if fg.size < 10:
            fg = gray.reshape(-1)
        mu = float(np.mean(fg))
        sigma = float(np.std(fg)) + 1e-6
        gray_norm = (gray - mu) / sigma

        gx = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(gx * gx + gy * gy)[crop_msk > 0]
        lap = cv2.Laplacian(gray_norm, cv2.CV_32F, ksize=3)[crop_msk > 0]
        fg_norm = gray_norm[crop_msk > 0]

        bw_grad_mean.append(float(np.mean(gmag)) if gmag.size else 0.0)
        bw_grad_std.append(float(np.std(gmag)) if gmag.size else 0.0)
        bw_lap_std.append(float(np.std(lap)) if lap.size else 0.0)
        bw_iqr.append(float(np.percentile(fg_norm, 75) - np.percentile(fg_norm, 25)) if fg_norm.size else 0.0)

    out = df.copy()
    out["bw_norm_grad_mean"] = bw_grad_mean
    out["bw_norm_grad_std"] = bw_grad_std
    out["bw_norm_lap_std"] = bw_lap_std
    out["bw_norm_iqr"] = bw_iqr
    return out, True


def run_benchmark(reference_run_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df, embeddings = load_aligned_train_val_data(reference_run_dir)
    df, has_bw_features = try_add_bw_crop_features(df)

    train_mask = (df["split"].astype(str) == "train").to_numpy()
    val_mask = (df["split"].astype(str) == "val").to_numpy()

    df_train = df.loc[train_mask].reset_index(drop=True)
    df_val = df.loc[val_mask].reset_index(drop=True)

    variants: dict[str, dict[str, Any]] = {
        "1_baseline_hybrid_pca20": {"type": "hybrid_pca20", "algo": "kmeans"},
        "2_morph_10d_raw": {"type": "tabular", "cols": MORPH_10_COLS, "algo": "kmeans"},
        "3_stain_inv_shape_7d": {"type": "tabular", "cols": SHAPE_7_COLS, "algo": "kmeans"},
        "4_stain_inv_struct_8d": {"type": "tabular", "cols": STRUCT_8_COLS, "algo": "kmeans"},
        "5_stain_inv_struct_8d_gmm": {"type": "tabular", "cols": STRUCT_8_COLS, "algo": "gmm"},
    }
    if has_bw_features:
        bw_cols = STRUCT_8_COLS + ["bw_norm_grad_mean", "bw_norm_grad_std", "bw_norm_lap_std", "bw_norm_iqr"]
        variants["6_bw_norm_struct_12d"] = {"type": "tabular", "cols": bw_cols, "algo": "kmeans"}
    else:
        print("Local WSI image patches not found; running 5 tabular/embedding benchmark variants.")

    rows = []
    k4_details: dict[str, Any] = {}

    for v_name, cfg in variants.items():
        if cfg["type"] == "hybrid_pca20":
            X_raw_tr = np.hstack([embeddings[train_mask], df_train[MORPH_10_COLS].to_numpy(dtype=np.float64)])
            X_raw_va = np.hstack([embeddings[val_mask], df_val[MORPH_10_COLS].to_numpy(dtype=np.float64)])
            scaler = StandardScaler().fit(X_raw_tr)
            pca = PCA(n_components=20, random_state=42).fit(scaler.transform(X_raw_tr))
            Z_tr = pca.transform(scaler.transform(X_raw_tr))
            Z_va = pca.transform(scaler.transform(X_raw_va))
        else:
            cols = cfg["cols"]
            X_raw_tr = df_train[cols].to_numpy(dtype=np.float64)
            X_raw_va = df_val[cols].to_numpy(dtype=np.float64)
            scaler = StandardScaler().fit(X_raw_tr)
            Z_tr = scaler.transform(X_raw_tr)
            Z_va = scaler.transform(X_raw_va)

        for k in [2, 3, 4, 5, 6]:
            if cfg["algo"] == "kmeans":
                model = KMeans(n_clusters=k, n_init=30, random_state=42).fit(Z_tr)
                tr_raw_labels = model.predict(Z_tr)
                va_raw_labels = model.predict(Z_va)
            else:
                model = GaussianMixture(
                    n_components=k, covariance_type="full", n_init=10, reg_covar=1e-4, random_state=42
                ).fit(Z_tr)
                tr_raw_labels = model.predict(Z_tr)
                va_raw_labels = model.predict(Z_va)

            raw_to_letter, _ = fit_label_mapping(
                tr_raw_labels, df_train["peripheral_clear_fraction"].to_numpy(), k
            )
            letters = sorted(raw_to_letter.values())

            tr_letters = np.array([raw_to_letter[int(c)] for c in tr_raw_labels])
            va_letters = np.array([raw_to_letter[int(c)] for c in va_raw_labels])

            tr_sil = float(silhouette_score(Z_tr, tr_raw_labels))
            tr_ch = float(calinski_harabasz_score(Z_tr, tr_raw_labels))
            tr_db = float(davies_bouldin_score(Z_tr, tr_raw_labels))
            tr_cramers_v = compute_cramers_v(df_train["slide_id"].to_numpy(), tr_letters)

            va_counts = np.array([int(np.sum(va_letters == l)) for l in letters], dtype=int)
            tr_counts = np.array([int(np.sum(tr_letters == l)) for l in letters], dtype=int)
            va_collapse = float(np.max(va_counts) / len(va_letters))
            va_entropy = normalized_entropy(va_counts, k)
            va_occupied = int(np.sum(va_counts > 0))

            tr_meds = []
            va_meds = []
            for idx_l, l in enumerate(letters):
                if va_counts[idx_l] >= 2:
                    tr_meds.append(float(np.median(df_train.loc[tr_letters == l, "peripheral_clear_fraction"])))
                    va_meds.append(float(np.median(df_val.loc[va_letters == l, "peripheral_clear_fraction"])))
            if len(va_meds) >= 2:
                rho = float(spearmanr(tr_meds, va_meds)[0])
            else:
                rho = np.nan

            rows.append(
                {
                    "variant": v_name,
                    "k": k,
                    "train_silhouette": tr_sil,
                    "train_ch": tr_ch,
                    "train_db": tr_db,
                    "train_cramers_v_slide": tr_cramers_v,
                    "val_max_cluster_frac": va_collapse,
                    "val_norm_entropy": va_entropy,
                    "val_occupied_clusters": va_occupied,
                    "val_rim_order_spearman": rho,
                    "train_counts": json.dumps(dict(zip(letters, tr_counts.tolist()))),
                    "val_counts": json.dumps(dict(zip(letters, va_counts.tolist()))),
                }
            )

            if k == 4:
                k4_details[v_name] = {
                    "tr_letters": tr_letters,
                    "va_letters": va_letters,
                    "letters": letters,
                    "tr_counts": dict(zip(letters, tr_counts.tolist())),
                    "va_counts": dict(zip(letters, va_counts.tolist())),
                    "tr_sil": tr_sil,
                    "tr_cramers_v": tr_cramers_v,
                    "va_collapse": va_collapse,
                    "va_entropy": va_entropy,
                    "val_rim_order_spearman": rho,
                }

    res_df = pd.DataFrame(rows)
    res_df.to_csv(out_dir / "benchmark_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    palette = {
        "1_baseline_hybrid_pca20": "#d62728",
        "2_morph_10d_raw": "#ff7f0e",
        "3_stain_inv_shape_7d": "#2ca02c",
        "4_stain_inv_struct_8d": "#1f77b4",
        "5_stain_inv_struct_8d_gmm": "#9467bd",
        "6_bw_norm_struct_12d": "#17becf",
    }

    for v_name, grp in res_df.groupby("variant"):
        c = palette[v_name]
        axes[0].plot(grp["k"], grp["train_silhouette"], marker="o", lw=2, label=v_name, color=c)
        axes[1].plot(grp["k"], grp["val_max_cluster_frac"] * 100.0, marker="o", lw=2, label=v_name, color=c)
        axes[2].plot(grp["k"], grp["train_cramers_v_slide"], marker="o", lw=2, label=v_name, color=c)

    axes[0].set_title("Train Silhouette Score (Higher = Better)")
    axes[0].set_xlabel("K (Number of Clusters)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].grid(alpha=0.3)

    axes[1].axhline(80.0, color="black", linestyle="--", lw=1.2, label="80% Collapse Threshold")
    axes[1].set_title("Validation Cluster Collapse (Lower = Better)")
    axes[1].set_xlabel("K (Number of Clusters)")
    axes[1].set_ylabel("Max Cluster Share on Val (%)")
    axes[1].set_ylim(20, 105)
    axes[1].grid(alpha=0.3)

    axes[2].axhline(0.30, color="black", linestyle="--", lw=1.2, label="0.30 Slide Confounding Limit")
    axes[2].set_title("Train Slide Confounding Cramér's V (Lower = Better)")
    axes[2].set_xlabel("K (Number of Clusters)")
    axes[2].set_ylabel("Cramér's V (Cluster vs Slide)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "01_stain_invariant_benchmark.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    for idx, (v_name, info) in enumerate(k4_details.items()):
        ax = axes[idx]
        letters = info["letters"]
        tr_pct = [info["tr_counts"][l] / len(df_train) * 100.0 for l in letters]
        va_pct = [info["va_counts"][l] / len(df_val) * 100.0 for l in letters]
        x = np.arange(len(letters))
        ax.bar(x - 0.18, tr_pct, width=0.36, label="Train (6 slides)", color="#4c78a8")
        ax.bar(x + 0.18, va_pct, width=0.36, label="Val (RECHERCHE-016)", color="#e45756")
        ax.set_xticks(x)
        ax.set_xticklabels(letters)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Share (%)")
        ax.set_title(
            f"{v_name}\nSil={info['tr_sil']:.3f} | V_slide={info['tr_cramers_v']:.3f} | ValMax={info['va_collapse']*100:.1f}%",
            fontsize=9.5,
        )
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    for j in range(len(k4_details), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "02_k4_train_vs_val_distributions.png", dpi=180)
    plt.close(fig)

    md_lines = [
        "# Step 2: Stain-Invariant Clustering Benchmark (Train vs Validation)",
        "",
        "Fit strictly on `train` (N=411, 6 slides); frozen evaluation on `val` (`RECHERCHE-016`, N=57). Test slides (`015`, `017`) untouched.",
        "",
        "## Summary Table for K=4",
        "",
        "| Variant | Train Sil | Train CH | Train Cramér's V (Slide) | Val Max Cluster % | Val Norm Entropy | Val Occupied | Val Rim Order Spearman | Val Counts |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    k4_sub = res_df[res_df["k"] == 4]
    for _, r in k4_sub.iterrows():
        md_lines.append(
            f"| {r['variant']} | {r['train_silhouette']:.4f} | {r['train_ch']:.1f} | "
            f"{r['train_cramers_v_slide']:.4f} | {r['val_max_cluster_frac']*100:.1f}% | "
            f"{r['val_norm_entropy']:.3f} | {r['val_occupied_clusters']}/4 | "
            f"{r['val_rim_order_spearman']:.3f} | `{r['val_counts']}` |"
        )
    md_lines.extend(["", "## Full Table (K=2..6)", ""])
    md_lines.append(dataframe_to_markdown(res_df))
    (out_dir / "benchmark_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved Step 2 benchmark artifacts to: {out_dir}")
    print("\n--- K=4 COMPARISON ---")
    print(
        k4_sub[
            [
                "variant",
                "train_silhouette",
                "train_cramers_v_slide",
                "val_max_cluster_frac",
                "val_norm_entropy",
                "val_occupied_clusters",
                "val_counts",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run_benchmark(
        reference_run_dir=PROJECT_ROOT / "runs" / "clustering" / "clustering_k4",
        out_dir=PROJECT_ROOT / "runs" / "clustering_step2_benchmark",
    )
