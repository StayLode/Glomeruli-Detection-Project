#!/usr/bin/env python3
"""
Step 3: Leave-One-Slide-Out (LOSO) Cross-Validation & Full-Cohort Stability Audit
for Stain-Invariant Glomeruli Phenotyping across all 7 development WSIs (N=468).

Strictly excludes test WSIs (RECHERCHE-015, RECHERCHE-017).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = Path("runs/clustering/clustering_k4/features.npz")
OUT_DIR = Path("runs/clustering_loso")
DEV_SLIDES = {
    "RECHERCHE-003",
    "RECHERCHE-004",
    "RECHERCHE-005",
    "RECHERCHE-009",
    "RECHERCHE-010",
    "RECHERCHE-011",
    "RECHERCHE-016",
}
TEST_SLIDES = {"RECHERCHE-015", "RECHERCHE-017"}

CANONICAL_MORPH_NAMES = [
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

# Pure geometric + internal texture features (excluding slide-dependent color/brightness)
STAIN_INVARIANT_NAMES = [
    "glom_area_um2",
    "glom_perimeter_um",
    "circularity",
    "solidity",
    "extent",
    "aspect_ratio",
    "stain_heterogeneity",
]


def load_development_data(npz_path: Path) -> Dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing {npz_path}. Run: git lfs pull -I '{npz_path}'"
        )
    with np.load(npz_path, allow_pickle=False) as archive:
        keys = list(archive.files)
        data = {k: archive[k] for k in keys}

    slide_ids = data["slide_id"].astype(str)
    if set(slide_ids) & TEST_SLIDES:
        raise ValueError("Leakage guard triggered: test slides present in features.npz!")
    if set(slide_ids) != DEV_SLIDES:
        raise ValueError(f"Unexpected slide set: {sorted(set(slide_ids))}")

    morph_raw = data["morphological_features_raw"].astype(np.float64)
    deep_raw = data["deep_features_raw"].astype(np.float64)

    if "morphological_feature_names" in data:
        morph_names = [str(x) for x in data["morphological_feature_names"]]
    else:
        morph_names = CANONICAL_MORPH_NAMES[: morph_raw.shape[1]]

    df_morph = pd.DataFrame(morph_raw, columns=morph_names)

    # Include peripheral_clear_fraction if stored in archive
    has_rim = "peripheral_clear_fraction" in data
    if has_rim:
        df_morph["peripheral_clear_fraction"] = data[
            "peripheral_clear_fraction"
        ].astype(np.float64)

    inv_cols = list(STAIN_INVARIANT_NAMES)
    if has_rim and "peripheral_clear_fraction" not in inv_cols:
        inv_cols.append("peripheral_clear_fraction")

    return {
        "annotation_id": data["annotation_id"].astype(str),
        "slide_id": slide_ids,
        "split": data["split"].astype(str),
        "deep_raw": deep_raw,
        "morph_df": df_morph,
        "inv_cols": inv_cols,
        "has_rim": has_rim,
        "npz_keys": keys,
    }


def order_labels_by_proxy(
    labels: np.ndarray, proxy_values: np.ndarray, k: int
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Order clusters A, B, C, ... deterministically on training data by descending
    median of the ordering proxy (e.g., peripheral_clear_fraction or circularity).
    """
    medians = {}
    for cid in range(k):
        mask = labels == cid
        medians[cid] = float(np.median(proxy_values[mask])) if np.any(mask) else -1e9
    ordered_cids = sorted(range(k), key=lambda c: (-medians[c], c))
    cid_to_letter = {cid: chr(65 + idx) for idx, cid in enumerate(ordered_cids)}
    letter_labels = np.array([cid_to_letter[int(c)] for c in labels], dtype=object)
    return letter_labels, cid_to_letter


def fit_and_predict_model(
    model_name: str,
    k: int,
    X_deep_tr: np.ndarray,
    X_morph_all_tr: np.ndarray,
    X_inv_tr: np.ndarray,
    X_deep_va: np.ndarray,
    X_morph_all_va: np.ndarray,
    X_inv_va: np.ndarray,
    proxy_tr: np.ndarray,
    seed: int = 42,
) -> Dict:
    """Fit transform + clusterer strictly on train fold, predict frozen on val fold."""
    if model_name == "baseline_hybrid_pca20_kmeans":
        sc_d = StandardScaler().fit(X_deep_tr)
        sc_m = StandardScaler().fit(X_morph_all_tr)
        d_tr = sc_d.transform(X_deep_tr) / np.sqrt(X_deep_tr.shape[1])
        m_tr = sc_m.transform(X_morph_all_tr) / np.sqrt(X_morph_all_tr.shape[1])
        H_tr = np.hstack([d_tr, m_tr])

        d_va = sc_d.transform(X_deep_va) / np.sqrt(X_deep_va.shape[1])
        m_va = sc_m.transform(X_morph_all_va) / np.sqrt(X_morph_all_va.shape[1])
        H_va = np.hstack([d_va, m_va])

        pca = PCA(n_components=20, random_state=seed).fit(H_tr)
        Z_tr = pca.transform(H_tr)
        Z_va = pca.transform(H_va)

        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z_tr)
        raw_tr = km.predict(Z_tr)
        raw_va = km.predict(Z_va)
    else:
        scaler = StandardScaler().fit(X_inv_tr)
        Z_tr = scaler.transform(X_inv_tr)
        Z_va = scaler.transform(X_inv_va)

        if model_name == "stain_inv_kmeans":
            clf = KMeans(n_clusters=k, n_init=30, random_state=seed).fit(Z_tr)
            raw_tr = clf.predict(Z_tr)
            raw_va = clf.predict(Z_va)
        elif model_name == "stain_inv_gmm_full":
            clf = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=10,
                reg_covar=1e-4,
                random_state=seed,
            ).fit(Z_tr)
            raw_tr = clf.predict(Z_tr)
            raw_va = clf.predict(Z_va)
        elif model_name == "stain_inv_gmm_tied":
            clf = GaussianMixture(
                n_components=k,
                covariance_type="tied",
                n_init=10,
                reg_covar=1e-4,
                random_state=seed,
            ).fit(Z_tr)
            raw_tr = clf.predict(Z_tr)
            raw_va = clf.predict(Z_va)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    letters_tr, cid_to_letter = order_labels_by_proxy(raw_tr, proxy_tr, k)
    letters_va = np.array([cid_to_letter[int(c)] for c in raw_va], dtype=object)

    return {
        "Z_tr": Z_tr,
        "Z_va": Z_va,
        "letters_tr": letters_tr,
        "letters_va": letters_va,
    }


def compute_cramers_v(labels: np.ndarray, slides: np.ndarray) -> float:
    table = pd.crosstab(pd.Series(labels), pd.Series(slides))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, c = table.shape
    return float(np.sqrt(chi2 / (n * (min(r - 1, c - 1)))))


def min_matched_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes_a = np.unique(y_true)
    classes_b = np.unique(y_pred)
    k = max(len(classes_a), len(classes_b))
    cost = np.zeros((k, k), dtype=float)
    for i, ca in enumerate(classes_a):
        ma = y_true == ca
        for j, cb in enumerate(classes_b):
            mb = y_pred == cb
            inter = np.logical_and(ma, mb).sum()
            union = np.logical_or(ma, mb).sum()
            cost[i, j] = -(inter / union) if union > 0 else 0.0
    row_ind, col_ind = linear_sum_assignment(cost)
    jaccards = [-cost[r, c] for r, c in zip(row_ind, col_ind)]
    return float(np.min(jaccards))


def audit_full_cohort_criteria(
    model_name: str,
    k: int,
    X_deep: np.ndarray,
    X_morph_all: np.ndarray,
    X_inv: np.ndarray,
    proxy: np.ndarray,
    slide_ids: np.ndarray,
    seed: int = 42,
) -> Dict:
    """Evaluate the 5 official project criteria across all 7 development WSIs (N=468)."""
    full_res = fit_and_predict_model(
        model_name, k, X_deep, X_morph_all, X_inv, X_deep, X_morph_all, X_inv, proxy, seed
    )
    Z = full_res["Z_tr"]
    labels = full_res["letters_tr"]
    n = len(labels)

    occ = len(np.unique(labels))
    sil = float(silhouette_score(Z, labels)) if occ >= 2 else np.nan
    db = float(davies_bouldin_score(Z, labels)) if occ >= 2 else np.nan
    ch = float(calinski_harabasz_score(Z, labels)) if occ >= 2 else np.nan

    # 1 & 2: Subsampling stability (20 x 80% subsamples)
    rng = np.random.default_rng(seed)
    aris, jaccards = [], []
    for b in range(20):
        sub_idx = np.sort(rng.choice(n, size=int(0.8 * n), replace=False))
        sub_res = fit_and_predict_model(
            model_name,
            k,
            X_deep[sub_idx],
            X_morph_all[sub_idx],
            X_inv[sub_idx],
            X_deep[sub_idx],
            X_morph_all[sub_idx],
            X_inv[sub_idx],
            proxy[sub_idx],
            seed + b + 1,
        )
        aris.append(adjusted_rand_score(labels[sub_idx], sub_res["letters_tr"]))
        jaccards.append(min_matched_jaccard(labels[sub_idx], sub_res["letters_tr"]))

    ari_mean = float(np.mean(aris))
    jaccard_min = float(np.min(jaccards))

    # 3: Silhouette Z-score vs covariance-matched Gaussian null (30 replicates)
    cov = np.cov(Z, rowvar=False)
    mean = np.mean(Z, axis=0)
    null_sils = []
    for b in range(30):
        Z_null = rng.multivariate_normal(mean, cov, size=n)
        if "gmm" in model_name:
            cov_type = "tied" if "tied" in model_name else "full"
            null_lbl = GaussianMixture(
                n_components=k, covariance_type=cov_type, n_init=3, reg_covar=1e-4, random_state=seed + b
            ).fit_predict(Z_null)
        else:
            null_lbl = KMeans(n_clusters=k, n_init=10, random_state=seed + b).fit_predict(Z_null)
        if len(np.unique(null_lbl)) >= 2:
            null_sils.append(silhouette_score(Z_null, null_lbl))
    sil_z = (
        float((sil - np.mean(null_sils)) / (np.std(null_sils) + 1e-12))
        if (not np.isnan(sil) and len(null_sils) > 5)
        else np.nan
    )

    # 4 & 5: Slide confounding across the 7 WSIs
    cv = compute_cramers_v(labels, slide_ids)
    dom_fracs = []
    for cl in np.unique(labels):
        cl_slides = slide_ids[labels == cl]
        _, counts = np.unique(cl_slides, return_counts=True)
        dom_fracs.append(counts.max() / len(cl_slides))
    max_dom_slide = float(max(dom_fracs))

    passed = sum([
        ari_mean >= 0.75,
        jaccard_min >= 0.60,
        (not np.isnan(sil_z) and sil_z >= 2.0),
        (not np.isnan(cv) and cv <= 0.30),
        max_dom_slide <= 0.80,
    ])

    return {
        "full_silhouette": sil,
        "full_db": db,
        "full_ch": ch,
        "ari_subsampling": ari_mean,
        "jaccard_min": jaccard_min,
        "silhouette_z": sil_z,
        "cramers_v": cv,
        "max_dominant_slide": max_dom_slide,
        "criteria_passed": f"{passed}/5",
        "full_labels": labels,
    }


def make_plots(
    df_loso: pd.DataFrame,
    df_morph: pd.DataFrame,
    winning_labels: np.ndarray,
    has_rim: bool,
) -> None:
    # Plot 1: LOSO Max Cluster Share (%) per Patient for K=4
    sub = df_loso[df_loso["k"] == 4].copy()
    slides = sorted(DEV_SLIDES)
    models = [
        ("baseline_hybrid_pca20_kmeans", "Baseline Hybrid PCA20 + K-Means", "#7f8c8d"),
        ("stain_inv_kmeans", "Stain-Invariant + K-Means", "#e67e22"),
        ("stain_inv_gmm_full", "Stain-Invariant + GMM (full)", "#2980b9"),
        ("stain_inv_gmm_tied", "Stain-Invariant + GMM (tied)", "#27ae60"),
    ]

    x = np.arange(len(slides))
    width = 0.20

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=160)
    for idx, (m_key, m_label, color) in enumerate(models):
        m_df = sub[sub["model"] == m_key].set_index("held_out_slide")
        vals = [m_df.loc[s, "val_max_share"] * 100.0 for s in slides]
        ax.bar(x + (idx - 1.5) * width, vals, width=width, label=m_label, color=color, edgecolor="white")

    ax.axhline(80.0, color="#c0392b", linestyle="--", linewidth=1.5, label="Collapse Threshold (80%)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("RECHERCHE-", "WSI-") for s in slides], fontsize=10, fontweight="bold")
    ax.set_ylabel("Max Single-Cluster Share on Held-Out Slide (%)", fontsize=10)
    ax.set_title("Leave-One-Slide-Out (LOSO) Validation across All 7 Patients (K=4)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_loso_collapse_comparison.png")
    plt.close(fig)

    # Plot 2: Morphological Profiles of Clusters A-D (Winning Model)
    df_prof = df_morph.copy()
    df_prof["cluster"] = winning_labels
    clusters = sorted(df_prof["cluster"].unique())

    features_to_plot = [
        ("circularity", "Circularity (4πA / P²)"),
        ("solidity", "Solidity (Area / Convex Area)"),
        ("glom_area_um2", "Glomerular Area (µm²)"),
        (
            "peripheral_clear_fraction" if has_rim else "stain_heterogeneity",
            "Peripheral Clear Fraction" if has_rim else "Intra-Glomerular Texture (Std)",
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), dpi=160)
    palette = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
    for ax, (col, title) in zip(axes, features_to_plot):
        box_data = [df_prof.loc[df_prof["cluster"] == c, col].to_numpy() for c in clusters]
        bp = ax.boxplot(box_data, tick_labels=clusters, patch_artist=True, widths=0.55)
        for patch, color in zip(bp["boxes"], palette[: len(clusters)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Assigned Cluster", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        "Morphological & Structural Profiles of Stain-Invariant Clusters (N=468 Development Glomeruli)",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_cluster_morphological_profiles.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_development_data(FEATURES_PATH)

    slide_ids = data["slide_id"]
    X_deep = data["deep_raw"]
    df_morph = data["morph_df"]
    inv_cols = data["inv_cols"]
    has_rim = data["has_rim"]

    X_morph_all = df_morph[CANONICAL_MORPH_NAMES].to_numpy(dtype=np.float64)
    X_inv = df_morph[inv_cols].to_numpy(dtype=np.float64)

    proxy_name = "peripheral_clear_fraction" if has_rim else "circularity"
    proxy = df_morph[proxy_name].to_numpy(dtype=np.float64)

    print(f"Loaded N={len(slide_ids)} glomeruli across {len(DEV_SLIDES)} development WSIs.")
    print(f"Stain-invariant feature set ({len(inv_cols)}D): {inv_cols}")
    print(f"Deterministic cluster ordering proxy: {proxy_name} (descending)\n")

    models = [
        "baseline_hybrid_pca20_kmeans",
        "stain_inv_kmeans",
        "stain_inv_gmm_full",
        "stain_inv_gmm_tied",
    ]
    k_values = [3, 4]
    slides_sorted = sorted(DEV_SLIDES)

    loso_rows = []
    summary_rows = []
    winning_labels_k4 = None

    for k in k_values:
        for m in models:
            fold_max_shares = []
            fold_occ = []
            fold_val_sils = []
            collapsed_slides = []

            for held_out in slides_sorted:
                va_mask = slide_ids == held_out
                tr_mask = ~va_mask

                res = fit_and_predict_model(
                    m,
                    k,
                    X_deep[tr_mask],
                    X_morph_all[tr_mask],
                    X_inv[tr_mask],
                    X_deep[va_mask],
                    X_morph_all[va_mask],
                    X_inv[va_mask],
                    proxy[tr_mask],
                    seed=42,
                )

                l_va = res["letters_va"]
                vals, cnts = np.unique(l_va, return_counts=True)
                counts_dict = {str(v): int(c) for v, c in zip(vals, cnts)}
                occ = len(vals)
                max_share = float(cnts.max() / len(l_va))
                val_sil = (
                    float(silhouette_score(res["Z_va"], l_va)) if occ >= 2 else np.nan
                )

                if max_share > 0.80:
                    collapsed_slides.append(held_out)

                fold_max_shares.append(max_share)
                fold_occ.append(occ)
                if not np.isnan(val_sil):
                    fold_val_sils.append(val_sil)

                loso_rows.append(
                    {
                        "k": k,
                        "model": m,
                        "held_out_slide": held_out,
                        "n_val": int(va_mask.sum()),
                        "val_occupied": occ,
                        "val_max_share": round(max_share, 4),
                        "collapsed_gt_80pct": max_share > 0.80,
                        "val_silhouette": round(val_sil, 4) if not np.isnan(val_sil) else np.nan,
                        "val_counts": json.dumps(counts_dict, sort_keys=True),
                    }
                )

            audit = audit_full_cohort_criteria(
                m, k, X_deep, X_morph_all, X_inv, proxy, slide_ids, seed=42
            )

            if k == 4 and m == "stain_inv_gmm_tied":
                winning_labels_k4 = audit["full_labels"]

            summary_rows.append(
                {
                    "k": k,
                    "model": m,
                    "loso_mean_occupied": round(float(np.mean(fold_occ)), 2),
                    "loso_mean_max_share": round(float(np.mean(fold_max_shares)), 4),
                    "loso_worst_max_share": round(float(np.max(fold_max_shares)), 4),
                    "loso_collapsed_slides_count": len(collapsed_slides),
                    "loso_mean_val_sil": round(float(np.mean(fold_val_sils)), 4) if fold_val_sils else np.nan,
                    "full_sil": round(audit["full_silhouette"], 4),
                    "ari_sub80": round(audit["ari_subsampling"], 4),
                    "jaccard_min": round(audit["jaccard_min"], 4),
                    "sil_z": round(audit["silhouette_z"], 2),
                    "cramers_v_slide": round(audit["cramers_v"], 4),
                    "max_dom_slide": round(audit["max_dominant_slide"], 4),
                    "criteria": audit["criteria_passed"],
                }
            )

    df_loso = pd.DataFrame(loso_rows)
    df_summary = pd.DataFrame(summary_rows)

    df_loso.to_csv(OUT_DIR / "loso_per_slide_results.csv", index=False)
    df_summary.to_csv(OUT_DIR / "loso_summary_comparison.csv", index=False)

    print("=== 1. LOSO & FULL-COHORT BENCHMARK SUMMARY (ALL 7 DEV WSIs) ===")
    print(df_summary.to_string(index=False))

    print("\n=== 2. PER-PATIENT DETAIL FOR K=4 (HELD-OUT SLIDE ASSIGNMENTS) ===")
    sub_k4 = df_loso[df_loso["k"] == 4][
        ["model", "held_out_slide", "n_val", "val_occupied", "val_max_share", "collapsed_gt_80pct", "val_counts"]
    ]
    print(sub_k4.to_string(index=False))

    if winning_labels_k4 is not None:
        df_prof = df_morph.copy()
        df_prof["cluster"] = winning_labels_k4
        cols_summary = [
            "glom_area_um2",
            "circularity",
            "solidity",
            "extent",
            "aspect_ratio",
            "stain_heterogeneity",
        ]
        if has_rim:
            cols_summary.append("peripheral_clear_fraction")

        prof_table = df_prof.groupby("cluster")[cols_summary].median().round(4)
        prof_table.insert(0, "n_glomeruli", df_prof.groupby("cluster").size())
        print("\n=== 3. MORPHOLOGICAL PROFILE OF CLUSTERS A-D (K=4, STAIN-INVARIANT GMM TIED) ===")
        print(prof_table.to_string())
        prof_table.to_csv(OUT_DIR / "cluster_morphological_profiles_k4.csv")

        make_plots(df_loso, df_morph, winning_labels_k4, has_rim)
        print(f"\nSaved plots and CSV tables to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
