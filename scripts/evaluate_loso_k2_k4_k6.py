from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

OUT_DIR = Path("runs/clustering_loso_k2_k4_k6")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [2, 4, 6]
MORPH_STAIN_FEATURES = [
    "glom_area_um2",
    "glom_perimeter_um",
    "circularity",
    "solidity",
    "extent",
    "aspect_ratio",
    "stain_heterogeneity",
]

# 1. Load the complete data (468 development glomeruli + 2 test slides)
dev_df = pd.read_csv("runs/clustering/clustering_k6/assignments.csv")
test15_df = pd.read_csv("runs/clustering_predictions/RECHERCHE-015/clustering_k6/assignments.csv")
test17_df = pd.read_csv("runs/clustering_predictions/RECHERCHE-017/clustering_k6/assignments.csv")

with np.load("runs/clustering/clustering_k6/features.npz", allow_pickle=False) as archive:
    required = {"annotation_id", "slide_id", "split", "model_features"}
    missing = required.difference(archive.files)
    if missing:
        raise ValueError(f"features.npz is missing required keys: {sorted(missing)}")

    feature_ids = archive["annotation_id"].astype(str)
    if len(feature_ids) != len(set(feature_ids)):
        raise ValueError("Duplicate annotation_id values detected in features.npz")

    dev_ids = dev_df["annotation_id"].astype(str).to_numpy()
    if len(dev_ids) != len(set(dev_ids)) or set(dev_ids) != set(feature_ids):
        raise ValueError("Feature arrays and assignments have different annotation_id sets")

    feature_index = {annotation_id: i for i, annotation_id in enumerate(feature_ids)}
    feature_order = np.asarray([feature_index[annotation_id] for annotation_id in dev_ids], dtype=np.intp)
    X_hybrid_all = archive["model_features"][feature_order]

    feature_splits = archive["split"].astype(str)[feature_order]
    feature_slides = archive["slide_id"].astype(str)[feature_order]
    if not np.array_equal(feature_splits, dev_df["split"].astype(str).to_numpy()):
        raise ValueError("Split metadata in features.npz does not match assignments.csv")
    if not np.array_equal(feature_slides, dev_df["slide_id"].astype(str).to_numpy()):
        raise ValueError("Slide metadata in features.npz does not match assignments.csv")
    if not set(feature_splits).issubset({"train", "val"}):
        raise ValueError("Test samples detected in features.npz; aborting to prevent data leakage.")
    if set(feature_slides) & {"RECHERCHE-015", "RECHERCHE-017"}:
        raise ValueError("Test slides detected in features.npz; aborting to prevent data leakage.")

assert len(dev_df) == len(X_hybrid_all), "Row count mismatch between assignments.csv and features.npz"

X_shape_all = dev_df[MORPH_STAIN_FEATURES].to_numpy(dtype=np.float64)
slides = dev_df["slide_id"].to_numpy()
unique_slides = sorted(dev_df["slide_id"].unique())


def assign_letters_by_pcf(raw_labels, pcf_values, k):
    """Orders clusters from A to F by descending median peripheral_clear_fraction."""
    order = sorted(
        range(k),
        key=lambda c: float(np.median(pcf_values[raw_labels == c])) if np.any(raw_labels == c) else -1.0,
        reverse=True,
    )
    return {old_c: chr(ord("A") + idx) for idx, old_c in enumerate(order)}


def evaluate_5_criteria(X, labels, slides_arr, k, seed=42):
    """Calculates the five project validation criteria across the full development dataset (N=468)."""
    sil = float(silhouette_score(X, labels))

    # 1 & 2: Stability using 80% subsamples (30 repetitions)
    rng = np.random.default_rng(seed)
    n = len(X)
    sub_n = int(0.8 * n)
    aris = []
    jaccards = [[] for _ in range(k)]

    for rep in range(30):
        idx = rng.choice(n, size=sub_n, replace=False)
        km_sub = KMeans(n_clusters=k, random_state=seed + rep, n_init=10)
        sub_labels = km_sub.fit_predict(X[idx])
        ref_labels = labels[idx]
        aris.append(adjusted_rand_score(ref_labels, sub_labels))

        # Optimal matching between reference clusters and subsample clusters
        cost = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                inter = np.sum((ref_labels == i) & (sub_labels == j))
                union = np.sum((ref_labels == i) | (sub_labels == j))
                cost[i, j] = -(inter / union) if union > 0 else 0.0
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            jaccards[r].append(-cost[r, c])

    ari_mean = float(np.mean(aris))
    jaccard_min = float(min(np.mean(j) for j in jaccards))

    # 3: Geometric significance compared with random data (z-score)
    cov = np.cov(X, rowvar=False)
    null_sils = []
    for rep in range(20):
        X_null = rng.multivariate_normal(X.mean(axis=0), cov, size=n)
        null_lbl = KMeans(n_clusters=k, random_state=seed + 500 + rep, n_init=5).fit_predict(X_null)
        if len(np.unique(null_lbl)) > 1:
            null_sils.append(silhouette_score(X_null, null_lbl))
    sil_z = float((sil - np.mean(null_sils)) / (np.std(null_sils) + 1e-8))

    # 4: Independence from slide (Cramér V)
    ct = pd.crosstab(labels, slides_arr)
    chi2 = chi2_contingency(ct)[0]
    min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0

    # 5: No cluster dominated by a single slide (>80%)
    max_dom_slide = float((ct.div(ct.sum(axis=1), axis=0)).max().max())

    passed = sum([
        ari_mean >= 0.75,
        jaccard_min >= 0.60,
        sil_z >= 2.0,
        cramers_v <= 0.30,
        max_dom_slide <= 0.80,
    ])
    return {
        "full_sil": round(sil, 4),
        "ari_sub80": round(ari_mean, 4),
        "jaccard_min": round(jaccard_min, 4),
        "sil_z": round(sil_z, 2),
        "cramers_v_slide": round(cramers_v, 4),
        "max_dom_slide": round(max_dom_slide, 4),
        "criteria": f"{passed}/5",
    }


summary_rows = []
per_slide_rows = []

for k in K_VALUES:
    for model_name in ["baseline_hybrid_pca20", "morphology_plus_stain_heterogeneity"]:
        # --- A. Leave-one-slide-out validation (one slide held out at a time) ---
        occ_list, max_share_list, val_sil_list = [], [], []
        collapsed_count = 0

        for held_slide in unique_slides:
            val_mask = slides == held_slide
            tr_mask = ~val_mask

            scaler = StandardScaler()
            if model_name == "baseline_hybrid_pca20":
                X_tr = scaler.fit_transform(X_hybrid_all[tr_mask])
                X_va = scaler.transform(X_hybrid_all[val_mask])
                pca = PCA(n_components=20, random_state=42)
                X_tr = pca.fit_transform(X_tr)
                X_va = pca.transform(X_va)
            else:
                X_tr = scaler.fit_transform(X_shape_all[tr_mask])
                X_va = scaler.transform(X_shape_all[val_mask])

            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            tr_raw = km.fit_predict(X_tr)
            va_raw = km.predict(X_va)

            lmap = assign_letters_by_pcf(tr_raw, dev_df.loc[tr_mask, "peripheral_clear_fraction"].to_numpy(), k)
            va_letters = np.array([lmap[c] for c in va_raw])

            counts = {chr(ord("A") + i): int(np.sum(va_letters == chr(ord("A") + i))) for i in range(k)}
            n_val = int(np.sum(val_mask))
            occ = sum(1 for v in counts.values() if v > 0)
            max_sh = max(counts.values()) / n_val
            collapsed = max_sh > 0.80
            if collapsed:
                collapsed_count += 1

            v_sil = float(silhouette_score(X_va, va_raw)) if len(np.unique(va_raw)) > 1 else np.nan
            occ_list.append(occ)
            max_share_list.append(max_sh)
            if not np.isnan(v_sil):
                val_sil_list.append(v_sil)

            per_slide_rows.append({
                "k": k,
                "model": model_name,
                "held_out_slide": held_slide,
                "n_val": n_val,
                "val_occupied": occ,
                "val_max_share": round(max_sh, 4),
                "collapsed_gt_80pct": collapsed,
                "val_silhouette": round(v_sil, 4) if not np.isnan(v_sil) else np.nan,
                "val_counts": json.dumps(counts),
            })

        # --- B. Validation criteria across the full development cohort (N=468) ---
        scaler_full = StandardScaler()
        if model_name == "baseline_hybrid_pca20":
            X_full = PCA(n_components=20, random_state=42).fit_transform(scaler_full.fit_transform(X_hybrid_all))
        else:
            X_full = scaler_full.fit_transform(X_shape_all)
        full_labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X_full)
        crit = evaluate_5_criteria(X_full, full_labels, slides, k)

        summary_rows.append({
            "k": k,
            "model": model_name,
            "loso_mean_occupied": round(float(np.mean(occ_list)), 2),
            "loso_mean_max_share": round(float(np.mean(max_share_list)), 4),
            "loso_worst_max_share": round(float(np.max(max_share_list)), 4),
            "loso_collapsed_slides": f"{collapsed_count}/7",
            "loso_mean_val_sil": round(float(np.mean(val_sil_list)), 4) if val_sil_list else np.nan,
            **crit,
        })

summary_df = pd.DataFrame(summary_rows)
per_slide_df = pd.DataFrame(per_slide_rows)
summary_df.to_csv(OUT_DIR / "loso_summary_k2_k4_k6.csv", index=False)
per_slide_df.to_csv(OUT_DIR / "loso_per_slide_k2_k4_k6.csv", index=False)

print("=== 1. COMPARISON: RESNET BASELINE + MORPHOLOGY/COLOR MEASUREMENTS VS MORPHOLOGY + STAIN HETEROGENEITY ===")
print(summary_df.to_string(index=False))

# === 2. K=6 EXPLORATORY GROUP PROFILES (A-F) ===
train_mask = dev_df["split"] == "train"
val_mask = dev_df["split"] == "val"

scaler_k6 = StandardScaler()
X_tr6 = scaler_k6.fit_transform(dev_df.loc[train_mask, MORPH_STAIN_FEATURES].to_numpy())
X_va6 = scaler_k6.transform(dev_df.loc[val_mask, MORPH_STAIN_FEATURES].to_numpy())
X_t15_6 = scaler_k6.transform(test15_df[MORPH_STAIN_FEATURES].to_numpy())
X_t17_6 = scaler_k6.transform(test17_df[MORPH_STAIN_FEATURES].to_numpy())

km6 = KMeans(n_clusters=6, random_state=42, n_init=20)
tr6_raw = km6.fit_predict(X_tr6)
va6_raw = km6.predict(X_va6)
t15_raw = km6.predict(X_t15_6)
t17_raw = km6.predict(X_t17_6)

lmap6 = assign_letters_by_pcf(tr6_raw, dev_df.loc[train_mask, "peripheral_clear_fraction"].to_numpy(), 6)
dev_df["shape_k6_label"] = ""
dev_df.loc[train_mask, "shape_k6_label"] = [lmap6[c] for c in tr6_raw]
dev_df.loc[val_mask, "shape_k6_label"] = [lmap6[c] for c in va6_raw]
test15_df["shape_k6_label"] = [lmap6[c] for c in t15_raw]
test17_df["shape_k6_label"] = [lmap6[c] for c in t17_raw]

letters_6 = ["A", "B", "C", "D", "E", "F"]
print("\n=== 2. DISTRIBUTION OF EXPLORATORY GROUPS A-F, ORDERED BY A PROXY, ACROSS TRAIN, VALIDATION, AND TEST ===")
dist_rows = []
for name, sub in [
    ("Train (6 slides, N=411)", dev_df[train_mask]),
    ("Validation WSI-016 (N=57)", dev_df[val_mask]),
    ("Test 1 WSI-015 (N=36)", test15_df),
    ("Test 2 WSI-017 (N=56)", test17_df),
]:
    counts = {lt: int((sub["shape_k6_label"] == lt).sum()) for lt in letters_6}
    max_sh = max(counts.values()) / len(sub)
    print(f"{name:28s} | counts={counts} | largest_group={max_sh*100:.1f}%")
    dist_rows.append({"split": name, "n": len(sub), **counts, "max_share_pct": round(max_sh * 100, 1)})
pd.DataFrame(dist_rows).to_csv(OUT_DIR / "k6_split_distributions.csv", index=False)

print("\n=== 3. PROFILES OF GROUPS A-F ACROSS 468 DEVELOPMENT GLOMERULI ===")
prof_rows = []
for lt in letters_6:
    sub = dev_df[dev_df["shape_k6_label"] == lt]
    prof_rows.append({
        "Group": lt,
        "N_glomeruli": len(sub),
        "Share_pct": round(100.0 * len(sub) / len(dev_df), 1),
        "Peripheral_clear_fraction_median": round(float(sub["peripheral_clear_fraction"].median()), 3),
        "Glomerular_area_um2_median": round(float(sub["glom_area_um2"].median()), 1),
        "Circularity_median": round(float(sub["circularity"].median()), 3),
        "Solidity_median": round(float(sub["solidity"].median()), 3),
        "Aspect_ratio_median": round(float(sub["aspect_ratio"].median()), 3),
        "Stain_heterogeneity_median": round(float(sub["stain_heterogeneity"].median()), 2),
    })
prof_df = pd.DataFrame(prof_rows)
prof_df.to_csv(OUT_DIR / "k6_morphological_profiles.csv", index=False)
print(prof_df.to_string(index=False))

# === 4. GENERATE THREE FIGURES FOR THE REPORT AND PRESENTATION ===
# Figure 1: LOSO comparison for K=2, 4, 6
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
slide_short = [s.replace("RECHERCHE-", "WSI-") for s in unique_slides]
x = np.arange(len(slide_short))
w = 0.36

for ax, k in zip(axes, K_VALUES):
    sub_base = per_slide_df[(per_slide_df["k"] == k) & (per_slide_df["model"] == "baseline_hybrid_pca20")].sort_values("held_out_slide")
    sub_new = per_slide_df[(per_slide_df["k"] == k) & (per_slide_df["model"] == "morphology_plus_stain_heterogeneity")].sort_values("held_out_slide")

    ax.bar(x - w/2, sub_base["val_max_share"] * 100, width=w, label="Baseline ResNet + morphology/color measurements", color="#7f8c8d")
    ax.bar(x + w/2, sub_new["val_max_share"] * 100, width=w, label="Morphology + stain heterogeneity", color="#e67e22")
    ax.axhline(80, color="#c0392b", linestyle="--", linewidth=1.4, label="Concentration threshold (80%)")
    ax.set_title(f"K = {k}" + (" (6 exploratory groups)" if k == 6 else ""), fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(slide_short, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    if k == 2:
        ax.set_ylabel("Largest-group share on held-out slide (%)", fontsize=10)
    if k == 6:
        ax.legend(loc="upper right", fontsize=8.5)

plt.suptitle("LOSO: largest-group share per slide (K=2, 4, 6)",
             fontsize=12.5, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_loso_collapse_k2_k4_k6.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 2: Morphological profiles of exploratory groups A-F
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
plot_specs = [
    ("circularity", "Circularity"),
    ("solidity", "Boundary solidity"),
    ("glom_area_um2", "Glomerular area (μm²)"),
    ("peripheral_clear_fraction", "Peripheral clear space"),
]
palette_6 = ["#27ae60", "#2ecc71", "#3498db", "#f1c40f", "#e67e22", "#e74c3c"]
for ax, (col, title) in zip(axes, plot_specs):
    data = [dev_df.loc[dev_df["shape_k6_label"] == lt, col].to_numpy() for lt in letters_6]
    bp = ax.boxplot(data, tick_labels=letters_6, patch_artist=True)
    for patch, c in zip(bp["boxes"], palette_6):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Group A-F (proxy order defined on the training set)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

plt.suptitle("Exploratory profiles of groups A-F (K=6, N=468)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_k6_morphological_profiles.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 3: Distribution of groups A-F across train, validation, and test
fig, ax = plt.subplots(figsize=(9.5, 4.6))
split_names = ["Train\n(6 slides, N=411)", "Validation\n(WSI-016, N=57)", "Test 1\n(WSI-015, N=36)", "Test 2\n(WSI-017, N=56)"]
sub_dfs = [dev_df[train_mask], dev_df[val_mask], test15_df, test17_df]
bottom = np.zeros(len(sub_dfs))

for lt, c in zip(letters_6, palette_6):
    vals = np.array([100.0 * (s["shape_k6_label"] == lt).sum() / len(s) for s in sub_dfs])
    bars = ax.bar(split_names, vals, bottom=bottom, label=f"Group {lt}", color=c, edgecolor="white", width=0.52)
    for bar, v in zip(bars, vals):
        if v >= 6.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + v / 2, f"{v:.1f}%",
                    ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
    bottom += vals

ax.set_ylabel("Percentage of glomeruli (%)", fontsize=10.5)
ax.set_ylim(0, 105)
ax.set_title("Distribution of groups A-F across train, validation, and test sets (K=6)",
             fontsize=11.5, fontweight="bold")
ax.legend(loc="upper right", ncol=2, fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_k6_split_distributions.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved CSV files and figures to:", OUT_DIR)
