#!/usr/bin/env python3
"""Generate visual prototype grids for the winning Stain-Invariant K-Means model (K=4 and K=6)."""

import importlib.util
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Carica direttamente src/clustering/crops.py evitando src/clustering/__init__.py (che importa torch)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CROPS_PATH = PROJECT_ROOT / "src" / "clustering" / "crops.py"
spec = importlib.util.spec_from_file_location("crops", CROPS_PATH)
crops_mod = importlib.util.module_from_spec(spec)
sys.modules["crops"] = crops_mod
spec.loader.exec_module(crops_mod)
square_padded_crop = crops_mod.square_padded_crop

OUT_DIR = Path("runs/clustering_loso")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAIN_INV_FEATURES = [
    "glom_area_um2",
    "glom_perimeter_um",
    "circularity",
    "solidity",
    "extent",
    "aspect_ratio",
    "stain_heterogeneity",
]

CLUSTER_DESCRIPTIONS_K4 = {
    "A": "Cluster A — Large, round, high clear rim",
    "B": "Cluster B — Compact, regular, preserved shape",
    "C": "Cluster C — Intermediate / mild remodeling",
    "D": "Cluster D — Irregular / lobulated contour",
}


def resolve_local_asset(raw_path: str, manifest_root: Path) -> Path:
    p_str = str(raw_path)
    if "dataset/" in p_str:
        p_str = p_str.split("dataset/", 1)[1]
    return manifest_root / p_str


def load_crop_and_contour(row: pd.Series, manifest_root: Path, crop_size: int = 512):
    img_path = resolve_local_asset(row["image_path"], manifest_root)
    mask_path = resolve_local_asset(row["mask_path"], manifest_root)
    img_bgr = cv2.imread(str(img_path))
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img_bgr is None or mask_gray is None:
        return None, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask_bin = (mask_gray > 127).astype(np.uint8)
    _, labels = cv2.connectedComponents(mask_bin, connectivity=8)
    comp_label = int(row["component_label"])
    selected_mask = (labels == comp_label).astype(np.uint8) * 255

    x1, y1, x2, y2 = map(int, json.loads(str(row["bbox_patch"])))
    crop, _, _ = square_padded_crop(img_rgb, (x1, y1, x2, y2), crop_size, 0.25, "reflect", False)
    crop_mask, _, _ = square_padded_crop(selected_mask, (x1, y1, x2, y2), crop_size, 0.25, "background", True)
    return crop, crop_mask


def fit_and_assign(df: pd.DataFrame, k: int, seed: int = 42) -> pd.DataFrame:
    out = df.copy()
    train_mask = out["split"] == "train"

    scaler = StandardScaler()
    X_train = scaler.fit_transform(out.loc[train_mask, STAIN_INV_FEATURES].to_numpy())
    X_all = scaler.transform(out[STAIN_INV_FEATURES].to_numpy())

    km = KMeans(n_clusters=k, random_state=seed, n_init=30)
    km.fit(X_train)
    raw_all = km.predict(X_all)
    raw_train = raw_all[train_mask.to_numpy()]

    # Deterministic ordering A -> final letter by descending train-median peripheral_clear_fraction
    train_pcf = out.loc[train_mask, "peripheral_clear_fraction"].to_numpy()
    order = sorted(range(k), key=lambda c: float(np.median(train_pcf[raw_train == c])), reverse=True)
    cid_to_letter = {old_c: chr(ord("A") + idx) for idx, old_c in enumerate(order)}

    dists = np.linalg.norm(X_all - km.cluster_centers_[raw_all], axis=1)
    out["cluster_letter"] = [cid_to_letter[int(c)] for c in raw_all]
    out["dist_to_centroid"] = dists
    return out


def build_grid(df: pd.DataFrame, k: int, n_cols: int, out_path: Path, title: str):
    letters = [chr(ord("A") + i) for i in range(k)]
    manifest_root = Path("dataset")
    loaded_count = 0

    fig, axes = plt.subplots(
        k, n_cols, figsize=(3.1 * n_cols, 3.25 * k), facecolor="white"
    )
    if k == 1:
        axes = np.expand_dims(axes, axis=0)

    for r_idx, letter in enumerate(letters):
        sub = df[df["cluster_letter"] == letter].copy()
        # Prefer unseen validation slide (RECHERCHE-016) first, then fill with closest train prototypes
        sub["is_val"] = (sub["split"] == "val").astype(int)
        sub = sub.sort_values(by=["is_val", "dist_to_centroid"], ascending=[False, True])
        chosen = sub.head(n_cols)

        for c_idx in range(n_cols):
            ax = axes[r_idx, c_idx]
            if c_idx >= len(chosen):
                ax.axis("off")
                continue

            row = chosen.iloc[c_idx]
            crop, crop_mask = load_crop_and_contour(row, manifest_root)
            if crop is None:
                ax.axis("off")
                continue

            loaded_count += 1
            # Overlay thin cyan contour of the segmented glomerulus mask
            vis = crop.copy()
            contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 220, 255), thickness=2)

            ax.imshow(vis)
            ax.set_xticks([])
            ax.set_yticks([])

            split_tag = "VAL-016" if row["split"] == "val" else row["slide_id"].replace("RECHERCHE-", "TR-")
            ax.set_title(
                f"{split_tag} | d={row['dist_to_centroid']:.2f}\n"
                f"circ={row['circularity']:.2f} | clear={row['peripheral_clear_fraction']:.2f}",
                fontsize=8.5,
                pad=4,
            )
            if c_idx == 0:
                ylabel = CLUSTER_DESCRIPTIONS_K4.get(letter, f"Cluster {letter}") if k == 4 else f"Group {letter} (N={len(sub)})"
                ax.set_ylabel(ylabel, fontsize=9.5, fontweight="bold", labelpad=8)

    if loaded_count == 0:
        print("ATTENZIONE: Nessuna immagine trovata in 'dataset/'. Verifica che i patch siano presenti localmente.")

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visual grid ({loaded_count} crops): {out_path}")


def main():
    assignments = pd.read_csv("runs/clustering/clustering_k4/assignments.csv")
    cohort = pd.read_csv("runs/clustering/cohort_train_val/cohort.csv")

    # Prende da cohort solo le colonne mancanti in assignments per evitare suffissi _x / _y
    extra_cols = [c for c in ["component_label", "bbox_patch", "image_path", "mask_path"] if c not in assignments.columns]
    merged = assignments.merge(
        cohort[["annotation_id"] + extra_cols],
        on="annotation_id",
        how="inner",
    )

    df_k4 = fit_and_assign(merged, k=4, seed=42)
    build_grid(
        df_k4,
        k=4,
        n_cols=5,
        out_path=OUT_DIR / "fig4_visual_grid_kmeans_k4.png",
        title="Winning Model Prototypes: Stain-Invariant K-Means (K=4, Clusters A–D)",
    )

    df_k6 = fit_and_assign(merged, k=6, seed=42)
    build_grid(
        df_k6,
        k=6,
        n_cols=4,
        out_path=OUT_DIR / "fig5_visual_grid_kmeans_k6.png",
        title="Exploratory 6-Stage Granularity: Stain-Invariant K-Means (K=6, Groups A–F)",
    )


if __name__ == "__main__":
    main()
