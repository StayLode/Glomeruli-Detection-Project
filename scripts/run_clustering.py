#!/usr/bin/env python3
"""
CLI Script for Unsupervised Glomeruli Clustering and Manifold Learning (Step 3).

Usage:
    python scripts/run_clustering.py --config configs/clustering_config.yaml --device 0
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering.feature_extractor import GlomeruliFeatureExtractor
from src.clustering.manifold_clustering import GlomeruliClusterer
from src.clustering.cluster_visualizer import ClusterVisualizer, CLINICAL_NAMES
from src.models.mask_to_bbox import extract_bboxes_from_mask
from src.models.unet_dataset import extract_glomerulus_crop


def setup_logger() -> logging.Logger:
    """Configure structured console logging."""
    logger = logging.getLogger("clustering")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_and_extract_glomeruli_crops(
    manifest_path: str,
    splits: List[str],
    crop_size: int = 512,
    margin_ratio: float = 0.25,
    min_glom_area: int = 400,
    max_samples: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
    """
    Load dataset patches and extract standardized, context-preserving crops for each glomerulus.
    """
    manifest_p = Path(manifest_path)
    dataset_root = manifest_p.parent
    df = pd.read_csv(manifest_p)

    # Filter requested splits and positive patches only
    df_filtered = df[(df["split"].isin(splits)) & (df["is_positive"] == True)].reset_index(drop=True)

    crops_rgb: List[np.ndarray] = []
    masks_binary: List[np.ndarray] = []
    metadata_list: List[Dict[str, Any]] = []

    for _, row in df_filtered.iterrows():
        img_path = dataset_root / row["image_path"]
        mask_path = dataset_root / row["mask_path"]

        img_bgr = cv2.imread(str(img_path))
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or mask_gray is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask_bin = (mask_gray > 127).astype(np.uint8) * 255

        # Extract individual glomeruli bounding boxes from the mask
        detections = extract_bboxes_from_mask(mask_bin, mask_bin.astype(float) / 255.0, min_area=min_glom_area)

        for det in detections:
            bbox = det["bbox"]
            crop_img, crop_coords = extract_glomerulus_crop(
                img_rgb, bbox, margin_ratio=margin_ratio, target_size=crop_size
            )

            # Extract corresponding cropped mask
            c_x1, c_y1, c_x2, c_y2 = crop_coords
            mask_patch = mask_bin[c_y1:c_y2, c_x1:c_x2]
            if mask_patch.shape[0] == 0 or mask_patch.shape[1] == 0:
                crop_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
            else:
                crop_mask = cv2.resize(mask_patch, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

            crops_rgb.append(crop_img)
            masks_binary.append(crop_mask)
            metadata_list.append({
                "patch_id": row["patch_id"],
                "slide_id": row["slide_id"],
                "split": row["split"],
                "bbox": bbox,
                "glom_area_native": det["area"],
            })

            if max_samples is not None and len(crops_rgb) >= max_samples:
                break
        if max_samples is not None and len(crops_rgb) >= max_samples:
            break

    return crops_rgb, masks_binary, metadata_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsupervised Glomeruli Clustering & Manifold Learning.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/clustering_config.yaml",
        help="Path to clustering configuration YAML.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Override number of target clusters (e.g. 3).",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default=None,
        choices=["hybrid", "deep", "morphological"],
        help="Feature modality: 'hybrid', 'deep', 'morphological'.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["tsne", "umap", "pca"],
        help="2D Manifold projection method.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run deep feature extractor on (e.g. 0, 'cpu').",
    )
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("Initializing Unsupervised Glomerular Clustering Pipeline...")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # CLI Overrides
    if args.n_clusters is not None:
        cfg["clustering"]["n_clusters"] = args.n_clusters
    if args.feature_type is not None:
        cfg["features"]["feature_type"] = args.feature_type
    if args.method is not None:
        cfg["manifold"]["method"] = args.method
    if args.device is not None:
        cfg["features"]["device"] = args.device

    out_dir = Path(cfg["output"].get("project_dir", "runs/clustering")) / cfg["output"].get(
        "experiment_name", "unsupervised_grading"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # 1. Extract Glomeruli Crops
    logger.info("Step 1: Extracting focused glomeruli crops from dataset...")
    crops_rgb, masks_binary, metadata = load_and_extract_glomeruli_crops(
        manifest_path=cfg["data"]["manifest_csv"],
        splits=cfg["data"].get("splits", ["train", "val", "test"]),
        crop_size=int(cfg["data"].get("crop_size", 512)),
        margin_ratio=float(cfg["data"].get("margin_ratio", 0.25)),
        min_glom_area=int(cfg["data"].get("min_glom_area", 400)),
        max_samples=cfg["data"].get("max_samples", None),
    )
    N = len(crops_rgb)
    logger.info(f"Successfully extracted {N} glomeruli crops with segmentation masks.")

    if N < 5:
        logger.error("Insufficient glomeruli extracted for clustering. Aborting.")
        sys.exit(1)

    # 2. Extract Features
    logger.info("Step 2: Extracting morphological and deep features...")
    extractor = GlomeruliFeatureExtractor(
        backbone_name=cfg["features"].get("deep_backbone", "resnet34"),
        pretrained=bool(cfg["features"].get("pretrained", True)),
        device=cfg["features"].get("device", 0),
        batch_size=int(cfg["features"].get("batch_size", 32)),
        normalize_embeddings=bool(cfg["features"].get("normalize_embeddings", True)),
    )

    feat_type = cfg["features"].get("feature_type", "hybrid").lower()
    morph_feats, morph_names = extractor.extract_morphological_dataset(crops_rgb, masks_binary)

    features_matrix: np.ndarray
    feature_names: List[str]

    if feat_type == "deep":
        deep_embeddings = extractor.extract_deep_embeddings(crops_rgb)
        features_matrix = deep_embeddings
        feature_names = [f"deep_emb_{i}" for i in range(deep_embeddings.shape[1])]
    elif feat_type == "morphological":
        features_matrix = morph_feats
        feature_names = morph_names
    else:  # hybrid
        deep_embeddings = extractor.extract_deep_embeddings(crops_rgb)
        from sklearn.preprocessing import StandardScaler
        morph_scaled = StandardScaler().fit_transform(morph_feats)
        features_matrix = np.hstack([deep_embeddings, morph_scaled]).astype(np.float32)
        feature_names = [f"deep_emb_{i}" for i in range(deep_embeddings.shape[1])] + morph_names

    logger.info(f"Constructed {feat_type.upper()} feature matrix: {features_matrix.shape}")

    # 3. Manifold Learning & Clustering
    logger.info("Step 3: Performing Manifold Learning & Unsupervised Subtyping...")
    n_clusters = int(cfg["clustering"].get("n_clusters", 3))
    clusterer = GlomeruliClusterer(
        n_clusters=n_clusters,
        pca_variance=float(cfg["manifold"].get("pca_variance", 0.95)),
        manifold_method=cfg["manifold"].get("method", "tsne"),
        random_state=int(cfg["clustering"].get("seed", 42)),
    )

    # Optimal K Profiling
    k_range = cfg["clustering"].get("k_search_range", [2, 3, 4, 5, 6])
    logger.info(f"Evaluating cluster stability across K={k_range}...")
    k_results = clusterer.evaluate_optimal_k(features_matrix, k_range=k_range)
    logger.info(f"Optimal cluster count suggested by Silhouette: K={k_results['optimal_k_by_silhouette']}")

    # Main Clustering Fit
    clustering_results = clusterer.fit_transform(
        features=features_matrix,
        algorithm=cfg["clustering"].get("algorithm", "kmeans"),
        order_by_progression=bool(cfg["clustering"].get("order_clusters_by_progression", True)),
    )
    labels = clustering_results["labels"]

    # 4. Generate Visualizations & Artifacts
    logger.info(f"Step 4: Generating analytical visualizations in {out_dir}...")
    visualizer = ClusterVisualizer(output_dir=out_dir)

    p_scatter = visualizer.plot_manifold_scatter(
        embedding_2d=clustering_results["embedding_2d"],
        labels=labels,
        method_name=cfg["manifold"].get("method", "t-SNE"),
        silhouette=clustering_results["silhouette_score"],
    )

    p_k = visualizer.plot_optimal_k_curves(k_results=k_results)

    p_box = visualizer.plot_morphological_distributions(
        features_matrix=morph_feats,
        feature_names=morph_names,
        labels=labels,
    )

    if cfg["output"].get("save_gallery", True):
        visualizer.plot_cluster_gallery(
            crops_rgb=crops_rgb,
            masks_binary=masks_binary,
            labels=labels,
            pca_features=clustering_results["pca_features"],
            samples_per_cluster=int(cfg["output"].get("gallery_samples_per_cluster", 6)),
        )

    # 5. Export Datasets & Final Summary
    assignments_df = pd.DataFrame(metadata)
    assignments_df["cluster_label"] = labels
    assignments_df["cluster_name"] = [
        CLINICAL_NAMES[l] if l < len(CLINICAL_NAMES) else f"Class {l}" for l in labels
    ]
    for col_i, col_name in enumerate(morph_names):
        assignments_df[col_name] = morph_feats[:, col_i]

    csv_path = out_dir / "cluster_assignments.csv"
    assignments_df.to_csv(csv_path, index=False)

    report = {
        "n_samples": int(N),
        "n_clusters": int(n_clusters),
        "feature_type": feat_type,
        "silhouette_score": clustering_results["silhouette_score"],
        "davies_bouldin_score": clustering_results["davies_bouldin_score"],
        "calinski_harabasz_score": clustering_results["calinski_harabasz_score"],
        "class_distribution": clustering_results["distribution"],
        "pca_components": clustering_results["n_components_pca"],
        "total_runtime_sec": round(time.time() - start_time, 2),
    }

    report_path = out_dir / "clustering_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Artifacts saved successfully to: {out_dir}")

    # Print Final Summary Table
    print("\n" + "=" * 68)
    print("UNSUPERVISED GLOMERULI CLUSTERING & MANIFOLD LEARNING REPORT")
    print("=" * 68)
    print(f"Total Glomeruli Analyzed : {N}")
    print(f"Feature Space Modality   : {feat_type.upper()} ({features_matrix.shape[1]} dims)")
    print(f"PCA Reduced Dimensions   : {clustering_results['n_components_pca']} (Var: {clustering_results['variance_explained']:.2%})")
    print(f"Manifold 2D Method       : {cfg['manifold'].get('method', 'tsne').upper()}")
    print("-" * 68)
    print("CLUSTERING QUALITY METRICS:")
    print(f"   Silhouette Score      : {clustering_results['silhouette_score']:.4f}")
    print(f"   Davies-Bouldin Index  : {clustering_results['davies_bouldin_score']:.4f}")
    print(f"   Calinski-Harabasz     : {clustering_results['calinski_harabasz_score']:.2f}")
    print("-" * 68)
    print("DISCOVERED SUBTYPES BREAKDOWN (Progression Ordered):")
    for k_name, count in clustering_results["distribution"].items():
        pct = (count / N) * 100.0
        class_idx = int(k_name.split()[-1])
        c_desc = CLINICAL_NAMES[class_idx] if class_idx < len(CLINICAL_NAMES) else k_name
        print(f"   {c_desc:<38} : {count:>4} ({pct:5.1f}%)")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
