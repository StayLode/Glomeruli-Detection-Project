#!/usr/bin/env python3
"""Evaluate one clustering run using shared annotation-level resampling."""

import argparse
import itertools
import json
from pathlib import Path
from typing import Optional, Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from src.clustering.manifold_clustering import GlomeruliClusterer
from src.clustering.modeling import FittedFeatureModel, fit_feature_model
from src.clustering.random_baseline import random_geometry_baseline
from src.clustering.metrics import _matched_cluster_jaccards
from src.clustering.resampling import create_or_load_resampling
from src.utils.evaluation import (
    CLUSTERING_EVALUATION_SCHEMA_VERSION,
    EvaluationRun,
    file_identity,
    write_json,
)


def _fit_eval_model(
    data: Dict[str, np.ndarray], config: Dict[str, Any], indices: np.ndarray, seed: int
) -> FittedFeatureModel:
    clustering, features_cfg, manifold = (
        config["clustering"],
        config["features"],
        config["manifold"],
    )
    feature_type = str(features_cfg["feature_type"])
    default_method = "none" if feature_type == "stain_invariant_morph" else "pca"
    manifold_method = str(manifold.get("method") or default_method)
    default_scaling = "none" if features_cfg.get("block_weighting") == "equal" else "standard"
    return fit_feature_model(
        int(clustering["n_clusters"]),
        str(clustering.get("algorithm", "kmeans")),
        seed,
        manifold.get("pca_variance", 20 if manifold_method == "pca" else None),
        feature_type,
        str(features_cfg.get("block_weighting", "none")),
        str(clustering.get("input_scaling", default_scaling)),
        data["model_features"][indices] if feature_type != "hybrid" else None,
        data["deep_features_raw"][indices] if feature_type == "hybrid" else None,
        data["morphological_features_raw"][indices] if feature_type == "hybrid" else None,
        manifold_method=manifold_method,
        selected_features=features_cfg.get("selected_features"),
    )


def _predict_eval_model(
    model: FittedFeatureModel,
    data: Dict[str, np.ndarray],
    indices: np.ndarray,
    feature_type: str,
) -> np.ndarray:
    if feature_type == "hybrid":
        return model.predict(
            deep_features=data["deep_features_raw"][indices],
            morph_features=data["morphological_features_raw"][indices],
        )
    return model.predict(features=data["model_features"][indices])


def _internal_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    if len(features) == 0:
        return {
            "status": "not_evaluated",
            "reason": "no samples were assigned in this split",
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return {
            "status": "not_evaluable",
            "reason": "requires at least two non-degenerate clusters",
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }
    return {
        "status": "evaluated",
        "reason": None,
        "silhouette": float(silhouette_score(features, labels)),
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
    }


def _cramers_v(clusters: np.ndarray, slides: Sequence[str]) -> Dict[str, Any]:
    slide_series = pd.Series(list(slides), dtype=str)
    if len(clusters) == 0 or slide_series.nunique() < 2 or len(np.unique(clusters)) < 2:
        return {
            "status": "not_evaluable",
            "reason": "requires at least two slides and two clusters",
            "cramers_v_cluster_slide": None,
            "nmi_cluster_slide": None,
            "max_dominant_slide_fraction": None,
        }
    table = pd.crosstab(pd.Series(clusters, name="cluster"), slide_series)
    chi2 = float(chi2_contingency(table)[0])
    n = int(table.to_numpy().sum())
    denom = min(table.shape[0] - 1, table.shape[1] - 1)
    v = float(np.sqrt(chi2 / (n * denom))) if n > 0 and denom > 0 else None
    dominant = float((table.div(table.sum(axis=1), axis=0)).max(axis=1).max())
    return {
        "status": "evaluated",
        "reason": None,
        "cramers_v_cluster_slide": v,
        "nmi_cluster_slide": float(normalized_mutual_info_score(slide_series, clusters)),
        "max_dominant_slide_fraction": dominant,
    }


def _eta_squared(values: pd.Series, labels: np.ndarray) -> Optional[float]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna().to_numpy()
    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return None
    y = numeric.to_numpy(dtype=float)[valid]
    g = labels[valid]
    grand = float(np.mean(y))
    ss_total = float(np.sum((y - grand) ** 2))
    if ss_total <= 1e-12:
        return 0.0
    ss_between = sum(
        float(np.sum(g == c)) * float((np.mean(y[g == c]) - grand) ** 2)
        for c in np.unique(g)
    )
    return float(ss_between / ss_total)


def _mean_std(records: Sequence[Dict[str, Any]], key: str) -> Tuple[Optional[float], Optional[float]]:
    values = [float(r[key]) for r in records if r.get(key) is not None and np.isfinite(r[key])]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=0))


def evaluate_run(run_dir: Path, resampling_path: Path, seed: int) -> Tuple[Dict[str, Any], EvaluationRun]:
    from src.clustering.splits import TRAINING_PROTOCOL, TRAINING_SPLITS, TEST_SLIDES

    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run_json.get("status") != "completed":
        raise ValueError(f"Run is not completed: {run_dir}")
    if not run_json.get("cohort_hash"):
        raise ValueError("A frozen annotation-level cohort is required")
    if run_json.get("training_protocol") != TRAINING_PROTOCOL:
        raise ValueError("Evaluation requires the train/validation-only protocol; rebuild the run")
    if (
        run_json.get("cohort_schema_version") != 2
        or run_json.get("preprocessing_schema_version") != 2
        or run_json.get("feature_schema_version") != 2
        or run_json.get("cohort_provenance_mode") != "schema_v2"
    ):
        raise ValueError(
            "This run predates the corrected cohort/preprocessing/feature schemas; "
            "preserve it as diagnostic evidence and create a new clustering run."
        )

    assignments = pd.read_csv(run_dir / "assignments.csv")
    if not set(assignments["split"]).issubset(TRAINING_SPLITS) or set(assignments["slide_id"]) & TEST_SLIDES:
        raise ValueError("Training evaluation cannot contain test samples")

    with np.load(run_dir / "features.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if not set(data["common_split"].astype(str)).issubset(TRAINING_SPLITS) or set(data["common_slide_id"].astype(str)) & TEST_SLIDES:
        raise ValueError("Training evaluation common space cannot contain test samples")
    if "common_morph_features" not in data:
        raise ValueError("Run is missing the shared native-morphology evaluation space")

    ids = data["annotation_id"].astype(str)
    lookup = {value: index for index, value in enumerate(ids)}
    if len(lookup) != len(ids):
        raise ValueError("Run contains duplicate annotation_id values")
    assignment_ids = assignments["annotation_id"].astype(str).tolist()
    if len(set(assignment_ids)) != len(assignment_ids) or set(assignment_ids) != set(ids):
        raise ValueError("Feature arrays and assignments have different annotation_id sets")
    feature_order = np.asarray([lookup[annotation_id] for annotation_id in assignment_ids])
    for key in (
        "deep_embeddings",
        "deep_features_raw",
        "morphological_features",
        "morphological_features_raw",
        "model_features",
        "annotation_id",
        "slide_id",
        "split",
    ):
        if key in data and len(data[key]) == len(ids):
            data[key] = data[key][feature_order]

    feature_type = str(run_json["config"]["features"]["feature_type"])
    if feature_type == "hybrid" and not {"deep_features_raw", "morphological_features_raw"}.issubset(data):
        raise ValueError("Hybrid evaluation requires cached raw feature blocks from schema v2")

    train_splits = list(run_json["config"]["clustering"].get("fit_splits", ["train"]))
    fit_mask = assignments["split"].astype(str).isin(train_splits).to_numpy()
    train_rows = np.asarray(
        sorted(np.flatnonzero(fit_mask), key=lambda index: str(assignments.iloc[index]["annotation_id"])),
        dtype=int,
    )
    train_ids = assignments.iloc[train_rows]["annotation_id"].astype(str).tolist()
    cohort_hash = str(run_json["cohort_hash"])
    resampling, resampling_hash = create_or_load_resampling(
        resampling_path,
        train_ids,
        assignments.iloc[train_rows]["slide_id"].astype(str),
        cohort_hash,
        seed,
    )

    config = run_json["config"]
    labels = assignments["cluster"].to_numpy(dtype=int)
    n_clusters = int(config["clustering"]["n_clusters"])
    algorithm = str(config["clustering"].get("algorithm", "kmeans"))
    base_seed = int(config["clustering"].get("seed", 42))

    eval_run = EvaluationRun(
        "clustering_eval",
        "train_val",
        config,
        str(run_dir),
        config.get("output", {}).get("evaluation_name", "evaluation"),
        [run_dir, resampling_path],
    )
    try:
        full_models: List[FittedFeatureModel] = []
        fit_records = []
        seed_labels = []
        for offset in range(10):
            model = _fit_eval_model(data, config, train_rows, base_seed + offset)
            full_models.append(model)
            fit_records.append({"replicate_id": f"seed-{offset:02d}", **model.clusterer.fit_diagnostics_})
            seed_labels.append(_predict_eval_model(model, data, train_rows, feature_type))

        seed_ari = [
            {"pair": f"{i}-{j}", "value": float(adjusted_rand_score(seed_labels[i], seed_labels[j]))}
            for i, j in itertools.combinations(range(len(seed_labels)), 2)
        ]

        id_to_row = {str(assignments.iloc[row]["annotation_id"]): int(row) for row in train_rows}
        sub_models: List[FittedFeatureModel] = []
        sub_indices: List[np.ndarray] = []
        for rep in resampling["subsamples"]:
            idx = np.asarray([id_to_row[str(aid)] for aid in rep["annotation_ids"]], dtype=int)
            sub_indices.append(idx)
            sub_models.append(_fit_eval_model(data, config, idx, int(rep["seed"])))

        pair_records = []
        for i, j in itertools.combinations(range(len(sub_models)), 2):
            shared = np.intersect1d(sub_indices[i], sub_indices[j])
            if len(shared) < 2:
                continue
            pred_i = _predict_eval_model(sub_models[i], data, shared, feature_type)
            pred_j = _predict_eval_model(sub_models[j], data, shared, feature_type)
            jaccards = _matched_cluster_jaccards(pred_i, pred_j, n_clusters)
            pair_records.append({
                "pair": f"{i}-{j}",
                "n_shared": int(len(shared)),
                "ari": float(adjusted_rand_score(pred_i, pred_j)),
                "cluster_jaccard_mean": float(np.mean(jaccards)),
                "cluster_jaccard_min": float(np.min(jaccards)),
            })

        loso_records = []
        train_slides = sorted(assignments.iloc[train_rows]["slide_id"].astype(str).unique())
        for fold_idx, held_slide in enumerate(train_slides):
            keep = np.asarray(
                [r for r in train_rows if str(assignments.iloc[r]["slide_id"]) != held_slide],
                dtype=int,
            )
            if len(keep) < n_clusters:
                continue
            loso_model = _fit_eval_model(data, config, keep, base_seed + fold_idx)
            pred_keep = _predict_eval_model(loso_model, data, keep, feature_type)
            ref_keep = seed_labels[0][np.isin(train_rows, keep)]
            loso_records.append({
                "held_out_slide": held_slide,
                "n_train": int(len(keep)),
                "value": float(adjusted_rand_score(ref_keep, pred_keep)),
            })

        common_lookup = {str(aid): idx for idx, aid in enumerate(data["common_annotation_id"].astype(str))}
        common_order = np.asarray([common_lookup[aid] for aid in assignment_ids], dtype=int)
        common_features = data["common_morph_features"][common_order]

        common_split_metrics: Dict[str, Any] = {}
        paired_records: Dict[str, List[Dict[str, Any]]] = {
            "train_silhouette": [],
            "train_davies_bouldin": [],
            "train_calinski_harabasz": [],
            "train_cramers_v_slide": [],
            "train_nmi_slide": [],
            "train_dominant_slide_fraction": [],
            "completeness_eta2": [],
            "border_distance_px_eta2": [],
            "tissue_ratio_eta2": [],
            "background_mean_intensity_eta2": [],
        }

        for split_name in TRAINING_SPLITS:
            mask = assignments["split"].astype(str).to_numpy() == split_name
            common_split_metrics[split_name] = _internal_metrics(common_features[mask], labels[mask])

        for rep_idx, (idx, model) in enumerate(zip(sub_indices, sub_models)):
            pred = _predict_eval_model(model, data, idx, feature_type)
            m = _internal_metrics(common_features[idx], pred)
            conf = _cramers_v(pred, assignments.iloc[idx]["slide_id"].astype(str))
            sub_df = assignments.iloc[idx]
            paired_records["train_silhouette"].append({"replicate_id": f"sub-{rep_idx:02d}", "value": m["silhouette"]})
            paired_records["train_davies_bouldin"].append({"replicate_id": f"sub-{rep_idx:02d}", "value": m["davies_bouldin"]})
            paired_records["train_calinski_harabasz"].append({"replicate_id": f"sub-{rep_idx:02d}", "value": m["calinski_harabasz"]})
            paired_records["train_cramers_v_slide"].append({"replicate_id": f"sub-{rep_idx:02d}", "value": conf["cramers_v_cluster_slide"]})
            paired_records["train_nmi_slide"].append({"replicate_id": f"sub-{rep_idx:02d}", "value": conf["nmi_cluster_slide"]})
            paired_records["train_dominant_slide_fraction"].append({"replicate_id": f"sub-{rep_idx:02d}", "value": conf["max_dominant_slide_fraction"]})
            for col, rec_key in (
                ("completeness", "completeness_eta2"),
                ("border_distance_px", "border_distance_px_eta2"),
                ("tissue_ratio", "tissue_ratio_eta2"),
                ("background_mean_intensity", "background_mean_intensity_eta2"),
            ):
                val = _eta_squared(sub_df[col], pred) if col in sub_df else None
                paired_records[rec_key].append({"replicate_id": f"sub-{rep_idx:02d}", "value": val})

        own_train_input = (
            full_models[0].block_transformer.transform(
                data["deep_features_raw"][train_rows],
                data["morphological_features_raw"][train_rows],
            )
            if feature_type == "hybrid"
            else full_models[0].block_transformer.transform(data["model_features"][train_rows])
        )
        own_train_space = full_models[0].clusterer.transform(own_train_input)
        own_space_train = _internal_metrics(own_train_space, labels[train_rows])

        baseline_summary = random_geometry_baseline(
            common_features[train_rows],
            labels[train_rows],
            n_clusters=n_clusters,
            algorithm=algorithm,
            seed=base_seed,
            n_replicates=25,
            seed_ari_observed=_mean_std(seed_ari, "value")[0],
        )

        train_df = assignments.iloc[train_rows]
        tech_eta2 = {
            f"{col}_eta2": _eta_squared(train_df[col], labels[train_rows]) if col in train_df else None
            for col in ("completeness", "border_distance_px", "tissue_ratio", "background_mean_intensity")
        }

        seed_ari_mean, seed_ari_std = _mean_std(seed_ari, "value")
        sub_ari_mean, sub_ari_std = _mean_std(pair_records, "ari")
        jaccard_mean, _ = _mean_std(pair_records, "cluster_jaccard_mean")
        jaccard_min_vals = [r["cluster_jaccard_min"] for r in pair_records if r.get("cluster_jaccard_min") is not None]
        jaccard_min = float(np.min(jaccard_min_vals)) if jaccard_min_vals else None
        loso_ari_mean, loso_ari_std = _mean_std(loso_records, "value")

        metrics: Dict[str, Any] = {
            "evaluation_schema_version": CLUSTERING_EVALUATION_SCHEMA_VERSION,
            "run_dir": str(run_dir),
            "cohort_hash": cohort_hash,
            "resampling_hash": resampling_hash,
            "feature_type": feature_type,
            "n_clusters": n_clusters,
            "stability": {
                "seed_ari_mean": seed_ari_mean,
                "seed_ari_std": seed_ari_std,
                "seed_ari_replicates": seed_ari,
                "subsampling_ari_mean": sub_ari_mean,
                "subsampling_ari_std": sub_ari_std,
                "cluster_jaccard_mean": jaccard_mean,
                "cluster_jaccard_min": jaccard_min,
                "subsampling_pair_replicates": pair_records,
                "leave_one_slide_out_ari_mean": loso_ari_mean,
                "leave_one_slide_out_ari_std": loso_ari_std,
                "leave_one_slide_out_replicates": loso_records,
                "fit_diagnostics": fit_records,
            },
            "common_space_internal_metrics": common_split_metrics,
            "own_space_train_metrics": own_space_train,
            "slide_confounding": {
                "train": _cramers_v(labels[train_rows], train_df["slide_id"].astype(str)),
            },
            "technical_confounding_eta2": tech_eta2,
            "random_baseline": baseline_summary,
            "paired_metric_records": paired_records,
        }
        write_json(eval_run.run_dir / "resampling.json", resampling)
        eval_run.metadata.update(
            cohort_hash=cohort_hash,
            resampling_hash=resampling_hash,
            common_evaluation_space_hash=run_json.get("common_evaluation_space_hash"),
        )
        eval_run.complete(metrics, [eval_run.run_dir / "resampling.json"])
        return metrics, eval_run
    except Exception as error:
        eval_run.fail(error)
        raise
