#!/usr/bin/env python3
"""Evaluate one clustering run using shared annotation-level resampling."""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
        config["clustering"], config["features"], config["manifold"]
    )
    feature_type = features_cfg["feature_type"]
    return fit_feature_model(
        int(clustering["n_clusters"]),
        str(clustering.get("algorithm", "kmeans")),
        seed,
        manifold.get("pca_variance", 0.95),
        feature_type,
        str(features_cfg.get("block_weighting", "none")),
        str(clustering.get(
            "input_scaling",
            "none" if features_cfg.get("block_weighting") == "equal" else "standard",
        )),
        data["model_features"][indices] if feature_type != "hybrid" else None,
        data["deep_features_raw"][indices] if feature_type == "hybrid" else None,
        data["morphological_features_raw"][indices] if feature_type == "hybrid" else None,

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
            "silhouette": None, "davies_bouldin": None, "calinski_harabasz": None,
        }
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return {
            "status": "not_evaluable",
            "reason": "requires at least two non-degenerate clusters",
            "silhouette": None, "davies_bouldin": None, "calinski_harabasz": None,
        }
    return {
        "status": "evaluated",
        "reason": None,
        "silhouette": float(silhouette_score(features, labels)),
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
    }


def _cramers_v(labels: Sequence[Any], groups: Sequence[Any]) -> float:
    table = pd.crosstab(pd.Series(labels), pd.Series(groups)).to_numpy()
    if min(table.shape) < 2:
        return 0.0
    chi2 = float(chi2_contingency(table, correction=False)[0])
    return float(np.sqrt(chi2 / (table.sum() * min(table.shape[0] - 1, table.shape[1] - 1))))


def _max_dominant_group_fraction(labels: Sequence[Any], groups: Sequence[Any]) -> float:
    table = pd.crosstab(pd.Series(labels), pd.Series(groups)).to_numpy()
    if table.size == 0:
        return float("nan")
    return float(np.max(table.max(axis=1) / np.maximum(table.sum(axis=1), 1)))


def _eta_squared(values: Sequence[float], labels: Sequence[Any]) -> float:
    value_array = np.asarray(values, dtype=float)
    label_array = np.asarray(labels)
    valid = np.isfinite(value_array)
    value_array, label_array = value_array[valid], label_array[valid]
    if len(value_array) < 2 or np.var(value_array) == 0:
        return 0.0
    mean = value_array.mean()
    between = sum(
        np.count_nonzero(label_array == label) * (value_array[label_array == label].mean() - mean) ** 2
        for label in np.unique(label_array)
    )
    return float(between / np.sum((value_array - mean) ** 2))


def _finite_eta(values: Sequence[float], labels: Sequence[Any]) -> float:
    return _eta_squared(values, labels)


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
        "deep_embeddings", "deep_features_raw", "morphological_features",
        "morphological_features_raw", "model_features",
        "annotation_id", "slide_id", "split",
    ):
        if key in data and len(data[key]) == len(ids):
            data[key] = data[key][feature_order]
    ids = data["annotation_id"].astype(str)
    feature_type = str(run_json["config"]["features"]["feature_type"])
    if feature_type == "hybrid" and not {
        "deep_features_raw", "morphological_features_raw"
    }.issubset(data):
        raise ValueError("Hybrid evaluation requires cached raw feature blocks from schema v2")
    train_splits = list(run_json["config"]["clustering"].get("fit_splits", ["train"]))
    fit_mask = assignments["split"].astype(str).isin(train_splits).to_numpy()
    train_rows = np.asarray(sorted(
        np.flatnonzero(fit_mask),
        key=lambda index: str(assignments.iloc[index]["annotation_id"]),
    ), dtype=int)
    train_ids = assignments.iloc[train_rows]["annotation_id"].astype(str).tolist()
    cohort_hash = str(run_json["cohort_hash"])
    resampling, resampling_hash = create_or_load_resampling(
        resampling_path, train_ids, assignments.iloc[train_rows]["slide_id"].astype(str),
        cohort_hash, seed,
    )
    config = run_json["config"]
    model_features = data["model_features"]
    labels = assignments["cluster"].to_numpy(dtype=int)
    n_clusters = int(config["clustering"]["n_clusters"])
    algorithm = str(config["clustering"].get("algorithm", "kmeans"))
    base_seed = int(config["clustering"].get("seed", 42))
    full_models: List[FittedFeatureModel] = []
    fit_records = []
    seed_labels = []
    for offset in range(10):
        model = _fit_eval_model(data, config, train_rows, base_seed + offset)
        full_models.append(model)
        fit_records.append({"replicate_id": f"seed-{offset:02d}", **model.clusterer.fit_diagnostics_})
        seed_labels.append(_predict_eval_model(model, data, train_rows, feature_type))
    seed_ari = [
        float(adjusted_rand_score(seed_labels[i], seed_labels[j]))
        for i, j in itertools.combinations(range(len(seed_labels)), 2)
    ]

    subsampling_models: List[FittedFeatureModel] = []
    subsampling_sample_indices: List[np.ndarray] = []
    subsampling_records = resampling["subsampling_replicates"]
    id_to_local = {value: i for i, value in enumerate(train_ids)}
    for offset, record in enumerate(subsampling_records):
        sample_ids = record["annotation_ids"]
        indices = np.asarray([id_to_local[value] for value in sample_ids], dtype=int)
        subsampling_sample_indices.append(indices)
        model = _fit_eval_model(data, config, train_rows[indices], base_seed + 100 + offset)
        subsampling_models.append(model)
        fit_records.append({"replicate_id": record["replicate_id"], **model.clusterer.fit_diagnostics_})
    subsampling_ari: List[float] = []
    subsampling_jaccard: List[float] = []
    subsampling_jaccard_mean_by_pair: List[float] = []
    subsampling_jaccard_min_by_pair: List[float] = []
    subsampling_pair_ids: List[str] = []
    for i, j in itertools.combinations(range(len(subsampling_models)), 2):
        ids_left = subsampling_records[i]["annotation_ids"]
        ids_right = subsampling_records[j]["annotation_ids"]
        shared = sorted(set(ids_left) & set(ids_right))
        common_rows = np.asarray([id_to_local[value] for value in shared], dtype=int)
        common_data_rows = train_rows[common_rows]
        left = _predict_eval_model(
            subsampling_models[i], data, common_data_rows, feature_type
        )
        right = _predict_eval_model(
            subsampling_models[j], data, common_data_rows, feature_type
        )
        subsampling_ari.append(float(adjusted_rand_score(left, right)))
        matched_jaccards = _matched_cluster_jaccards(
            left, right, n_clusters=n_clusters
        )
        subsampling_jaccard.extend(matched_jaccards)
        subsampling_jaccard_mean_by_pair.append(float(np.mean(matched_jaccards)))
        subsampling_jaccard_min_by_pair.append(float(np.min(matched_jaccards)))
        subsampling_pair_ids.append(
            f"{subsampling_records[i]['replicate_id']}|{subsampling_records[j]['replicate_id']}"
        )

    base_model = full_models[0]
    full_train_labels = seed_labels[0]
    loo_results = {}
    train_id_indices = {annotation_id: i for i, annotation_id in enumerate(train_ids)}
    for fold in resampling["leave_one_slide_out"]:
        holdout_local = np.asarray(
            [train_id_indices[value] for value in fold["holdout_annotation_ids"]],
            dtype=int,
        )
        fit_local = np.asarray(
            [train_id_indices[value] for value in fold["fit_annotation_ids"]],
            dtype=int,
        )
        if len(holdout_local) == 0 or len(fit_local) <= n_clusters:
            continue
        loo = _fit_eval_model(data, config, train_rows[fit_local], base_seed)
        fit_records.append({"replicate_id": fold["fold_id"], **loo.clusterer.fit_diagnostics_})
        predicted = _predict_eval_model(
            loo, data, train_rows[holdout_local], feature_type
        )
        loo_results[fold["held_out_slide_id"]] = float(
            adjusted_rand_score(full_train_labels[holdout_local], predicted)
        )

    common_ids = data["common_annotation_id"].astype(str)
    common_lookup = {value: i for i, value in enumerate(common_ids)}
    assignment_lookup = {
        str(row.annotation_id): i for i, row in assignments.reset_index(drop=True).iterrows()
    }
    expected_splits = set(
        config["clustering"].get("fit_splits", ["train"])
        + config["clustering"].get("assign_splits", [])
    )
    expected_ids = {
        str(annotation_id) for annotation_id, split in zip(
            data["common_annotation_id"], data["common_split"]
        ) if str(split) in expected_splits
    }
    if set(assignment_lookup) != expected_ids:
        raise ValueError(
            "Clustering assignments do not contain exactly the annotation IDs for configured splits"
        )
    common_indices, assignment_indices = [], []
    for i, annotation_id in enumerate(common_ids):
        if annotation_id in assignment_lookup:
            common_indices.append(i)
            assignment_indices.append(assignment_lookup[annotation_id])
    common_features = data["common_morph_features"][common_indices]
    common_labels = labels[assignment_indices]
    common_splits = data["common_split"][common_indices].astype(str)
    common_slides = data["common_slide_id"][common_indices].astype(str)
    common_ids = common_ids[common_indices]
    common_index_by_id = {value: index for index, value in enumerate(common_ids)}
    paired_samples: Dict[str, List[float]] = {
        name: [] for name in (
            "train_silhouette", "train_davies_bouldin", "train_calinski_harabasz",
            "train_cramers_v_slide", "train_nmi_slide",
            "train_dominant_slide_fraction",
            "completeness_eta2", "border_distance_px_eta2", "tissue_ratio_eta2",
            "background_mean_intensity_eta2",
        )
    }
    paired_sample_records: Dict[str, List[Dict[str, Any]]] = {
        key: [] for key in paired_samples
    }

    def record_paired(name: str, replicate_id: str, value: Any) -> None:
        paired_samples[name].append(float(value))
        paired_sample_records[name].append({
            "replicate_id": replicate_id,
            "value": float(value),
        })

    for model, local_indices, record in zip(
        subsampling_models, subsampling_sample_indices, subsampling_records
    ):
        sample_ids = record["annotation_ids"]
        model_row_indices = train_rows[local_indices]
        predicted = _predict_eval_model(model, data, model_row_indices, feature_type)
        selected_common = [
            (sample_pos, common_index_by_id[annotation_id])
            for sample_pos, annotation_id in enumerate(sample_ids)
            if annotation_id in common_index_by_id
        ]
        if len(selected_common) < 3:
            continue
        model_positions, common_positions = map(np.asarray, zip(*selected_common))
        common_sample_features = common_features[common_positions]
        sample_labels = predicted[model_positions]
        sample_slides = common_slides[common_positions]
        internal = _internal_metrics(common_sample_features, sample_labels)
        replicate_id = str(record["replicate_id"])
        for key in ("silhouette", "davies_bouldin", "calinski_harabasz"):
            if internal[key] is not None:
                record_paired(f"train_{key}", replicate_id, internal[key])
        record_paired(
            "train_cramers_v_slide", replicate_id, _cramers_v(sample_labels, sample_slides)
        )
        record_paired(
            "train_dominant_slide_fraction", replicate_id,
            _max_dominant_group_fraction(sample_labels, sample_slides),
        )
        record_paired(
            "train_nmi_slide", replicate_id,
            normalized_mutual_info_score(sample_labels, sample_slides),
        )
        sample_metadata = assignments.iloc[model_row_indices[model_positions]]
        for column, output_name in (
            ("completeness", "completeness_eta2"),
            ("border_distance_px", "border_distance_px_eta2"),
            ("tissue_ratio", "tissue_ratio_eta2"),
            ("background_mean_intensity", "background_mean_intensity_eta2"),
        ):
            if column in sample_metadata:
                record_paired(
                    output_name, replicate_id,
                    _finite_eta(sample_metadata[column], sample_labels),
                )
    common_metrics = {}
    for split in TRAINING_SPLITS:
        selected = common_splits == split
        common_metrics[split] = _internal_metrics(common_features[selected], common_labels[selected])

    train_common_positions = np.flatnonzero(common_splits == "train")
    train_common = common_features[train_common_positions]
    train_common_ids = common_ids[train_common_positions]
    canonical_common_order = np.argsort(train_common_ids, kind="stable")
    train_common = train_common[canonical_common_order]
    train_common_ids = train_common_ids[canonical_common_order]
    train_common_positions = train_common_positions[canonical_common_order]
    train_local_positions = np.asarray([id_to_local[value] for value in train_common_ids])
    seed_common_labels = [values[train_local_positions] for values in seed_labels]
    baseline = random_geometry_baseline(
        train_common, common_labels[train_common_positions], n_clusters,
        algorithm, seed, replicates=20, observed_seed_labels=seed_common_labels,
    )

    confounding: Dict[str, Any] = {}
    for split in ("all", *TRAINING_SPLITS):
        select = np.ones(len(common_labels), dtype=bool) if split == "all" else common_splits == split
        if not select.any():
            confounding[split] = {
                "status": "not_evaluated",
                "reason": "no samples from this split were assigned",
            }
            continue
        confounding[split] = {
            "status": "evaluated",
            "cramers_v_cluster_slide": _cramers_v(common_labels[select], common_slides[select]),
            "nmi_cluster_slide": float(normalized_mutual_info_score(
                common_labels[select], common_slides[select]
            )),
            "max_dominant_slide_fraction": _max_dominant_group_fraction(
                common_labels[select], common_slides[select]
            ),
            "dominant_slide_fraction_by_cluster": {
                f"C{int(label)}": float(
                    pd.Series(common_slides[select][common_labels[select] == label])
                    .value_counts(normalize=True).max()
                )
                for label in np.unique(common_labels[select])
            },
        }
    technical = {}
    if "completeness" in assignments:
        technical["completeness_eta2"] = _eta_squared(
            assignments.loc[fit_mask, "completeness"], assignments.loc[fit_mask, "cluster"]
        )
    for column in ("border_distance_px", "tissue_ratio", "background_mean_intensity"):
        if column in assignments:
            technical[f"{column}_eta2"] = _eta_squared(
                assignments.loc[fit_mask, column], assignments.loc[fit_mask, "cluster"]
            )

    profiles = {}
    profile_mask = common_splits == "train"
    profile_labels = common_labels[profile_mask]
    profile_slides = common_slides[profile_mask]
    profile_ids = common_ids[profile_mask]
    rng = np.random.default_rng(seed + 10000)
    raw_common = data["common_morph_raw"][common_indices][profile_mask]
    feature_names = data["common_feature_names"].astype(str)
    cohort_medians = np.median(raw_common, axis=0)
    for label in np.unique(profile_labels):
        subset = raw_common[profile_labels == label]
        cluster_medians = np.median(subset, axis=0)
        descriptions = []
        for feature_name in ("glom_area_um2", "mean_optical_density"):
            if feature_name in feature_names:
                feature_index = int(np.flatnonzero(feature_names == feature_name)[0])
                direction = "higher" if cluster_medians[feature_index] > cohort_medians[feature_index] else "lower"
                descriptions.append(f"{feature_name} {direction} than cohort median")
        profiles[f"C{int(label)}"] = {
            "n_glomeruli": int(len(subset)),
            "n_slides": int(len(np.unique(profile_slides[profile_labels == label]))),
            "descriptive_profile": "; ".join(descriptions) or "No predefined numeric descriptor available",
            "features": {
                name: {
                    "median": float(np.median(subset[:, index])),
                    "q1": float(np.percentile(subset[:, index], 25)),
                    "q3": float(np.percentile(subset[:, index], 75)),
                }
                for index, name in enumerate(feature_names)
            },
            "random_examples": rng.choice(
                profile_ids[profile_labels == label],
                size=min(8, np.count_nonzero(profile_labels == label)), replace=False
            ).tolist(),
        }

    clustering_report = json.loads((run_dir / "clustering_report.json").read_text())
    report_train = clustering_report.get("train_metrics", {})
    if "model_diagnostics" in clustering_report:
        fit_records.insert(0, {"replicate_id": "saved_run", **clustering_report["model_diagnostics"]})
    metrics = {
        "evaluation_schema_version": CLUSTERING_EVALUATION_SCHEMA_VERSION,
        "resampling_method": resampling["method"],
        "paired_interval_label": (
            "empirical 2.5th-97.5th percentile range of matched replicate differences; "
            "not a confidence interval for the mean"
        ),
        "source_run": str(run_dir),
        "cohort_hash": cohort_hash,
        "resampling_hash": resampling_hash,
        "common_evaluation_space_hash": run_json.get("common_evaluation_space_hash"),
        "model_diagnostics": {
            "all_fits_valid": all(item["valid"] for item in fit_records),
            "n_fits": len(fit_records),
            "invalid_fit_ids": [item["replicate_id"] for item in fit_records if not item["valid"]],
            "fits": fit_records,
        },
        "stability": {
            "seed_ari_mean": float(np.mean(seed_ari)),
            "seed_ari_std": float(np.std(seed_ari, ddof=1)),
            "seed_ari_samples": seed_ari,
            "seed_ari_replicates": [
                {
                    "replicate_id": f"seed-{i:02d}|seed-{j:02d}",
                    "value": seed_ari[position],
                }
                for position, (i, j) in enumerate(itertools.combinations(range(10), 2))
            ],
            "subsampling_ari_mean": float(np.mean(subsampling_ari)),
            "subsampling_ari_std": float(np.std(subsampling_ari, ddof=1)),
            "subsampling_ari_min": float(np.min(subsampling_ari)),
            "subsampling_ari_samples": subsampling_ari,
            "subsampling_pair_replicates": [
                {
                    "replicate_id": pair_id,
                    "ari": ari_value,
                    "cluster_jaccard_mean": jaccard_mean,
                    "cluster_jaccard_min": jaccard_min,
                }
                for pair_id, ari_value, jaccard_mean, jaccard_min in zip(
                    subsampling_pair_ids, subsampling_ari,
                    subsampling_jaccard_mean_by_pair, subsampling_jaccard_min_by_pair,
                )
            ],
            "cluster_jaccard_mean": float(np.mean(subsampling_jaccard)),
            "cluster_jaccard_std": float(np.std(subsampling_jaccard, ddof=1)),
            "cluster_jaccard_min": float(np.min(subsampling_jaccard)),
            "cluster_jaccard_samples": subsampling_jaccard,
            "leave_one_slide_out_ari": loo_results,
            "leave_one_slide_out_ari_mean": float(np.mean(list(loo_results.values())))
            if loo_results else float("nan"),
            "leave_one_slide_out_ari_samples": list(loo_results.values()),
            "leave_one_slide_out_replicates": [
                {"replicate_id": f"leave-slide-{slide}", "value": value}
                for slide, value in sorted(loo_results.items())
            ],
        },
        "common_space_internal_metrics": common_metrics,
        "own_space_train_metrics": {
            "silhouette": report_train.get("silhouette_score"),
            "davies_bouldin": report_train.get("davies_bouldin_score"),
            "calinski_harabasz": report_train.get("calinski_harabasz_score"),
        },
        "random_baseline": baseline,
        "slide_confounding": confounding,
        "technical_confounding_eta2": technical,
        "cluster_profiles": profiles,
        "cluster_profiles_scope": "train",
        "paired_metric_samples": paired_samples,
        "paired_metric_records": paired_sample_records,
    }
    run = EvaluationRun(
        "clustering_eval", "train_val", config,
        output_root=config.get("output", {}).get("evaluation_root", "runs/clustering/evaluations"),
        run_name=config.get("output", {}).get("evaluation_name", run_dir.name),
        input_paths=[run_dir, resampling_path],
    )
    run.metadata.update({
        "evaluation_schema_version": CLUSTERING_EVALUATION_SCHEMA_VERSION,
        "cohort_hash": cohort_hash,
        "resampling_hash": resampling_hash,
        "common_evaluation_space_hash": run_json.get("common_evaluation_space_hash"),
    })
    write_json(run.run_json, run.metadata)
    (run.run_dir / "resampling.json").write_text(
        json.dumps(resampling, indent=2) + "\n", encoding="utf-8"
    )
    run.complete(metrics, [run.run_dir / "resampling.json"])
    return metrics, run


