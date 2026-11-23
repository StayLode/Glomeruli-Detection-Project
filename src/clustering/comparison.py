#!/usr/bin/env python3
"""Compare clustering evaluation runs without making an automatic improvement claim."""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


from src.utils.evaluation import (
    CLUSTERING_EVALUATION_SCHEMA_VERSION, create_artifact_directory, write_json,
)


METRICS = {
    "stability.seed_ari_mean": ("higher", "stability.seed_ari_replicates", "value"),
    "stability.subsampling_ari_mean": (
        "higher", "stability.subsampling_pair_replicates", "ari"
    ),
    "stability.cluster_jaccard_mean": (
        "higher", "stability.subsampling_pair_replicates", "cluster_jaccard_mean"
    ),
    "stability.cluster_jaccard_min": (
        "higher", "stability.subsampling_pair_replicates", "cluster_jaccard_min"
    ),
    "stability.leave_one_slide_out_ari_mean": (
        "higher", "stability.leave_one_slide_out_replicates", "value"
    ),
    "common_space_internal_metrics.train.silhouette": (
        "higher", "paired_metric_records.train_silhouette", "value"
    ),
    "common_space_internal_metrics.train.davies_bouldin": (
        "lower", "paired_metric_records.train_davies_bouldin", "value"
    ),
    "common_space_internal_metrics.train.calinski_harabasz": (
        "higher", "paired_metric_records.train_calinski_harabasz", "value"
    ),
    "common_space_internal_metrics.val.silhouette": ("higher", None),
    "common_space_internal_metrics.val.davies_bouldin": ("lower", None),
    "common_space_internal_metrics.val.calinski_harabasz": ("higher", None),
    "own_space_train_metrics.silhouette": ("higher", None),
    "own_space_train_metrics.davies_bouldin": ("lower", None),
    "slide_confounding.train.cramers_v_cluster_slide": (
        "lower", "paired_metric_records.train_cramers_v_slide", "value"
    ),
    "slide_confounding.train.nmi_cluster_slide": (
        "lower", "paired_metric_records.train_nmi_slide", "value"
    ),
    "slide_confounding.train.max_dominant_slide_fraction": (
        "lower", "paired_metric_records.train_dominant_slide_fraction", "value"
    ),
    "random_baseline.silhouette.z_score": ("higher", None, None),
    "random_baseline.seed_stability.z_score": ("higher", None, None),
    "technical_confounding_eta2.completeness_eta2": (
        "lower", "paired_metric_records.completeness_eta2", "value"
    ),
    "technical_confounding_eta2.border_distance_px_eta2": (
        "lower", "paired_metric_records.border_distance_px_eta2", "value"
    ),
    "technical_confounding_eta2.tissue_ratio_eta2": (
        "lower", "paired_metric_records.tissue_ratio_eta2", "value"
    ),
    "technical_confounding_eta2.background_mean_intensity_eta2": (
        "lower", "paired_metric_records.background_mean_intensity_eta2", "value"
    ),
}


def _value(document: Dict[str, Any], dotted_path: str) -> Optional[float]:
    node: Any = document
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return float(node) if isinstance(node, (int, float)) and np.isfinite(node) else None


def _array(document: Dict[str, Any], dotted_path: Optional[str]) -> np.ndarray:
    if dotted_path is None:
        return np.asarray([], dtype=float)
    node: Any = document
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return np.asarray([], dtype=float)
        node = node[key]
    return np.asarray(node, dtype=float)


def _load_eval(
    path: Path, allow_obsolete: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metadata = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if metadata.get("stage") != "clustering_eval" or metadata.get("status") != "completed":
        raise ValueError(f"Not a completed clustering_eval run: {path}")
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if (
        metadata.get("evaluation_schema_version") != CLUSTERING_EVALUATION_SCHEMA_VERSION
        or metrics.get("evaluation_schema_version") != CLUSTERING_EVALUATION_SCHEMA_VERSION
    ):
        if allow_obsolete:
            metadata["_obsolete_protocol"] = True
            return metadata, metrics
        raise ValueError(
            f"Evaluation uses an obsolete protocol and is diagnostic only: {path}"
        )
    return metadata, metrics


def _paired_map(
    document: Dict[str, Any], path: Optional[str], value_key: Optional[str]
) -> Dict[str, float]:
    if path is None:
        return {}
    node: Any = document
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return {}
        node = node[key]
    if not isinstance(node, list):
        return {}
    values = {}
    for item in node:
        if not isinstance(item, dict) or "replicate_id" not in item:
            continue
        value = item.get(value_key or "value")
        if isinstance(value, (int, float)) and np.isfinite(value):
            values[str(item["replicate_id"])] = float(value)
    return values


def compare_runs(run_paths: List[Path], descriptive_only: bool = False) -> List[Dict[str, Any]]:
    if len(run_paths) < 2:
        raise ValueError("Provide at least two clustering_eval run directories")
    loaded = [_load_eval(path, allow_obsolete=descriptive_only) for path in run_paths]
    identities = [
        (
            metadata.get("cohort_hash"),
            metadata.get("resampling_hash"),
            metadata.get("common_evaluation_space_hash"),
        )
        for metadata, _ in loaded
    ]
    for left_index, right_index in itertools.combinations(range(len(identities)), 2):
        left_identity, right_identity = identities[left_index], identities[right_index]
        left_obsolete = bool(loaded[left_index][0].get("_obsolete_protocol"))
        right_obsolete = bool(loaded[right_index][0].get("_obsolete_protocol"))
        if (left_obsolete or right_obsolete) and not (
            descriptive_only and left_identity[0] != right_identity[0]
        ):
            raise ValueError(
                "Obsolete evaluation output is diagnostic only; compare it only "
                "descriptively against a different cohort."
            )
        if left_identity != right_identity and not (
            descriptive_only and left_identity[0] != right_identity[0]
        ):
            raise ValueError(
                "Runs must have identical cohort, resampling, and common-space hashes. "
                "--descriptive_only only permits explicitly cross-cohort rows."
            )

    rows = []
    for left_index, right_index in itertools.combinations(range(len(loaded)), 2):
        _, left_metrics = loaded[left_index]
        _, right_metrics = loaded[right_index]
        not_comparable = identities[left_index][0] != identities[right_index][0]
        for metric_path, metric_spec in METRICS.items():
            direction, samples_path = metric_spec[:2]
            sample_value_key = metric_spec[2] if len(metric_spec) > 2 else None
            left_value, right_value = _value(left_metrics, metric_path), _value(right_metrics, metric_path)
            deltas = None
            interval = None
            status = "NOT comparable: different cohorts" if not_comparable else "inconclusive"
            own_space = metric_path.startswith("own_space_train_metrics.")
            if own_space and not not_comparable:
                status = "NOT comparable: different feature spaces"
            if not not_comparable and not own_space and left_value is not None and right_value is not None:
                delta = right_value - left_value
                if samples_path:
                    left_samples = _paired_map(left_metrics, samples_path, sample_value_key)
                    right_samples = _paired_map(right_metrics, samples_path, sample_value_key)
                    shared_ids = sorted(set(left_samples) & set(right_samples))
                    if len(shared_ids) >= 2:
                        deltas = [
                            right_samples[replicate_id] - left_samples[replicate_id]
                            for replicate_id in shared_ids
                        ]
                        interval = np.percentile(deltas, [2.5, 97.5]).tolist()
                        favorable = interval[0] > 0 if direction == "higher" else interval[1] < 0
                        unfavorable = interval[1] < 0 if direction == "higher" else interval[0] > 0
                        if metric_path.endswith("silhouette") and abs(float(np.mean(deltas))) < 0.02:
                            status = "inconclusive: silhouette difference < 0.02"
                        elif favorable:
                            status = "favorable"
                        elif unfavorable:
                            status = "unfavorable"
                    elif len(shared_ids) == 1:
                        deltas = [
                            right_samples[shared_ids[0]] - left_samples[shared_ids[0]]
                        ]
                        status = "inconclusive: fewer than two valid paired replicates"
                    else:
                        status = "inconclusive: no valid paired replicates"
                else:
                    status = "directional only: single measurement, no paired interval"
                rows.append({
                    "baseline_run": str(run_paths[left_index]),
                    "candidate_run": str(run_paths[right_index]),
                    "metric": metric_path,
                    "baseline_value": left_value,
                    "candidate_value": right_value,
                    "delta_candidate_minus_baseline": None if not_comparable else float(delta),
                    "mean_paired_difference": (
                        float(np.mean(deltas)) if deltas is not None and not not_comparable else None
                    ),
                    "paired_empirical_range_95": interval,
                    "paired_interval_type": (
                        "empirical_percentile_range_not_mean_confidence_interval"
                        if interval is not None else None
                    ),
                    "n_paired_replicates": len(deltas) if deltas is not None else 0,
                    "favorable_direction": direction,
                    "status": status,
                    "paired_differences": deltas,
                })
            else:
                rows.append({
                    "baseline_run": str(run_paths[left_index]),
                    "candidate_run": str(run_paths[right_index]),
                    "metric": metric_path,
                    "baseline_value": left_value,
                    "candidate_value": right_value,
                    "delta_candidate_minus_baseline": None,
                    "mean_paired_difference": None,
                    "paired_empirical_range_95": None,
                    "paired_interval_type": None,
                    "n_paired_replicates": 0,
                    "favorable_direction": direction,
                    "status": status if not not_comparable else "NOT comparable: different cohorts",
                    "paired_differences": None,
                })
    return rows



