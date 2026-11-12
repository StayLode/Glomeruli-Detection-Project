"""Shared train-only geometric reference for clustering metrics."""

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

from sklearn.cluster import KMeans


def random_geometry_baseline(
    common_train: np.ndarray,
    observed_labels: Sequence[int],
    n_clusters: int,
    algorithm: str,
    seed: int,
    replicates: int = 20,
    observed_seed_labels: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, Any]:
    """Compare observed partitions with a Gaussian null in common-space coordinates.

    The null preserves the training mean and covariance, but does not simulate
    the deep embedding or the configuration-specific feature/PCA pipeline. Both
    observed and null silhouettes are measured in the same common-space
    coordinates with Euclidean distance.
    """
    values = np.asarray(common_train, dtype=np.float64)
    labels = np.asarray(observed_labels)
    if values.ndim != 2 or len(values) != len(labels):
        raise ValueError("common_train and observed_labels must have matching rows")
    if not np.isfinite(values).all():
        raise ValueError("common_train contains NaN or infinite values")
    if replicates < 2:
        raise ValueError("At least two random baseline replicates are required")
    observed_silhouette, observed_reason = _safe_silhouette(values, labels)
    observed_stability = _seed_stability(observed_seed_labels)
    covariance = np.atleast_2d(np.cov(values, rowvar=False))
    if covariance.shape != (values.shape[1], values.shape[1]):
        covariance = np.diag(np.asarray(covariance).reshape(-1))
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    rng = np.random.default_rng(seed)
    null_silhouettes, null_stabilities = [], []
    reasons = []
    invalid_fits = []
    for replicate in range(replicates):
        null_values = rng.multivariate_normal(
            values.mean(axis=0), covariance, size=len(values), check_valid="ignore"
        )
        partitions = []
        first_labels = None
        for offset in range(10):
            estimator = _estimator(
                n_clusters, algorithm, seed + replicate * 100 + offset
            )
            estimator.fit(null_values)
            predicted = estimator.predict(null_values)
            counts = np.bincount(predicted, minlength=n_clusters)
            if np.count_nonzero(counts) != n_clusters:
                invalid_fits.append({"replicate_id": f"null-{replicate:03d}/seed-{offset:02d}", "counts": counts.tolist()})
            partitions.append(predicted)
            if offset == 0:
                first_labels = predicted
        score, reason = _safe_silhouette(null_values, first_labels)
        if score is None:
            reasons.append({"replicate_id": f"null-{replicate:03d}", "reason": reason})
        else:
            null_silhouettes.append(score)
        replicate_stability = _seed_stability(partitions)
        if replicate_stability is not None:
            null_stabilities.append(replicate_stability)
    silhouette_stats = _summarize(
        observed_silhouette, null_silhouettes, observed_reason
    )
    stability_stats = _summarize(
        observed_stability, null_stabilities,
        None if observed_stability is not None else "observed seed partitions unavailable",
    )
    # Keep all values as diagnostic evidence, but never silently drop failed fits
    # and report a passing z-score from the remaining, more favorable subset.
    if invalid_fits:
        for stats in (silhouette_stats, stability_stats):
            stats.update(z_score=None, status="not_evaluable",
                         reason="null contains fewer than K occupied groups")
    return {
        "status": "evaluated" if silhouette_stats["z_score"] is not None else "not_evaluable",
        "null_model": "multivariate_gaussian_preserving_train_mean_covariance",
        "scope": "geometric reference in common native-morphology space; not a full hybrid-pipeline null",
        "space": "common_train_standardized_morphology",
        "distance": "euclidean",
        "n_clusters": int(n_clusters),
        "algorithm": algorithm,
        "estimator_parameters": {"n_init": 10},
        "fit_diagnostics": {"n_fits": replicates * 10, "invalid_fits": invalid_fits,
                            "all_fits_valid": not invalid_fits},
        "seed": int(seed),
        "replicates_requested": int(replicates),
        "replicate_ids": [f"null-{index:03d}" for index in range(replicates)],
        "observed_silhouette": observed_silhouette,
        "observed_seed_stability": observed_stability,
        "silhouette": silhouette_stats,
        "seed_stability": stability_stats,
        "null_silhouette_values": null_silhouettes,
        "null_seed_stability_values": null_stabilities,
        "excluded_replicates": reasons,
        "empirical_tail": (
            (1 + sum(value >= observed_silhouette for value in null_silhouettes))
            / (len(null_silhouettes) + 1)
            if observed_silhouette is not None and null_silhouettes else None
        ),
        "empirical_percentile_95": (
            float(np.percentile(null_silhouettes, 95))
            if null_silhouettes else None
        ),
        "protocol_z_threshold": 2.0,
    }


def _estimator(n_clusters: int, algorithm: str, seed: int) -> Any:
    if algorithm != "kmeans":
        raise ValueError("Only K-Means is supported")
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)


def _safe_silhouette(
    features: np.ndarray, labels: Optional[np.ndarray]
) -> tuple[Optional[float], Optional[str]]:
    if labels is None or len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return None, "degenerate cluster partition"
    try:
        value = float(silhouette_score(features, labels, metric="euclidean"))
    except ValueError as error:
        return None, str(error)
    if not np.isfinite(value):
        return None, "silhouette is not finite"
    return value, None


def _seed_stability(
    partitions: Optional[Sequence[Sequence[int]]]
) -> Optional[float]:
    if partitions is None or len(partitions) < 2:
        return None
    values = [
        adjusted_rand_score(partitions[left], partitions[right])
        for left in range(len(partitions))
        for right in range(left + 1, len(partitions))
    ]
    return float(np.mean(values)) if values else None


def _summarize(
    observed: Optional[float], null_values: Sequence[float], observed_reason: Optional[str]
) -> Dict[str, Any]:
    if observed is None or len(null_values) < 2:
        return {
            "observed": observed,
            "null_mean": float(np.mean(null_values)) if null_values else None,
            "null_std": float(np.std(null_values, ddof=1)) if len(null_values) > 1 else None,
            "z_score": None,
            "status": "not_evaluable",
            "reason": observed_reason or "fewer than two valid null values",
        }
    mean = float(np.mean(null_values))
    std = float(np.std(null_values, ddof=1))
    if std <= np.finfo(float).eps:
        return {
            "observed": observed, "null_mean": mean, "null_std": std,
            "z_score": None, "status": "not_evaluable",
            "reason": "null standard deviation is zero",
        }
    return {
        "observed": observed,
        "null_mean": mean,
        "null_std": std,
        "z_score": float((observed - mean) / std),
        "status": "evaluated",
        "reason": None,
    }
