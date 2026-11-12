"""Clustering metrics with explicit non-evaluable cases and neutral numeric IDs."""
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def _matched_cluster_jaccards(
    left: np.ndarray, right: np.ndarray, n_clusters: Optional[int] = None
) -> List[float]:
    left_ids = right_ids = (
        np.arange(n_clusters) if n_clusters is not None
        else np.union1d(np.unique(left), np.unique(right))
    )
    matrix = np.zeros((len(left_ids), len(right_ids)), dtype=np.float64)
    for i, left_id in enumerate(left_ids):
        left_mask = left == left_id
        for j, right_id in enumerate(right_ids):
            right_mask = right == right_id
            intersection = np.count_nonzero(left_mask & right_mask)
            union = np.count_nonzero(left_mask | right_mask)
            matrix[i, j] = intersection / union if union else 0.0
    rows, cols = linear_sum_assignment(-matrix)
    return [float(matrix[i, j]) for i, j in zip(rows, cols)]

def _metric_result(features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    n_labels = len(np.unique(labels))
    if len(features) == 0:
        return {
            "status": "not_evaluated", "reason": "nessun campione assegnato",
            "silhouette": None, "davies_bouldin": None, "calinski_harabasz": None,
        }
    if n_labels < 2:
        reason = "è presente un solo cluster nel split"
    elif n_labels >= len(labels):
        reason = "ogni campione forma un cluster distinto o il campione è troppo piccolo"
    else:
        reason = None
    if reason:
        return {
            "status": "not_evaluable", "reason": reason,
            "silhouette": None, "davies_bouldin": None, "calinski_harabasz": None,
        }
    return {
        "status": "evaluated", "reason": None,
        "silhouette": float(silhouette_score(features, labels)),
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
    }

def _cramers_v(frame: pd.DataFrame) -> Dict[str, Any]:
    n_slides = frame["slide_id"].nunique()
    n_clusters = frame["cluster"].nunique()
    if n_slides < 2:
        return {
            "value": None,
            "status": "not_evaluable",
            "reason": "un solo slide: l'associazione cluster-slide non è stimabile",
        }
    if n_clusters < 2:
        return {
            "value": None,
            "status": "not_evaluable",
            "reason": "un solo cluster occupato: manca variabilità nei cluster",
        }
    table = pd.crosstab(frame["cluster"], frame["slide_id"]).to_numpy()
    chi2 = float(chi2_contingency(table, correction=False)[0])
    denominator = table.sum() * min(table.shape[0] - 1, table.shape[1] - 1)
    return {
        "value": float(np.sqrt(chi2 / denominator)) if denominator else None,
        "status": "evaluated" if denominator else "not_evaluable",
        "reason": None if denominator else "tabella priva di gradi di libertà",
    }

def _eta_squared(values: Sequence[Any], labels: Sequence[Any]) -> Optional[float]:
    value_array = np.asarray(values, dtype=float)
    label_array = np.asarray(labels)
    valid = np.isfinite(value_array)
    value_array, label_array = value_array[valid], label_array[valid]
    if len(value_array) < 2 or np.var(value_array) == 0 or len(np.unique(label_array)) < 2:
        return None
    mean = value_array.mean()
    between = sum(
        np.count_nonzero(label_array == label)
        * (value_array[label_array == label].mean() - mean) ** 2
        for label in np.unique(label_array)
    )
    return float(between / np.sum((value_array - mean) ** 2))

def _profile_table(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    train_frame: pd.DataFrame,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for cluster in sorted(frame["cluster"].unique()):
        held_rows = frame[frame["cluster"] == cluster]
        train_rows = train_frame[train_frame["cluster"] == cluster]
        cluster_profile: Dict[str, Any] = {}
        for name in feature_names:
            values = pd.to_numeric(held_rows[name], errors="coerce").dropna()
            train_values = pd.to_numeric(train_rows[name], errors="coerce").dropna()
            if values.empty:
                cluster_profile[name] = {
                    "n": 0, "median": None, "mean": None, "q25": None, "q75": None,
                    "train_median": None, "median_delta_vs_train": None,
                }
                continue
            median = float(values.median())
            train_median = float(train_values.median()) if not train_values.empty else None
            cluster_profile[name] = {
                "n": int(len(values)),
                "median": median,
                "mean": float(values.mean()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "train_median": train_median,
                "median_delta_vs_train": (
                    median - train_median if train_median is not None else None
                ),
            }
        result[f"C{int(cluster)}"] = cluster_profile
    return result

def _distance_profiles(frame: pd.DataFrame, n_clusters: int) -> Dict[str, Any]:
    profiles: Dict[str, Any] = {}
    for cluster in range(n_clusters):
        values = frame.loc[frame["cluster"] == cluster, "distance_to_center"].to_numpy(float)
        profiles[f"C{cluster}"] = {
            "n": int(len(values)),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "q95": float(np.quantile(values, 0.95)) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
            "unit": "distanza euclidea nello spazio PCA20 del modello; non è probabilità",
        }
    return profiles
