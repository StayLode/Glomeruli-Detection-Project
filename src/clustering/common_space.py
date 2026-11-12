"""Shared native-morphology evaluation space for paired clustering comparisons."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.clustering.cohort import COHORT_SCHEMA_VERSION
from src.clustering.feature_extractor import extract_single_morphological_features
from src.utils.evaluation import file_identity


def load_or_create_common_space(
    cohort_path: str,
    manifest_path: str,
    crop_size: int = 512,
    margin_ratio: float = 0.25,
    padding_mode: str = "reflect",
    cache_dir: str = "runs/clustering/cache/common_spaces",
    allow_legacy_cohort: bool = False,
    xml_dir: str = "glomeruli_grading",
) -> Tuple[Dict[str, np.ndarray], str]:
    """Cache native single-component morphology standardized on train only."""
    from src.clustering.data import load_cohort_samples

    cohort_run_path = Path(cohort_path).resolve().parent / "run.json"
    cohort_metadata = (
        json.loads(cohort_run_path.read_text(encoding="utf-8"))
        if cohort_run_path.is_file() else {}
    )
    cohort_schema_version = cohort_metadata.get("cohort_schema_version")
    if cohort_schema_version != COHORT_SCHEMA_VERSION and not allow_legacy_cohort:
        raise ValueError(
            "The common evaluation space requires a schema-v2 cohort. "
            "Rebuild it with scripts/train_clustering.py --rebuild-cohort --overwrite, or explicitly "
            "enable data.allow_legacy_cohort for diagnostic use."
        )
    cohort_identity = cohort_metadata.get("cohort_hash", file_identity(cohort_path)["sha256"])
    cohort_rows = pd.read_csv(cohort_path)
    image_root = Path(manifest_path).resolve().parent
    image_hashes = [
        file_identity(image_root / relative)
        for relative in sorted(set(cohort_rows["image_path"].astype(str)))
    ]
    mask_hashes = [
        file_identity(image_root / relative)
        for relative in sorted(set(cohort_rows["mask_path"].astype(str)))
    ]
    parameters = {
        "common_space_schema_version": 2,
        "cohort_schema_version": cohort_schema_version or "legacy_unknown",
        "legacy_diagnostic": cohort_schema_version != COHORT_SCHEMA_VERSION,
        "cohort_hash": cohort_identity, "crop_size": crop_size,
        "margin_ratio": margin_ratio, "padding_mode": padding_mode,
        "crop_mode": "square_padded", "mask_mode": "single_component",
        "morph_units": "native", "pixel_size_um_40x": 0.2519,
        "image_hashes": image_hashes,
        "mask_hashes": mask_hashes,
        "preprocessing_code_hashes": {
            "cohort_loader": file_identity(Path(__file__).resolve().parent / "cohort.py")["sha256"],
            "crop": file_identity(Path(__file__).resolve().parent / "crops.py")["sha256"],
            "feature_extractor": file_identity(Path(__file__).resolve().parent / "feature_extractor.py")["sha256"],
        },
    }
    key = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f"{key}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            data = {name: stored[name] for name in stored.files}
        return data, file_identity(path)["sha256"]

    splits = list(dict.fromkeys(cohort_rows["split"].astype(str).tolist()))
    crops, masks, metadata, pixel_sizes, exclusions = load_cohort_samples(
        cohort_path, manifest_path, splits, crop_size, "square_padded",
        "single_component", margin_ratio, padding_mode,
        allow_legacy_cohort=allow_legacy_cohort,
        xml_dir=xml_dir,
    )
    if not exclusions.empty:
        raise RuntimeError(f"Common evaluation space has excluded rows: {exclusions['reason'].value_counts().to_dict()}")
    feature_rows = [
        extract_single_morphological_features(crop, mask, "native", pixel_size)
        for crop, mask, pixel_size in zip(crops, masks, pixel_sizes)
    ]
    feature_names = list(feature_rows[0]) if feature_rows else []
    raw = np.asarray([[row[name] for name in feature_names] for row in feature_rows], dtype=np.float64)
    ids = np.asarray([item["annotation_id"] for item in metadata])
    splits_array = np.asarray([item["split"] for item in metadata])
    fit_mask = splits_array == "train"
    if not fit_mask.any():
        raise ValueError("The common space requires at least one train sample")
    mean = raw[fit_mask].mean(axis=0)
    scale = raw[fit_mask].std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (raw - mean) / scale
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, annotation_id=ids, split=splits_array,
        slide_id=np.asarray([item["slide_id"] for item in metadata]),
        feature_names=np.asarray(feature_names), raw_features=raw,
        standardized_features=standardized, train_mean=mean, train_scale=scale,
        legacy_diagnostic=np.asarray(cohort_schema_version != COHORT_SCHEMA_VERSION),
    )
    with np.load(path, allow_pickle=False) as stored:
        data = {name: stored[name] for name in stored.files}
    return data, file_identity(path)["sha256"]
