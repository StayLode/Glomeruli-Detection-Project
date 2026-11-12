"""Content-addressed cache of pretrained ResNet embeddings."""
import hashlib, inspect, json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from src.clustering.feature_extractor import GlomeruliFeatureExtractor
from src.clustering.data import load_cohort_samples
from src.utils.evaluation import file_identity
PROJECT_ROOT=Path(__file__).resolve().parents[2]

def _cached_embeddings(
    crops: List[np.ndarray],
    metadata: List[Dict[str, Any]],
    extractor: GlomeruliFeatureExtractor,
    feature_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    manifest_path: str,
    cohort_path: Optional[str],
) -> np.ndarray:
    image_identities = []
    root = Path(manifest_path).resolve().parent
    for relative_path in sorted({item.get("image_path", "") for item in metadata}):
        if relative_path:
            image_identities.append(file_identity(root / relative_path))
    cohort_run_path = Path(cohort_path).resolve().parent / "run.json" if cohort_path else None
    cohort_identity = (
        json.loads(cohort_run_path.read_text(encoding="utf-8")).get("cohort_hash")
        if cohort_run_path and cohort_run_path.is_file()
        else file_identity(cohort_path)["sha256"] if cohort_path else None
    )
    cache_parameters = {
        "cohort_hash": cohort_identity,
        "manifest_hash": file_identity(manifest_path)["sha256"],
        "image_hashes": image_identities,
        "annotation_ids": [str(item["annotation_id"]) for item in metadata],
        "crop_mode": data_cfg.get("crop_mode", "legacy"),
        "mask_mode": data_cfg.get("mask_mode", "legacy"),
        "crop_size": int(data_cfg.get("crop_size", 512)),
        "margin_ratio": float(data_cfg.get("margin_ratio", 0.25)),
        "padding_mode": data_cfg.get("padding_mode", "reflect"),
        "preprocessing_schema_version": 2,
        "crop_code_hash": file_identity(PROJECT_ROOT / "src/clustering/crops.py")["sha256"],
        "cohort_loader_code_hash": hashlib.sha256(
            inspect.getsource(load_cohort_samples).encode("utf-8")
        ).hexdigest(),
        "deep_embedding_code_hash": hashlib.sha256(
            (
                inspect.getsource(GlomeruliFeatureExtractor.__init__)
                + inspect.getsource(GlomeruliFeatureExtractor.extract_deep_embeddings)
            ).encode("utf-8")
        ).hexdigest(),
        "backbone": feature_cfg.get("deep_backbone", "resnet34"),
        "pretrained": bool(feature_cfg.get("pretrained", True)),
        "weights_sha256": extractor.weights_sha256(),
        "normalize_embeddings": bool(feature_cfg.get("normalize_embeddings", True)),
    }
    key = hashlib.sha256(json.dumps(cache_parameters, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(feature_cfg.get("embedding_cache_dir", "runs/clustering/cache/embeddings"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.npz"
    expected_ids = np.asarray(cache_parameters["annotation_ids"])
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_ids, embeddings = archive["annotation_id"], archive["deep_embeddings"]
        if not np.array_equal(cached_ids, expected_ids):
            raise ValueError(f"Embedding cache ID mismatch: {cache_path}")
        return embeddings
    embeddings = extractor.extract_deep_embeddings(crops)
    with cache_path.open("xb") as stream:
        np.savez_compressed(stream, annotation_id=expected_ids, deep_embeddings=embeddings)
    return embeddings
