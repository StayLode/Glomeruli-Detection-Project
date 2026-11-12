"""Create or reuse annotation-aligned subsampling plans and slide holdouts."""

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.utils.evaluation import file_identity


def create_or_load_resampling(
    path: Path, train_ids: Sequence[str], slides: Sequence[str], cohort_hash: str, seed: int
) -> Tuple[Dict[str, Any], str]:
    ids = list(map(str, train_ids))
    if len(set(ids)) != len(ids):
        raise ValueError("Training annotation_id values must be unique")
    if len(slides) != len(ids):
        raise ValueError("slides must align one-to-one with train_ids")
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != 2:
            raise ValueError("Existing resampling.json uses an obsolete protocol; choose a new versioned path")
        if payload.get("cohort_hash") != cohort_hash or payload.get("train_ids") != sorted(ids):
            raise ValueError("Existing resampling.json belongs to a different cohort or train ID set")
    else:
        rng = np.random.default_rng(seed)
        canonical_ids = sorted(ids)
        n = max(2, int(np.ceil(0.8 * len(canonical_ids))))
        slide_by_id = dict(zip(ids, map(str, slides)))
        slide_groups: Dict[str, List[str]] = {}
        for annotation_id in canonical_ids:
            slide_groups.setdefault(slide_by_id[annotation_id], []).append(annotation_id)
        folds = []
        for slide in sorted(slide_groups):
            held_out = slide_groups[slide]
            folds.append({
                "fold_id": f"leave-slide-{slide}",
                "held_out_slide_id": slide,
                "seed": int(seed),
                "fit_annotation_ids": [
                    annotation_id for annotation_id in canonical_ids
                    if slide_by_id[annotation_id] != slide
                ],
                "holdout_annotation_ids": held_out,
            })
        payload = {
            "protocol_version": 2,
            "method": "subsampling_without_replacement",
            "sample_fraction": 0.8,
            "seed": seed,
            "cohort_hash": cohort_hash,
            "train_ids": canonical_ids,
            "subsampling_replicates": [
                {
                    "replicate_id": f"subsample-{index:03d}",
                    "seed": int(seed + index),
                    "annotation_ids": sorted(
                        rng.choice(canonical_ids, size=n, replace=False).tolist()
                    ),
                }
                for index in range(20)
            ],
            "leave_one_slide_out": folds,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    return payload, file_identity(path)["sha256"]


