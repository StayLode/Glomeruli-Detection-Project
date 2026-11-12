"""Load frozen cohort views and sample slides for clustering."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.clustering.cohort import COHORT_SCHEMA_VERSION
from src.clustering.crops import square_padded_crop
from src.models.mask_to_bbox import extract_bboxes_from_mask
from src.models.unet_dataset import extract_glomerulus_crop
from src.utils.evaluation import file_identity


def load_and_extract_glomeruli_crops(
    manifest_path: str,
    splits: List[str],
    crop_size: int = 512,
    margin_ratio: float = 0.25,
    min_glom_area: int = 400,
    max_samples: Optional[int] = None,
    exclusions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
    """Compatibility loader for legacy, patch-level (non-deduplicated) runs."""
    manifest_p = Path(manifest_path)
    df = pd.read_csv(manifest_p)
    df_filtered = df[(df["split"].isin(splits)) & (df["is_positive"] == True)]
    crops_rgb: List[np.ndarray] = []
    masks_binary: List[np.ndarray] = []
    metadata_list: List[Dict[str, Any]] = []
    for _, row in df_filtered.iterrows():
        img_path, mask_path = manifest_p.parent / row["image_path"], manifest_p.parent / row["mask_path"]
        img_bgr = cv2.imread(str(img_path))
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or mask_gray is None:
            logging.warning("Excluding legacy sample with missing image or mask: %s / %s", img_path, mask_path)
            if exclusions is not None:
                exclusions.append({
                    "patch_id": row["patch_id"], "slide_id": row["slide_id"],
                    "reason": "missing_file", "detail": f"{img_path if img_bgr is None else mask_path}",
                })
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask_bin = (mask_gray > 127).astype(np.uint8) * 255
        _, _, component_stats, _ = cv2.connectedComponentsWithStats(
            (mask_bin > 0).astype(np.uint8), connectivity=8
        )
        if exclusions is not None:
            for component_label, stats in enumerate(component_stats[1:], start=1):
                if int(stats[cv2.CC_STAT_AREA]) < min_glom_area:
                    exclusions.append({
                        "patch_id": row["patch_id"], "slide_id": row["slide_id"],
                        "component_label": component_label, "reason": "area_minima",
                        "area": int(stats[cv2.CC_STAT_AREA]),
                    })
        detections = extract_bboxes_from_mask(mask_bin, mask_bin.astype(float) / 255.0, min_area=min_glom_area)
        for det in detections:
            crop_img, crop_coords = extract_glomerulus_crop(
                img_rgb, det["bbox"], margin_ratio=margin_ratio, target_size=crop_size
            )
            x1, y1, x2, y2 = crop_coords
            crop_mask = mask_bin[y1:y2, x1:x2]
            crop_mask = (
                cv2.resize(crop_mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
                if crop_mask.size else np.zeros((crop_size, crop_size), dtype=np.uint8)
            )
            crops_rgb.append(crop_img)
            masks_binary.append(crop_mask)
            metadata_list.append({
                "annotation_id": f"{row['patch_id']}#legacy-{len(metadata_list)}",
                "patch_id": row["patch_id"], "slide_id": row["slide_id"], "split": row["split"],
                "bbox_patch": json.dumps(det["bbox"]), "glom_area_native": det["area"],
                "mask_path": str(row["mask_path"]), "image_path": str(row["image_path"]),
            })
            if max_samples is not None and len(crops_rgb) >= max_samples:
                return crops_rgb, masks_binary, metadata_list
    return crops_rgb, masks_binary, metadata_list


def load_cohort_samples(
    cohort_path: str,
    manifest_path: str,
    splits: List[str],
    crop_size: int,
    crop_mode: str,
    mask_mode: str,
    margin_ratio: float,
    padding_mode: str,
    max_samples: Optional[int] = None,
    sampling: str = "stratified_by_slide",
    sampling_seed: int = 42,
    allow_legacy_cohort: bool = False,
    xml_dir: str = "glomeruli_grading",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]], List[float], pd.DataFrame]:
    """Load exactly the frozen cohort rows selected for the requested splits."""
    if crop_mode not in {"legacy", "square_padded"}:
        raise ValueError(f"Unknown crop_mode: {crop_mode}")
    if mask_mode not in {"legacy", "single_component"}:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")
    cohort_run_path = Path(cohort_path).resolve().parent / "run.json"
    cohort_metadata = (
        json.loads(cohort_run_path.read_text(encoding="utf-8"))
        if cohort_run_path.is_file() else {}
    )
    schema_version = cohort_metadata.get("cohort_schema_version")
    if schema_version not in (None, 1, COHORT_SCHEMA_VERSION):
        raise ValueError(f"Unsupported cohort schema version: {schema_version}")
    if schema_version != COHORT_SCHEMA_VERSION and not allow_legacy_cohort:
        raise ValueError(
            f"Unsupported cohort schema {schema_version!r}; expected "
            f"{COHORT_SCHEMA_VERSION}. Rebuild the cohort, or set "
            "data.allow_legacy_cohort=true for explicitly diagnostic legacy use."
        )
    if schema_version != COHORT_SCHEMA_VERSION:
        logging.warning(
            "Using a legacy cohort without schema-v2 geometry validation; "
            "this run is diagnostic and excluded from corrected evaluations."
        )
    cohort = pd.read_csv(cohort_path)
    selected = cohort[cohort["split"].isin(splits)].reset_index(drop=True)
    if schema_version == COHORT_SCHEMA_VERSION:
        expected_manifest = (
            cohort_metadata.get("inputs", {}).get("manifest", {}).get("sha256")
        )
        if expected_manifest and file_identity(manifest_path)["sha256"] != expected_manifest:
            raise ValueError("Manifest hash differs from the frozen cohort provenance")
        expected_masks = {
            entry["path"]: entry["sha256"]
            for entry in cohort_metadata.get("inputs", {}).get("masks", [])
        }
        expected_xml = {
            entry["path"]: entry["sha256"]
            for entry in cohort_metadata.get("inputs", {}).get("xml", [])
        }
        for relative in sorted(set(selected["mask_path"].astype(str))):
            path = (Path(manifest_path).resolve().parent / relative).resolve()
            if not path.is_file():
                continue
            expected = expected_masks.get(str(path))
            if expected and file_identity(path)["sha256"] != expected:
                raise ValueError(f"Mask hash differs from the frozen cohort: {path}")
        for slide_id in sorted(set(selected["slide_id"].astype(str))):
            path = (Path(xml_dir) / f"{slide_id}.xml").resolve()
            if path.is_file():
                expected = expected_xml.get(str(path))
                if expected and file_identity(path)["sha256"] != expected:
                    raise ValueError(f"XML hash differs from the frozen cohort: {path}")
    if sampling == "first_n":
        raise ValueError("first_n sampling is only allowed for legacy E0_hist runs")
    if sampling != "stratified_by_slide":
        raise ValueError(f"Unknown sampling policy: {sampling}")
    if max_samples is not None and max_samples < len(selected):
        selected = stratified_sample_rows(selected, max_samples, sampling_seed)
    root = Path(manifest_path).resolve().parent
    crops: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    pixel_sizes_um: List[float] = []
    exclusions: List[Dict[str, Any]] = []
    for _, row in selected.iterrows():
        image_path, mask_path = root / str(row["image_path"]), root / str(row["mask_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            exclusions.append({
                "annotation_id": row["annotation_id"], "slide_id": row["slide_id"],
                "patch_id": row["patch_id"], "reason": "missing_file",
                "detail": f"{image_path if image is None else mask_path}",
            })
            continue
        component_count, labels, _, _ = cv2.connectedComponentsWithStats(
            (mask > 127).astype(np.uint8), connectivity=8
        )
        component_label = int(row["component_label"])
        if component_label <= 0 or component_label >= component_count:
            exclusions.append({
                "annotation_id": row["annotation_id"], "slide_id": row["slide_id"],
                "patch_id": row["patch_id"], "reason": "component_label_not_found",
            })
            continue
        x1, y1, x2, y2 = map(int, json.loads(row["bbox_patch"]))
        selected_mask = (labels == component_label).astype(np.uint8) * 255
        mask_source = selected_mask if mask_mode == "single_component" else (mask > 127).astype(np.uint8) * 255
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if crop_mode == "square_padded":
            crop, bounds, source_side = square_padded_crop(
                image_rgb, (x1, y1, x2, y2), crop_size, margin_ratio, padding_mode
            )
            crop_mask, _, _ = square_padded_crop(
                mask_source, (x1, y1, x2, y2), crop_size, margin_ratio, "background", True
            )
        else:
            crop, bounds = extract_glomerulus_crop(
                image_rgb, [x1, y1, x2, y2], margin_ratio=margin_ratio, target_size=crop_size
            )
            left, top, right, bottom = bounds
            patch_mask = mask_source[top:bottom, left:right]
            crop_mask = cv2.resize(
                patch_mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST
            ) if patch_mask.size else np.zeros((crop_size, crop_size), dtype=np.uint8)
            source_side = float(max(right - left, bottom - top))
        crops.append(crop)
        masks.append(crop_mask)
        downsample = float(row["size_l0"]) / int(row["patch_size"])
        pixel_sizes_um.append(0.2519 * downsample * source_side / crop_size)
        background = crop[crop_mask == 0]
        metadata.append({
            "annotation_id": row["annotation_id"], "patch_id": row["patch_id"],
            "slide_id": row["slide_id"], "split": row["split"],
            "completeness": float(row["completeness"]),
            "border_distance_px": float(row.get("border_distance_px", np.nan)),
            "tissue_ratio": float(row.get("tissue_ratio", np.nan)),
            "background_mean_intensity": float(background.mean()) if len(background) else float(crop.mean()),
            "bbox_l0": row.get("bbox_l0", ""),
            "image_path": str(row.get("image_path", "")),
            "mask_path": str(row.get("mask_path", "")),
        })
    return crops, masks, metadata, pixel_sizes_um, pd.DataFrame(exclusions)


def stratified_sample_rows(
    rows: pd.DataFrame, max_samples: int, seed: int
) -> pd.DataFrame:
    """Proportionally sample rows per slide with deterministic largest remainders."""
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    if max_samples >= len(rows):
        return rows.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    groups = {
        str(slide): list(indices)
        for slide, indices in rows.groupby("slide_id", sort=True).indices.items()
    }
    quotas = {slide: max_samples * len(indices) / len(rows) for slide, indices in groups.items()}
    allocation = {
        slide: min(len(indices), int(quotas[slide])) for slide, indices in groups.items()
    }
    remaining = max_samples - sum(allocation.values())
    for slide in sorted(groups, key=lambda value: (-(quotas[value] % 1), value)):
        if remaining == 0:
            break
        if allocation[slide] < len(groups[slide]):
            allocation[slide] += 1
            remaining -= 1
    chosen = []
    for slide in sorted(groups):
        chosen.extend(rng.choice(groups[slide], size=allocation[slide], replace=False).tolist())
    return rows.iloc[sorted(chosen)].reset_index(drop=True)


