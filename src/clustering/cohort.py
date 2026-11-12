"""Annotation-linked, deterministic cohort construction for clustering."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from src.utils.evaluation import create_artifact_directory, file_identity, write_json
from src.utils.xml_parser import parse_asap_xml


COHORT_SCHEMA_VERSION = 2
COMPLETENESS_CONVENTION = (
    "Rasterize full XML geometry and component on the same patch-pixel grid; "
    "completeness = pixels(full-geometry AND associated-component) / pixels(full-geometry); "
    "window_coverage = pixels(full-geometry inside patch) / pixels(full-geometry)."
)
COHORT_COLUMNS = [
    "annotation_id", "slide_id", "split", "patch_id", "completeness",
    "window_coverage", "component_area_px", "component_label", "bbox_patch",
    "bbox_l0", "center_x_l0", "center_y_l0", "center_distance",
    "border_distance_px", "overlap_fraction", "annotation_component_fraction",
    "x0_l0", "y0_l0", "size_l0", "patch_size", "mask_path", "image_path",
    "tissue_ratio",
]
EXCLUSION_COLUMNS = [
    "record_type", "annotation_id", "slide_id", "patch_id", "component_label",
    "reason", "completeness", "window_coverage", "overlap_fraction", "area", "detail",
]


def build_clustering_cohort(
    manifest_path: str,
    xml_dir: str,
    min_completeness: float = 0.95,
    overlap_threshold: float = 0.5,
    min_component_area: int = 400,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Associate connected mask components to XML polygons and select one best view.

    Candidate views are ranked by completeness, then normalized distance to the
    patch center, then patch ID. The XML index is the order of valid ASAP
    annotations returned by ``parse_asap_xml``.
    """
    if not 0 <= min_completeness <= 1:
        raise ValueError("min_completeness must be between 0 and 1")
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be between 0 and 1")
    manifest = Path(manifest_path).resolve()
    xml_root = Path(xml_dir).resolve()
    df = pd.read_csv(manifest)
    required = {"patch_id", "slide_id", "split", "x0_l0", "y0_l0", "size_l0",
                "patch_size", "mask_path"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")

    annotation_cache: Dict[str, List[Any]] = {}
    candidates: Dict[str, List[Dict[str, Any]]] = {}
    exclusions: List[Dict[str, Any]] = []
    inventory: Dict[str, str] = {}
    annotation_reasons: Dict[str, str] = {}

    for _, row in df.iterrows():
        slide_id = str(row["slide_id"])
        xml_path = xml_root / f"{slide_id}.xml"
        if slide_id not in annotation_cache:
            if not xml_path.is_file():
                annotation_cache[slide_id] = []
                exclusions.append({"slide_id": slide_id, "patch_id": row["patch_id"],
                                   "reason": "missing_xml", "detail": str(xml_path)})
            else:
                annotation_cache[slide_id] = parse_asap_xml(xml_path)
        annotations = annotation_cache[slide_id]
        for annotation_index in range(len(annotations)):
            inventory[f"{slide_id}#{annotation_index}"] = slide_id
        if not annotations:
            continue

        mask_path = manifest.parent / str(row["mask_path"])
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            exclusions.append({"slide_id": slide_id, "patch_id": row["patch_id"],
                               "reason": "missing_file", "detail": str(mask_path)})
            continue

        _, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (mask_gray > 127).astype(np.uint8), connectivity=8
        )
        patch_size = int(row["patch_size"])
        downsample = float(row["size_l0"]) / patch_size
        origin_x, origin_y = float(row["x0_l0"]), float(row["y0_l0"])
        for component_label in range(1, len(stats)):
            area = int(stats[component_label, cv2.CC_STAT_AREA])
            if area < min_component_area:
                exclusions.append({"slide_id": slide_id, "patch_id": row["patch_id"],
                                   "component_label": component_label, "reason": "area_minima",
                                   "area": area})
                continue
            component_mask = labels == component_label
            x, y, w, h, _ = map(int, stats[component_label])
            cx, cy = map(float, centroids[component_label])
            matches: List[Dict[str, Any]] = []
            for annotation_index, annotation in enumerate(annotations):
                projected = _project_geometry(
                    annotation.geometry, origin_x, origin_y, downsample
                )
                if projected.is_empty or projected.area <= 0:
                    continue
                full_mask, base_x, base_y = _rasterize_full_geometry(projected)
                full_area = int(np.count_nonzero(full_mask))
                if full_area == 0:
                    continue
                patch_x, patch_y = -base_x, -base_y
                x_start, y_start = max(0, patch_x), max(0, patch_y)
                x_end = min(full_mask.shape[1], patch_x + mask_gray.shape[1])
                y_end = min(full_mask.shape[0], patch_y + mask_gray.shape[0])
                visible_mask = np.zeros_like(component_mask, dtype=bool)
                if x_end > x_start and y_end > y_start:
                    visible_mask[
                        y_start - patch_y:y_end - patch_y,
                        x_start - patch_x:x_end - patch_x,
                    ] = full_mask[y_start:y_end, x_start:x_end]
                visible_area = int(np.count_nonzero(visible_mask))
                intersection = int(np.count_nonzero(visible_mask & component_mask))
                component_overlap = intersection / float(area)
                annotation_component_fraction = (
                    intersection / float(visible_area) if visible_area else 0.0
                )
                if (
                    component_overlap < overlap_threshold
                    or annotation_component_fraction < overlap_threshold
                ):
                    continue
                completeness = intersection / float(full_area)
                window_coverage = visible_area / float(full_area)
                if not np.isfinite(completeness) or not -1e-8 <= completeness <= 1 + 1e-8:
                    annotation_reasons[f"{slide_id}#{annotation_index}"] = (
                        "invalid_completeness_diagnostic"
                    )
                    exclusions.append({
                        "annotation_id": f"{slide_id}#{annotation_index}",
                        "slide_id": slide_id, "patch_id": str(row["patch_id"]),
                        "component_label": component_label,
                        "reason": "invalid_completeness_diagnostic",
                        "completeness": float(completeness),
                    })
                    continue
                matches.append({
                    "annotation_id": f"{slide_id}#{annotation_index}",
                    "slide_id": slide_id,
                    "split": str(row["split"]),
                    "patch_id": str(row["patch_id"]),
                    "completeness": float(np.clip(completeness, 0.0, 1.0)),
                    "window_coverage": float(np.clip(window_coverage, 0.0, 1.0)),
                    "component_area_px": area,
                    "component_label": component_label,
                    "bbox_patch": [x, y, x + w, y + h],
                    "bbox_l0": [
                        origin_x + x * downsample, origin_y + y * downsample,
                        origin_x + (x + w) * downsample, origin_y + (y + h) * downsample,
                    ],
                    "center_x_l0": origin_x + cx * downsample,
                    "center_y_l0": origin_y + cy * downsample,
                    "center_distance": float(np.hypot(
                        (cx - mask_gray.shape[1] / 2) / max(mask_gray.shape[1], 1),
                        (cy - mask_gray.shape[0] / 2) / max(mask_gray.shape[0], 1),
                    )),
                    "border_distance_px": float(min(cx, cy, mask_gray.shape[1] - cx,
                                                    mask_gray.shape[0] - cy)),
                    "overlap_fraction": float(component_overlap),
                    "annotation_component_fraction": annotation_component_fraction,
                    "x0_l0": origin_x,
                    "y0_l0": origin_y,
                    "size_l0": float(row["size_l0"]),
                    "patch_size": patch_size,
                    "mask_path": str(row["mask_path"]),
                    "image_path": str(row.get("image_path", "")),
                    "tissue_ratio": float(row.get("tissue_ratio", np.nan)),
                })
            if len(matches) > 1:
                for match in matches:
                    annotation_reasons[match["annotation_id"]] = (
                        "ambiguous_component_multiple_annotations"
                    )
                    exclusions.append({
                        "annotation_id": match["annotation_id"], "slide_id": slide_id,
                        "patch_id": str(row["patch_id"]),
                        "component_label": component_label,
                        "reason": "ambiguous_component_multiple_annotations",
                        "overlap_fraction": match["overlap_fraction"],
                    })
                continue
            if not matches:
                exclusions.append({"slide_id": slide_id, "patch_id": row["patch_id"],
                                   "component_label": component_label,
                                   "reason": "mancata_associazione_xml", "area": area})
                continue
            match = matches[0]
            candidates.setdefault(match["annotation_id"], []).append(match)

    selected: List[Dict[str, Any]] = []
    for annotation_id, views in sorted(candidates.items()):
        best = min(views, key=lambda item: (
            -item["completeness"], -item["annotation_component_fraction"],
            -item["overlap_fraction"], -item["window_coverage"],
            item["center_distance"], item["patch_id"],
            item["component_label"]
        ))
        for view in views:
            if view is not best:
                exclusions.append({
                    "annotation_id": annotation_id, "slide_id": view["slide_id"],
                    "patch_id": view["patch_id"], "component_label": view["component_label"],
                    "reason": "duplicato_best_view", "completeness": view["completeness"],
                })
        if best["completeness"] < min_completeness:
            annotation_reasons[annotation_id] = "completezza"
            exclusions.append({
                "annotation_id": annotation_id, "slide_id": best["slide_id"],
                "patch_id": best["patch_id"], "component_label": best["component_label"],
                "reason": "completezza", "completeness": best["completeness"],
                "window_coverage": best["window_coverage"],
            })
            continue
        best["bbox_l0"] = json.dumps(best["bbox_l0"])
        best["bbox_patch"] = json.dumps(best["bbox_patch"])
        selected.append(best)
    selected_ids = {str(row["annotation_id"]) for row in selected}
    for annotation_id, slide_id in sorted(inventory.items()):
        if annotation_id in selected_ids:
            continue
        if not any(str(row.get("annotation_id", "")) == annotation_id and
                   row.get("reason") == "annotation_exclusion" for row in exclusions):
            reason = annotation_reasons.get(
                annotation_id,
                "no_candidate_view" if annotation_id not in candidates else "no_eligible_view",
            )
            reason_detail = next(
                (
                    {
                        key: row[key]
                        for key in ("completeness", "window_coverage")
                        if key in row
                    }
                    for row in reversed(exclusions)
                    if row.get("annotation_id") == annotation_id
                    and row.get("reason") == reason
                ),
                {},
            )
            exclusions.append({
                "record_type": "annotation_exclusion",
                "annotation_id": annotation_id,
                "slide_id": slide_id,
                "reason": reason,
                **reason_detail,
            })
    return (
        pd.DataFrame(selected, columns=COHORT_COLUMNS),
        pd.DataFrame(exclusions).reindex(columns=EXCLUSION_COLUMNS),
    )


def write_cohort_run(
    cohort: pd.DataFrame,
    exclusions: pd.DataFrame,
    manifest_path: str,
    xml_dir: str,
    output_root: str = "runs/clustering",
    name: str = "cohort",
    min_completeness: float = 0.95,
    overlap_threshold: float = 0.5,
) -> Path:
    """Write a named immutable cohort directory and its input identities."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out_dir = create_artifact_directory(output_root, name)
    cohort.reindex(columns=COHORT_COLUMNS).to_csv(out_dir / "cohort.csv", index=False)
    exclusions.reindex(columns=EXCLUSION_COLUMNS).to_csv(
        out_dir / "exclusions.csv", index=False
    )
    manifest_rows = pd.read_csv(manifest_path)
    # A scoped manifest must never cause unrelated XMLs (including test) to be opened.
    xml_files = [Path(xml_dir) / f"{slide_id}.xml"
                 for slide_id in sorted(set(manifest_rows["slide_id"].astype(str)))]
    all_mask_paths = sorted({
        Path(manifest_path).resolve().parent / str(value)
        for value in manifest_rows["mask_path"].tolist()
    })
    payload = {
        "cohort_schema_version": COHORT_SCHEMA_VERSION,
        "completeness_convention": COMPLETENESS_CONVENTION,
        "created_at": stamp,
        "n_annotations": int(len(cohort)),
        "min_completeness": min_completeness,
        "overlap_threshold": overlap_threshold,
        "selection_rule": (
            "max full-annotation component completeness, then annotation/component "
            "overlap, then window coverage, then nearest patch center, lexical patch_id, "
            "and component label"
        ),
        "annotation_inventory_count": int(
            len(cohort) + (
                exclusions.loc[
                    exclusions.get("record_type", pd.Series(index=exclusions.index, dtype=object))
                    == "annotation_exclusion", "annotation_id"
                ].nunique()
                if not exclusions.empty and "annotation_id" in exclusions else 0
            )
        ),
        "selected_split_counts": (
            cohort["split"].astype(str).value_counts().sort_index().to_dict()
            if not cohort.empty else {}
        ),
        "annotation_exclusions_by_reason": (
            exclusions.loc[
                exclusions.get("record_type", pd.Series(index=exclusions.index, dtype=object))
                == "annotation_exclusion", "reason"
            ].value_counts().sort_index().to_dict()
            if not exclusions.empty and "reason" in exclusions else {}
        ),
        "source_files": {
            str(path): file_identity(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parent / "training.py",
                Path(__file__).resolve().parents[1] / "utils/xml_parser.py",
            )
        },
        "inputs": {
            "manifest": file_identity(manifest_path),
            "xml": [file_identity(path) for path in xml_files],
            "masks": [file_identity(path) for path in all_mask_paths if path.is_file()],
            "missing_masks": [str(path) for path in all_mask_paths if not path.is_file()],
        },
        "cohort_csv": {
            key: value for key, value in file_identity(out_dir / "cohort.csv").items()
            if key != "path"
        },
        "exclusions_csv": {
            key: value for key, value in file_identity(out_dir / "exclusions.csv").items()
            if key != "path"
        },
    }
    stable_payload = {key: value for key, value in payload.items() if key != "created_at"}
    payload["cohort_hash"] = _json_hash(stable_payload)
    write_json(out_dir / "run.json", payload)
    return out_dir


def audit_cohort_directory(
    cohort_path: str, manifest_path: str, xml_dir: str
) -> Dict[str, Any]:
    """Verify schema, source hashes and one-row-per-XML-annotation accounting."""
    csv_path = Path(cohort_path).resolve()
    run_path = csv_path.parent / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"Cohort run metadata is missing: {run_path}")
    metadata = json.loads(run_path.read_text(encoding="utf-8"))
    if metadata.get("cohort_schema_version") != COHORT_SCHEMA_VERSION:
        raise ValueError("Cohort audit requires schema version 2")
    exclusions_path = csv_path.parent / "exclusions.csv"
    cohort = pd.read_csv(csv_path)
    exclusions = pd.read_csv(exclusions_path)
    if file_identity(csv_path)["sha256"] != metadata["cohort_csv"]["sha256"]:
        raise ValueError("cohort.csv hash differs from run.json")
    if file_identity(exclusions_path)["sha256"] != metadata["exclusions_csv"]["sha256"]:
        raise ValueError("exclusions.csv hash differs from run.json")
    manifest_identity = metadata["inputs"]["manifest"]
    if file_identity(manifest_path)["sha256"] != manifest_identity["sha256"]:
        raise ValueError("Manifest hash differs from frozen cohort inputs")
    manifest = pd.read_csv(manifest_path)
    expected_xml = {
        entry["path"]: entry["sha256"] for entry in metadata["inputs"].get("xml", [])
    }
    expected_masks = {
        entry["path"]: entry["sha256"] for entry in metadata["inputs"].get("masks", [])
    }
    xml_root = Path(xml_dir).resolve()
    inventory_ids = set()
    missing_xml = []
    for slide_id in sorted(set(manifest["slide_id"].astype(str))):
        xml_path = xml_root / f"{slide_id}.xml"
        if not xml_path.is_file():
            missing_xml.append(slide_id)
            continue
        expected_hash = expected_xml.get(str(xml_path))
        if expected_hash is None or file_identity(xml_path)["sha256"] != expected_hash:
            raise ValueError(f"XML input hash is missing or changed: {xml_path}")
        inventory_ids.update(
            f"{slide_id}#{index}" for index, _ in enumerate(parse_asap_xml(xml_path))
        )
    missing_masks = []
    for mask_path, expected_hash in expected_masks.items():
        path = Path(mask_path)
        if not path.is_file():
            missing_masks.append(str(path))
        elif file_identity(path)["sha256"] != expected_hash:
            raise ValueError(f"Mask input hash changed: {path}")
    selected_ids = cohort["annotation_id"].astype(str).tolist()
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("cohort.csv contains duplicate annotation_id values")
    annotation_exclusions = exclusions[
        exclusions.get("record_type", pd.Series(index=exclusions.index, dtype=object))
        == "annotation_exclusion"
    ]
    excluded_ids = annotation_exclusions["annotation_id"].dropna().astype(str).tolist()
    if len(set(excluded_ids)) != len(excluded_ids):
        raise ValueError("An annotation has multiple terminal exclusion records")
    if set(selected_ids) & set(excluded_ids):
        raise ValueError("An annotation is both selected and terminally excluded")
    accounted_ids = set(selected_ids) | set(excluded_ids)
    if accounted_ids != inventory_ids:
        missing = sorted(inventory_ids - accounted_ids)
        extra = sorted(accounted_ids - inventory_ids)
        raise ValueError(
            f"XML inventory accounting mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    completeness = cohort["completeness"].astype(float)
    if not np.isfinite(completeness).all() or (
        (completeness < -1e-8) | (completeness > 1 + 1e-8)
    ).any():
        raise ValueError("Selected cohort has non-finite or out-of-range completeness")
    if (completeness < float(metadata["min_completeness"])).any():
        raise ValueError("Selected cohort contains a row below the completeness threshold")
    return {
        "status": "verified" if not missing_xml and not missing_masks else "verified_with_missing_sources",
        "cohort_schema_version": COHORT_SCHEMA_VERSION,
        "cohort_hash": metadata["cohort_hash"],
        "xml_annotation_count": len(inventory_ids),
        "selected_annotation_count": len(selected_ids),
        "excluded_annotation_count": len(excluded_ids),
        "selected_by_split": cohort["split"].astype(str).value_counts().sort_index().to_dict(),
        "excluded_by_reason": annotation_exclusions["reason"].value_counts().sort_index().to_dict(),
        "missing_xml_slides": missing_xml,
        "missing_masks": missing_masks,
        "completeness_min": float(completeness.min()) if len(completeness) else None,
        "completeness_max": float(completeness.max()) if len(completeness) else None,
    }


def _project_geometry(geometry: Any, x0: float, y0: float, downsample: float) -> Any:
    from shapely.ops import transform
    return transform(
        lambda x, y: ((np.asarray(x) - x0) / downsample,
                      (np.asarray(y) - y0) / downsample),
        geometry,
    )


def _rasterize_full_geometry(geometry: Any) -> Tuple[np.ndarray, int, int]:
    """Rasterize full geometry on an integer grid; return mask and grid origin.

    OpenCV rounds projected vertices to the nearest pixel center. The same
    convention is used for the full polygon and its patch-visible component.
    """
    min_x, min_y, max_x, max_y = geometry.bounds
    base_x, base_y = int(np.floor(min_x)) - 1, int(np.floor(min_y)) - 1
    end_x, end_y = int(np.ceil(max_x)) + 2, int(np.ceil(max_y)) + 2
    mask = np.zeros((max(1, end_y - base_y), max(1, end_x - base_x)), dtype=np.uint8)
    geoms = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    for polygon in geoms:
        if polygon.geom_type != "Polygon":
            continue
        exterior = np.rint(
            np.asarray(polygon.exterior.coords) - np.array([base_x, base_y])
        ).astype(np.int32)
        if len(exterior) >= 3:
            cv2.fillPoly(mask, [exterior], 1)
        for ring in polygon.interiors:
            hole = np.rint(
                np.asarray(ring.coords) - np.array([base_x, base_y])
            ).astype(np.int32)
            if len(hole) >= 3:
                cv2.fillPoly(mask, [hole], 0)
    return mask.astype(bool), base_x, base_y


def _json_hash(value: Dict[str, Any]) -> str:
    import hashlib
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()
