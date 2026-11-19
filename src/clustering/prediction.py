"""Apply frozen clustering models to segmented WSI detections, never to GT class labels."""

import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import joblib
import numpy as np
import pandas as pd
import openslide
from PIL import Image
from shapely.geometry import Polygon

from src.clustering.crops import square_padded_crop
from src.clustering.feature_extractor import extract_morphological_dataset
from src.clustering.labels import peripheral_clear_fraction, LABEL_LIMITATION
from src.clustering.metrics import _metric_result
from src.clustering.reporting import save_gallery, write_report
from src.clustering.splits import TRAINING_PROTOCOL, load_split_config, validate_test_prediction
from src.utils.evaluation import file_identity, write_json


def load_predictions(path, slide_path=None, allow_legacy=False):
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if data.get("stage") != "wsi_predictions" or not data.get("cascade_mode"):
            raise ValueError("Clustering requires post-NMS YOLO + U-Net predictions")
        if slide_path:
            data["svs_path"] = str(slide_path)
        return data
    if not allow_legacy or slide_path is None:
        raise ValueError("Legacy predicted XML needs --slide and --allow-legacy-unet-xml; prefer *_predictions.json")
    detections = []
    for i, annotation in enumerate(ET.parse(path).getroot().findall(".//Annotation")):
        coordinates = sorted(annotation.findall(".//Coordinate"), key=lambda n: int(n.get("Order", "0")))
        points = [[float(c.get("X")), float(c.get("Y"))] for c in coordinates]
        usable = annotation.get("Type") == "Polygon" and len(points) > 4
        detections.append({
            "prediction_id": f"{Path(slide_path).stem}#{i}",
            "polygon_l0": points,
            "mask_source": "unet" if usable else "bbox_or_unverified_polygon",
        })
    return {
        "slide_id": Path(slide_path).stem,
        "svs_path": str(slide_path),
        "cascade_mode": True,
        "detections": detections,
        "legacy_provenance": "User-declared historical YOLO+U-Net export; only >4-vertex polygons accepted",
    }


def load_models(root, k_values):
    models = {}
    identities = {}
    for k in k_values:
        source = Path(root) / f"clustering_k{k}"
        complete = json.loads((source / "complete.json").read_text())
        if complete.get("status") != "completed":
            raise ValueError(f"Incomplete model: {source}")
        if complete.get("training_protocol") != TRAINING_PROTOCOL:
            raise ValueError(
                f"Legacy model {source}: rerun train_clustering.py --overwrite --rebuild-cohort for the train/validation-only protocol"
            )
        path = source / "pipeline.joblib"
        identity = file_identity(path)
        if identity["sha256"] != complete["outputs"]["pipeline.joblib"]:
            raise ValueError(f"Changed model: {path}")
        model = joblib.load(path)
        if getattr(model, "training_protocol", None) != TRAINING_PROTOCOL:
            raise ValueError(f"Model lacks train/validation-only provenance: {path}")
        if model.clusterer.n_clusters != k:
            raise ValueError(f"Unexpected cluster count in {path}: {model.clusterer.n_clusters} != {k}")
        if model.feature_type == "hybrid" and (
            model.clusterer.pca is None or model.clusterer.pca.n_components_ != 20
        ):
            raise ValueError("Unexpected hybrid PCA20 clustering model")
        models[k] = model
        identities[f"clustering_k{k}"] = identity

    first = models[k_values[0]]
    for model in models.values():
        if model.feature_type != first.feature_type:
            raise ValueError("Models use different feature_type representations")
        if model.feature_type == "stain_invariant_morph":
            if getattr(model, "selected_features", None) != getattr(first, "selected_features", None):
                raise ValueError("Models use different selected stain-invariant features")
            for attr in ("mean_", "scale_"):
                if not np.array_equal(getattr(model.clusterer.scaler, attr), getattr(first.clusterer.scaler, attr)):
                    raise ValueError("Models have different fitted StandardScaler parameters")
        elif model.feature_type == "hybrid":
            if model.extractor.weights_sha256() != first.extractor.weights_sha256():
                raise ValueError("Different backbones")
            for name in ("deep_scaler", "morph_scaler"):
                for attr in ("mean_", "scale_"):
                    if not np.array_equal(
                        getattr(getattr(model.block_transformer, name), attr),
                        getattr(getattr(first.block_transformer, name), attr),
                    ):
                        raise ValueError("Models have different feature transforms")
    return models, identities


def select_test_predictions(inference_dir, dataset_config="configs/dataset_config.yaml"):
    """Require both test WSI exports and ignore exports for other splits."""
    config = load_split_config(dataset_config)
    paths = [Path(inference_dir) / f"{slide}_predictions.json" for slide in config["splits"]["test"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Missing test WSI exports; run WSI inference for both 015/017: {missing}")
    for path in paths:
        validate_test_prediction(load_predictions(path), dataset_config)
    return paths


def extract_wsi_crop(slide, polygon, target_size=512, margin_ratio=0.25, target_mpp=0.5038):
    points = np.asarray(polygon, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all() or len(points) < 3:
        raise ValueError("invalid_polygon")
    geometry = Polygon(points)
    if not geometry.is_valid or geometry.area <= 0:
        raise ValueError("invalid_polygon")
    try:
        mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
    except (KeyError, ValueError):
        raise ValueError("missing_physical_resolution")
    if not (mpp_x > 0 and mpp_y > 0 and np.isclose(mpp_x, mpp_y, rtol=0.01)):
        raise ValueError("unsupported_anisotropic_or_invalid_resolution")

    downsample = target_mpp / mpp_x
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    side = float(max(max_xy - min_xy) * (1 + margin_ratio))
    center = (min_xy + max_xy) / 2
    left, top = np.maximum(0, np.floor(center - side / 2)).astype(int)
    right, bottom = np.minimum(slide.dimensions, np.ceil(center + side / 2)).astype(int)
    if right <= left or bottom <= top:
        raise ValueError("polygon_outside_slide")

    level = slide.get_best_level_for_downsample(downsample)
    level_downsample = float(slide.level_downsamples[level])
    size = (max(1, int(np.ceil((right - left) / level_downsample))), max(1, int(np.ceil((bottom - top) / level_downsample))))
    rgba = slide.read_region((int(left), int(top)), level, size).convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    rgb = np.asarray(background.convert("RGB"))
    width = max(1, int(round((right - left) / downsample)))
    height = max(1, int(round((bottom - top) / downsample)))
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)

    local = (points - np.asarray([left, top])) / downsample
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.round(local).astype(np.int32)], 255)
    if not np.any(mask):
        raise ValueError("empty_segmented_mask")
    x1, y1 = np.floor(local.min(axis=0)).astype(int)
    x2, y2 = np.ceil(local.max(axis=0)).astype(int)
    box = (int(x1), int(y1), max(int(x1) + 1, int(x2)), max(int(y1) + 1, int(y2)))
    crop, _, source_side = square_padded_crop(rgb, box, target_size, margin_ratio, "reflect")
    crop_mask, _, _ = square_padded_crop(mask, box, target_size, margin_ratio, "background", True)
    if not np.any(crop_mask):
        raise ValueError("empty_resized_mask")
    return crop, crop_mask, target_mpp * source_side / target_size


def export_cluster_xml(path, records, labels):
    root = ET.Element("ASAP_Annotations")
    annotations = ET.SubElement(root, "Annotations")
    palette = {"A": "#1b9e77", "B": "#66a61e", "C": "#e6ab02", "D": "#e7298a", "E": "#7570b3", "F": "#d95f02"}
    for record, label in zip(records, labels):
        node = ET.SubElement(
            annotations,
            "Annotation",
            Name=record["prediction_id"],
            Type="Polygon",
            PartOfGroup=f"Experimental_{label}",
            Color=palette[label],
        )
        coords = ET.SubElement(node, "Coordinates")
        for i, (x, y) in enumerate(record["polygon_l0"]):
            ET.SubElement(coords, "Coordinate", Order=str(i), X=str(x), Y=str(y))
    groups = ET.SubElement(root, "AnnotationGroups")
    for label in sorted(set(labels)):
        ET.SubElement(groups, "Group", Name=f"Experimental_{label}", PartOfGroup="None", Color=palette[label])
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def run_prediction(
    predictions,
    models_root="runs/clustering",
    output_root="runs/clustering_predictions",
    k_values=(2, 4, 6),
    slide_path=None,
    allow_legacy=False,
    dataset_config="configs/dataset_config.yaml",
):
    document = load_predictions(predictions, slide_path, allow_legacy)
    validate_test_prediction(document, dataset_config)
    k_values = list(k_values)
    models, identities = load_models(models_root, k_values)
    source_slide = Path(document["svs_path"])

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / document["slide_id"]
    if output.is_symlink():
        raise ValueError(f"Refusing to overwrite symlink prediction output: {output}")
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Prediction output path exists and is not a directory: {output}")
        shutil.rmtree(output)
    output.mkdir()
    write_json(output / "run.json", {"status": "in_progress", "stage": "clustering_prediction"})

    crops = []
    masks = []
    pixels = []
    accepted = []
    rejected = []
    with openslide.OpenSlide(str(source_slide)) as slide:
        for record in document["detections"]:
            if record.get("mask_source") != "unet":
                rejected.append({"prediction_id": record["prediction_id"], "reason": "requires_unet_mask"})
                continue
            try:
                c, m, p = extract_wsi_crop(slide, record["polygon_l0"])
                crops.append(c)
                masks.append(m)
                pixels.append(p)
                accepted.append(record)
            except ValueError as error:
                rejected.append({"prediction_id": record["prediction_id"], "reason": str(error)})

    pd.DataFrame(rejected, columns=["prediction_id", "reason"]).to_csv(output / "excluded_predictions.csv", index=False)
    records = []
    first = models[k_values[0]]
    if crops:
        morph, names = extract_morphological_dataset(crops, masks, "native", pixels)
        features = first._features_for_new(crops, masks, pixels)
        if names != first.common_space["feature_names"]:
            raise ValueError("Morphology feature order changed")
        common = (morph - first.common_space["mean"]) / first.common_space["scale"]
        rim = np.asarray([peripheral_clear_fraction(c, m) for c, m in zip(crops, masks)])

    for k, model in models.items():
        dest = output / f"clustering_k{k}"
        dest.mkdir()
        if crops:
            labels = model.clusterer.predict(features)
            letters = model.class_labels(labels)
            frame = pd.DataFrame({
                "annotation_id": [r["prediction_id"] for r in accepted],
                "slide_id": document["slide_id"],
                "split": "test",
                "cluster": labels,
                "class_label": letters,
                "distance_to_center": model.clusterer.distances_to_centers(features),
                "peripheral_clear_fraction": rim,
            })
            for j, name in enumerate(names):
                frame[name] = morph[:, j]
            internal = _metric_result(common, labels)
            save_gallery(dest / "gallery.png", crops, frame, k, model.label_mapping)
            export_cluster_xml(dest / "clusters.xml", accepted, letters)
        else:
            frame = pd.DataFrame(
                columns=["annotation_id", "slide_id", "split", "cluster", "class_label", "distance_to_center", "peripheral_clear_fraction"]
            )
            internal = _metric_result(np.empty((0, 10)), np.asarray([], int))
            save_gallery(dest / "gallery.png", crops, frame, k, model.label_mapping)
        frame.to_csv(dest / "assignments.csv", index=False)
        counts = {letter: int(np.count_nonzero(frame.class_label == letter)) for letter in sorted(model.label_mapping.values())}
        metrics = {
            "split": "test",
            "slide_id": document["slide_id"],
            "n_detected": len(document["detections"]),
            "n_assigned": len(frame),
            "n_excluded": len(rejected),
            "counts": counts,
            "internal_common": internal,
            "distance_median": float(frame.distance_to_center.median()) if len(frame) else None,
            "distance_q95": float(frame.distance_to_center.quantile(0.95)) if len(frame) else None,
            "class_label_mapping": model.label_mapping,
            "label_limitation": LABEL_LIMITATION,
            "clinical_accuracy": None,
            "clinical_accuracy_reason": "No clinical class labels exist in the position/contour XML",
        }
        write_json(dest / "metrics.json", metrics)
        records.append({
            "model": f"clustering_k{k}",
            "split": document["slide_id"],
            "n_samples": len(frame),
            "counts": counts,
            "silhouette": internal["silhouette"],
            "distance_median": metrics["distance_median"],
        })

    write_report(
        output,
        records,
        "Clustering delle predizioni YOLO + U-Net",
        f"Glomeruli assegnati: {len(accepted)}/{len(document['detections'])}; esclusioni: {len(rejected)}. Nessun fit sulle predizioni WSI.",
    )
    write_json(
        output / "run.json",
        {
            "status": "completed",
            "stage": "clustering_prediction",
            "split": "test",
            "slide_id": document["slide_id"],
            "models": identities,
            "source_predictions": file_identity(predictions),
            "source_slide": file_identity(source_slide),
            "detection_segmentation_metrics": document.get("detection_segmentation_metrics"),
            "legacy_provenance": document.get("legacy_provenance"),
            "fit_on_predictions": False,
            "outputs": {
                str(p.relative_to(output)): file_identity(p)["sha256"]
                for p in output.rglob("*")
                if p.is_file() and p.name != "run.json"
            },
        },
    )
    return output
