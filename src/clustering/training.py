"""Stain-invariant morphological clustering (or baseline PCA20) fitting on train and validation assignment."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
import yaml

from src.clustering.cohort import build_clustering_cohort, write_cohort_run, audit_cohort_directory
from src.clustering.common_space import load_or_create_common_space
from src.clustering.data import load_cohort_samples
from src.clustering.embedding_cache import _cached_embeddings
from src.clustering.feature_extractor import (
    FeatureBlockTransformer,
    GlomeruliFeatureExtractor,
    STAIN_INVARIANT_MORPH_FEATURES,
    StainInvariantMorphTransformer,
    extract_morphological_dataset,
    select_morphological_columns,
)
from src.clustering.manifold_clustering import GlomeruliClusterer
from src.clustering.pipeline import ClusterAssignmentPipeline
from src.clustering.labels import peripheral_clear_fraction, fit_label_mapping, RIM_PARAMETERS, LABEL_LIMITATION
from src.clustering.evaluation import evaluate_run
from src.clustering.reporting import evaluate_partition, save_gallery, save_projection, write_report, criteria
from src.clustering.comparison import compare_runs
from src.clustering.splits import TRAINING_SPLITS, TEST_SLIDES, TRAINING_PROTOCOL, load_split_config, validate_dataset_splits
from src.utils.evaluation import EvaluationRun, file_identity, write_json, create_artifact_directory, _clustering_source_paths


def archive_existing(path):
    path = Path(path)
    if path.exists():
        target = create_artifact_directory("to_cleanup/replaced", path.name) / "result"
        shutil.move(str(path), str(target))
        return target


def validate_splits(cohort, dataset_config):
    validate_dataset_splits(dataset_config)
    if cohort.annotation_id.duplicated().any():
        raise ValueError("Duplicate annotation IDs")
    if cohort.groupby("slide_id").split.nunique().gt(1).any():
        raise ValueError("Slide leakage between splits")
    if not set(cohort.split.astype(str)).issubset(TRAINING_SPLITS) or set(cohort.slide_id.astype(str)) & TEST_SLIDES:
        raise ValueError("train_clustering accepts train/validation only; test slides 015/017 are forbidden")
    for split in TRAINING_SPLITS:
        actual = set(cohort.loc[cohort.split == split, "slide_id"])
        expected = set(dataset_config["splits"][split])
        if actual != expected:
            raise ValueError(f"Split {split} differs from prepare_dataset: {actual} != {expected}")


def prepare_training_manifest(config, rebuild=False):
    """Filter shared CSV metadata before any image, mask or XML is opened/hashed."""
    data = config["data"]
    source = Path(data.get("source_manifest_csv", data["manifest_csv"])).resolve()
    manifest = pd.read_csv(source)
    selected = manifest.loc[manifest["split"].isin(TRAINING_SPLITS)].copy()
    validate_splits(selected.assign(annotation_id=selected.patch_id), load_split_config(data["dataset_config"]))
    for column in ("image_path", "mask_path", "label_path"):
        if column in selected:
            selected[column] = selected[column].map(
                lambda value: str((source.parent / str(value)).resolve()) if pd.notna(value) else value
            )
    snapshot = Path(data["cohort_path"]).parent.parent / "train_val_manifest.csv"
    if snapshot.resolve() == source:
        raise ValueError("The scoped manifest must be separate from the prepare_dataset manifest")
    content = selected.to_csv(index=False)
    if snapshot.exists():
        if rebuild:
            archive_existing(snapshot)
        elif snapshot.read_text() != content:
            raise ValueError("Train/validation manifest changed; use --overwrite --rebuild-cohort")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.write_text(content)
    data["source_manifest_csv"] = str(source)
    data["manifest_csv"] = str(snapshot)
    data["splits"] = list(TRAINING_SPLITS)
    return selected


def ensure_cohort(config, rebuild=False):
    data = config["data"]
    path = Path(data["cohort_path"])
    manifest = prepare_training_manifest(config, rebuild)
    if rebuild:
        archive_existing(path.parent)
        archive_existing(config["output"]["resampling"])
    if not path.exists():
        cohort, exclusions = build_clustering_cohort(
            data["manifest_csv"],
            data["xml_dir"],
            data.get("min_completeness", 0.95),
            data.get("overlap_threshold", 0.5),
            data.get("min_component_area", 400),
        )
        result = write_cohort_run(
            cohort,
            exclusions,
            data["manifest_csv"],
            data["xml_dir"],
            str(path.parent.parent),
            path.parent.name,
            data.get("min_completeness", 0.95),
            data.get("overlap_threshold", 0.5),
        )
        if result != path.parent:
            raise ValueError("Unexpected cohort destination")
    cohort = pd.read_csv(path)
    validate_splits(cohort, load_split_config(data["dataset_config"]))
    metadata = json.loads((path.parent / "run.json").read_text())
    allowed_xml = {(Path(data["xml_dir"]) / f"{slide}.xml").resolve() for slide in manifest.slide_id.unique()}
    allowed_masks = {Path(value).resolve() for value in manifest.mask_path}
    for kind, allowed in (("xml", allowed_xml), ("masks", allowed_masks)):
        if any(Path(entry["path"]).resolve() not in allowed for entry in metadata["inputs"].get(kind, [])):
            raise ValueError("Cohort provenance includes inputs outside train/validation; rebuild the cohort")
    audit = audit_cohort_directory(str(path), data["manifest_csv"], data["xml_dir"])
    if audit["status"] != "verified":
        raise ValueError("Frozen cohort has missing sources")
    write_json(path.parent / "audit.json", audit)
    return cohort, metadata


def workflow_signature(config):
    document = {"config": config, "sources": {str(p): file_identity(p)["sha256"] for p in _clustering_source_paths()}}
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def run_training(
    config_path="configs/clustering_config.yaml",
    device="cpu",
    k_values=None,
    overwrite=False,
    rebuild_cohort=False,
):
    config = yaml.safe_load(Path(config_path).read_text())
    fcfg = config["features"]
    feature_type = str(fcfg.get("feature_type", "stain_invariant_morph"))
    block_weighting = str(fcfg.get("block_weighting", "none"))
    manifold_cfg = config.get("manifold", {})
    manifold_method = str(manifold_cfg.get("method") or ("none" if feature_type == "stain_invariant_morph" else "pca"))
    pca_variance = manifold_cfg.get("pca_variance")
    input_scaling = str(config["clustering"].get("input_scaling", "standard" if feature_type == "stain_invariant_morph" else "none"))

    if feature_type == "stain_invariant_morph":
        if block_weighting != "none" or manifold_method != "none" or config["clustering"]["algorithm"] != "kmeans":
            raise ValueError("stain_invariant_morph requires block_weighting='none', manifold.method='none', and kmeans")
    elif feature_type == "hybrid":
        if block_weighting != "equal" or pca_variance != 20 or config["clustering"]["algorithm"] != "kmeans":
            raise ValueError("Hybrid baseline requires block_weighting='equal', PCA20, and kmeans")
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    if config["clustering"]["fit_splits"] != ["train"] or config["clustering"]["assign_splits"] != ["val"]:
        raise ValueError("Fit train only; assign validation only. Test is reserved for predict_clustering")
    if config.get("labels", {}).get("meaning") != "experimental_morphology_not_clinical_necrosis":
        raise ValueError("Class labels must remain explicitly experimental")

    k_values = list(dict.fromkeys(k_values or config["clustering"]["k_values"]))
    if not k_values or any(k not in (2, 4, 6) for k in k_values):
        raise ValueError("K must be 2, 4 or 6")
    if rebuild_cohort and not overwrite:
        raise ValueError("--rebuild-cohort requires --overwrite")

    cohort, cohort_metadata = ensure_cohort(config, rebuild_cohort)
    validate_splits(cohort, load_split_config(config["data"]["dataset_config"]))
    root = Path(config["output"]["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    signature = workflow_signature(config)
    targets = {k: root / f"clustering_k{k}" for k in k_values}
    completed = {}

    for k, path in targets.items():
        if overwrite:
            archive_existing(path)
        if path.exists():
            marker = path / "complete.json"
            if not marker.exists():
                raise ValueError(f"Incomplete run {path}; use --overwrite to archive and rebuild")
            meta = json.loads(marker.read_text())
            if meta.get("training_protocol") != TRAINING_PROTOCOL:
                raise ValueError(f"Legacy train/test run {path}; use --overwrite to rebuild with train/validation only")
            if meta["signature"] != signature:
                raise ValueError(f"Changed code/config for {path}; use --overwrite")
            for relative, digest in meta["outputs"].items():
                if file_identity(path / relative)["sha256"] != digest:
                    raise ValueError(f"Changed output: {path / relative}")
            for source, digest in meta["inputs"].items():
                if file_identity(source)["sha256"] != digest:
                    raise ValueError(f"Changed input: {source}")
            completed[k] = json.loads((path / "summary.json").read_text())

    todo = [k for k in k_values if k not in completed]
    if todo:
        data = config["data"]
        crops, masks, metadata, pixels, exclusions = load_cohort_samples(
            data["cohort_path"],
            data["manifest_csv"],
            list(TRAINING_SPLITS),
            data["crop_size"],
            data["crop_mode"],
            data["mask_mode"],
            data["margin_ratio"],
            data["padding_mode"],
            xml_dir=data["xml_dir"],
        )
        if len(crops) != len(cohort) or not exclusions.empty:
            raise ValueError("Unexpected cohort exclusions")
        frame = pd.DataFrame(metadata)
        split = frame.split.to_numpy(str)
        validate_splits(frame, load_split_config(data["dataset_config"]))
        if set(frame.annotation_id) != set(cohort.annotation_id):
            raise ValueError("Loaded cohort IDs differ")
        train = np.asarray(sorted(np.flatnonzero(split == "train"), key=lambda i: str(frame.iloc[i].annotation_id)))

        morph, morph_names = extract_morphological_dataset(crops, masks, "native", pixels)
        if feature_type == "stain_invariant_morph":
            extractor = None
            deep = np.empty((len(crops), 0), dtype=np.float32)
            selected_features = list(fcfg.get("selected_features") or STAIN_INVARIANT_MORPH_FEATURES)
            selected_raw, selected_names = select_morphological_columns(morph, morph_names, selected_features)
            transformer = StainInvariantMorphTransformer(selected_features=selected_names)
            transformer.fit_transform(selected_raw[train])
            features = transformer.transform(selected_raw)
            backbone_hash = None
        else:
            extractor = GlomeruliFeatureExtractor(fcfg["deep_backbone"], True, device, fcfg["batch_size"], True)
            deep = _cached_embeddings(crops, metadata, extractor, fcfg, data, data["manifest_csv"], data["cohort_path"])
            selected_names = list(morph_names)
            transformer = FeatureBlockTransformer("equal")
            transformer.fit_transform(deep[train], morph[train])
            features = transformer.transform(deep, morph)
            backbone_hash = extractor.weights_sha256()

        common, common_hash = load_or_create_common_space(
            data["cohort_path"],
            data["manifest_csv"],
            data["crop_size"],
            data["margin_ratio"],
            data["padding_mode"],
            fcfg["common_space_cache"],
            xml_dir=data["xml_dir"],
        )
        if not set(common["split"].astype(str)).issubset(TRAINING_SPLITS) or set(common["slide_id"].astype(str)) & TEST_SLIDES:
            raise ValueError("Common-space cache contains test samples; select a train/validation cache")

        lookup = {str(value): i for i, value in enumerate(common["annotation_id"])}
        common_indices = np.asarray([lookup[str(x)] for x in frame.annotation_id])
        native = common["raw_features"][common_indices]
        common_values = common["standardized_features"][common_indices]
        names = common["feature_names"].astype(str).tolist()
        for j, name in enumerate(names):
            frame[name] = native[:, j]

        rim_options = {key: config["labels"][key] for key in RIM_PARAMETERS}
        scores = np.asarray([peripheral_clear_fraction(c, m, **rim_options) for c, m in zip(crops, masks)])
        frame["peripheral_clear_fraction"] = scores

        for k in todo:
            cfg = deepcopy(config)
            cfg["clustering"]["n_clusters"] = k
            cfg["data"]["splits"] = list(TRAINING_SPLITS)
            cfg["output"]["evaluation_root"] = str(targets[k])
            cfg["output"]["evaluation_name"] = "evaluation"
            run = EvaluationRun(
                "clustering",
                "train_val",
                cfg,
                str(root),
                f"clustering_k{k}",
                [data["manifest_csv"], data["cohort_path"]],
            )
            try:
                clusterer = GlomeruliClusterer(
                    n_clusters=k,
                    pca_variance=pca_variance,
                    manifold_method=manifold_method,
                    random_state=cfg["clustering"]["seed"],
                    input_scaling=input_scaling,
                ).fit(features[train])
                labels = clusterer.predict(features)
                distances = clusterer.distances_to_centers(features)
                mapping, medians = fit_label_mapping(labels[train], scores[train], k)
                pipeline = ClusterAssignmentPipeline(
                    extractor,
                    clusterer,
                    feature_type=feature_type,
                    morph_units=fcfg.get("morph_units", "native"),
                    block_transformer=transformer,
                    selected_features=selected_names,
                )
                pipeline.label_mapping = mapping
                pipeline.label_ordering = {
                    "method": cfg["labels"]["method"],
                    "parameters": rim_options,
                    "train_cluster_medians": medians,
                    "limitation": LABEL_LIMITATION,
                }
                pipeline.common_space = {
                    "feature_names": names,
                    "mean": common["train_mean"],
                    "scale": common["train_scale"],
                }
                pipeline.preprocessing_config = deepcopy(data)
                pipeline.training_protocol = deepcopy(TRAINING_PROTOCOL)

                rows = frame.copy()
                rows["cluster"] = labels
                rows["class_label"] = pipeline.class_labels(labels)
                rows["distance_to_center"] = distances
                rows.to_csv(run.run_dir / "assignments.csv", index=False)

                np.savez_compressed(
                    run.run_dir / "features.npz",
                    annotation_id=rows.annotation_id.to_numpy(dtype=str),
                    slide_id=rows.slide_id.to_numpy(dtype=str),
                    split=split,
                    model_features=features,
                    selected_feature_names=np.asarray(selected_names, dtype=str),
                    deep_features_raw=deep,
                    morphological_features_raw=morph,
                    common_annotation_id=common["annotation_id"],
                    common_split=common["split"],
                    common_slide_id=common["slide_id"],
                    common_morph_features=common["standardized_features"],
                    common_morph_raw=common["raw_features"],
                    common_feature_names=common["feature_names"],
                )
                joblib.dump(pipeline, run.run_dir / "pipeline.joblib")
                projected = clusterer.transform(features)
                report = {
                    "n_samples": len(rows),
                    "n_fit_samples": len(train),
                    "n_assigned_samples": len(rows) - len(train),
                    "n_clusters": k,
                    "feature_type": feature_type,
                    "selected_features": selected_names,
                    "train_metrics": clusterer._metrics(projected[train], labels[train]),
                    "pca_components": 20 if clusterer.pca is not None else None,
                    "pca_explained_variance_ratio_sum": (
                        float(clusterer.pca.explained_variance_ratio_.sum()) if clusterer.pca is not None else None
                    ),
                    "model_diagnostics": clusterer.fit_diagnostics_,
                    "label_mapping": mapping,
                    "label_ordering": pipeline.label_ordering,
                }
                write_json(run.run_dir / "clustering_report.json", report)
                write_json(run.run_dir / "label_mapping.json", pipeline.label_ordering | {"cluster_to_letter": mapping})
                run.metadata.update(
                    cohort_hash=cohort_metadata["cohort_hash"],
                    common_evaluation_space_hash=common_hash,
                    cohort_schema_version=2,
                    preprocessing_schema_version=2,
                    feature_schema_version=2,
                    cohort_provenance_mode="schema_v2",
                    fit_splits=["train"],
                    assign_splits=["val"],
                    training_protocol=TRAINING_PROTOCOL,
                    n_clusters=k,
                    algorithm="kmeans",
                    feature_type=feature_type,
                    selected_features=selected_names,
                    backbone_weights_hash=backbone_hash,
                )
                run.complete(
                    report,
                    [
                        run.run_dir / name
                        for name in ("features.npz", "pipeline.joblib", "assignments.csv", "clustering_report.json", "label_mapping.json")
                    ],
                )
                stability, _ = evaluate_run(run.run_dir, Path(cfg["output"]["resampling"]), 42)
                flags = criteria(stability)
                records = []
                for name in TRAINING_SPLITS:
                    idx = np.flatnonzero(split == name)
                    dest = run.run_dir / name
                    dest.mkdir()
                    subset = rows.iloc[idx].reset_index(drop=True)
                    part = evaluate_partition(subset, common_values[idx], names, rows.iloc[train], k, mapping)
                    write_json(dest / "metrics.json", part)
                    subset.to_csv(dest / "assignments.csv", index=False)
                    records.append({
                        "model": run.run_dir.name,
                        "split": name,
                        "n_samples": len(idx),
                        "counts": part["counts"],
                        "silhouette": part["internal_common"]["silhouette"],
                        "distance_median": part["distance_median"],
                        "criteria_passed": f"{sum(flags.values())}/5" if name == "train" else None,
                        "ari_subsampling": stability["stability"]["subsampling_ari_mean"] if name == "train" else None,
                        "jaccard_min": stability["stability"]["cluster_jaccard_min"] if name == "train" else None,
                    })
                if cfg["output"].get("save_gallery", True):
                    save_gallery(run.run_dir / "gallery.png", crops, rows, k, mapping)
                save_projection(run.run_dir / "projection.png", projected, rows, mapping)
                write_report(
                    run.run_dir,
                    records,
                    f"{run.run_dir.name}: train e validation",
                    "Fit solo sul train; mapping delle lettere congelato prima delle assegnazioni held-out. La rim fraction è un proxy esplorativo, non una diagnosi.",
                )
                inputs = {
                    str(data["manifest_csv"]): file_identity(data["manifest_csv"])["sha256"],
                    str(data["cohort_path"]): file_identity(data["cohort_path"])["sha256"],
                }
                for kind in ("masks", "xml"):
                    for identity in cohort_metadata["inputs"].get(kind, []):
                        inputs[identity["path"]] = identity["sha256"]
                for relative in sorted(set(cohort.image_path.astype(str))):
                    p = Path(data["manifest_csv"]).parent / relative
                    inputs[str(p)] = file_identity(p)["sha256"]
                outputs = {str(p.relative_to(run.run_dir)): file_identity(p)["sha256"] for p in run.run_dir.rglob("*") if p.is_file()}
                write_json(
                    run.run_dir / "complete.json",
                    {"status": "completed", "signature": signature, "training_protocol": TRAINING_PROTOCOL, "inputs": inputs, "outputs": outputs},
                )
                completed[k] = records
                print(f"Completed: {run.run_dir}", flush=True)
            except Exception as error:
                run.fail(error)
                raise

    all_records = [r for k in k_values for r in completed[k]]
    title = (
        "Stain-Invariant Morphological Clustering: confronto K=2/4/6"
        if feature_type == "stain_invariant_morph"
        else "PCA20 classica: confronto K=2/4/6"
    )
    write_report(
        root,
        all_records,
        title,
        "Fit: sei slide train; assegnazioni validation: RECHERCHE-016. Le WSI 015/017 sono riservate a inference e predict_clustering. Gallery: validation, poi train.",
    )
    if len(k_values) > 1:
        compared = compare_runs([targets[k] / "evaluation" for k in k_values])
        pd.DataFrame(compared).to_csv(root / "k_comparison.csv", index=False)
        write_json(root / "k_comparison.json", compared)
    return root
