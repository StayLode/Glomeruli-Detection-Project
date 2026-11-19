import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yaml

from src.clustering.feature_extractor import STAIN_INVARIANT_MORPH_FEATURES
from src.clustering.labels import fit_label_mapping, peripheral_clear_fraction
from src.clustering.manifold_clustering import GlomeruliClusterer
from src.clustering.metrics import _cramers_v, _metric_result
from src.clustering.prediction import (
    extract_wsi_crop,
    load_models,
    load_predictions,
    run_prediction,
    select_test_predictions,
)
from src.clustering.training import ensure_cohort, run_training, validate_splits
from src.utils.evaluation import file_identity, write_json

NAMES = [
    "glom_area_um2",
    "glom_perimeter_um",
    "circularity",
    "solidity",
    "extent",
    "aspect_ratio",
    "mean_intensity",
    "mean_optical_density",
    "stain_heterogeneity",
    "red_green_ratio",
]


class SyntheticExtractor:
    def __init__(self, *args, **kwargs):
        pass

    def weights_sha256(self):
        return "synthetic-encoder"

    def extract_deep_embeddings(self, crops):
        return np.asarray(
            [np.linspace(0, 1, 25) + np.mean(c) / 255 for c in crops],
            dtype=np.float32,
        )


class SyntheticSlide:
    dimensions = (2048, 2048)
    properties = {"openslide.mpp-x": "0.2519", "openslide.mpp-y": "0.2519"}
    level_downsamples = [1.0, 4.0]

    def get_best_level_for_downsample(self, value):
        return 0

    def read_region(self, location, level, size):
        return Image.new("RGBA", size, (180, 160, 170, 255))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class SolutionTests(unittest.TestCase):
    def test_cohort_build_and_audit_never_open_test_resources(self):
        import cv2
        from src.clustering.cohort import parse_asap_xml

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            xml = root / "xml"
            xml.mkdir()
            cfg = yaml.safe_load(Path("configs/clustering_config.yaml").read_text())
            split_config = yaml.safe_load(Path("configs/dataset_config.yaml").read_text())
            split_path = root / "splits.yaml"
            split_path.write_text(yaml.safe_dump(split_config))
            records = []
            for split, slides in split_config["splits"].items():
                for slide in slides:
                    image = f"{slide}_image.png"
                    mask = f"{slide}_mask.png"
                    records.append(
                        {
                            "patch_id": slide + "_patch",
                            "slide_id": slide,
                            "split": split,
                            "x0_l0": 0,
                            "y0_l0": 0,
                            "size_l0": 64,
                            "patch_size": 64,
                            "mask_path": mask,
                            "image_path": image,
                            "tissue_ratio": 1.0,
                        }
                    )
                    if split == "test":
                        (xml / f"{slide}.xml").write_text("must never be parsed or hashed")
                        continue
                    pixels = np.zeros((64, 64), np.uint8)
                    pixels[8:56, 8:56] = 255
                    Image.fromarray(pixels).save(dataset / mask)
                    Image.fromarray(np.full((64, 64, 3), 180, np.uint8)).save(dataset / image)
                    (xml / f"{slide}.xml").write_text(
                        '<ASAP_Annotations><Annotations><Annotation Name="g" Type="Polygon"><Coordinates>'
                        + "".join(
                            f'<Coordinate Order="{i}" X="{x}" Y="{y}"/>'
                            for i, (x, y) in enumerate([(8, 8), (55, 8), (55, 55), (8, 55)])
                        )
                        + "</Coordinates></Annotation></Annotations></ASAP_Annotations>"
                    )
            manifest = dataset / "manifest.csv"
            pd.DataFrame(records).to_csv(manifest, index=False)
            cfg["data"].update(
                manifest_csv=str(manifest),
                xml_dir=str(xml),
                dataset_config=str(split_path),
                cohort_path=str(root / "runs/cohort_train_val/cohort.csv"),
            )
            cfg["output"].update(
                run_root=str(root / "runs"),
                resampling=str(root / "resampling.json"),
            )

            def guarded(function):
                def call(path, *args, **kwargs):
                    self.assertNotIn("RECHERCHE-015", str(path))
                    self.assertNotIn("RECHERCHE-017", str(path))
                    return function(path, *args, **kwargs)

                return call

            with (
                patch("src.clustering.cohort.file_identity", side_effect=guarded(file_identity)),
                patch("src.clustering.cohort.parse_asap_xml", side_effect=guarded(parse_asap_xml)),
                patch("src.clustering.cohort.cv2.imread", side_effect=guarded(cv2.imread)),
            ):
                cohort, metadata = ensure_cohort(cfg)
                ensure_cohort(cfg)
            self.assertEqual(set(cohort.split), {"train", "val"})
            self.assertEqual(len(cohort), 7)
            self.assertEqual(set(pd.read_csv(cfg["data"]["manifest_csv"]).split), {"train", "val"})
            self.assertEqual(len(metadata["inputs"]["xml"]), 7)
            for kind in ("xml", "masks"):
                self.assertFalse(
                    any(
                        slide in entry["path"]
                        for entry in metadata["inputs"][kind]
                        for slide in ("RECHERCHE-015", "RECHERCHE-017")
                    )
                )

    def test_prediction_batch_requires_both_test_slides_and_rejects_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def export(slide):
                path = root / f"{slide}_predictions.json"
                write_json(
                    path,
                    {
                        "stage": "wsi_predictions",
                        "cascade_mode": True,
                        "slide_id": slide,
                        "svs_path": str(root / f"{slide}.svs"),
                        "detections": [],
                    },
                )
                return path

            export("RECHERCHE-015")
            with self.assertRaisesRegex(ValueError, "Missing test"):
                select_test_predictions(root)
            export("RECHERCHE-017")
            train = export("RECHERCHE-003")
            self.assertEqual(
                [p.name for p in select_test_predictions(root)],
                ["RECHERCHE-015_predictions.json", "RECHERCHE-017_predictions.json"],
            )
            with patch("src.clustering.prediction.load_models") as models:
                with self.assertRaisesRegex(ValueError, "test slides only"):
                    run_prediction(train)
                models.assert_not_called()
            source = root / "clustering_k2"
            source.mkdir()
            write_json(source / "complete.json", {"status": "completed", "outputs": {}})
            with self.assertRaisesRegex(ValueError, "Legacy model"):
                load_models(root, [2])

    def test_rim_proxy_and_train_mapping_are_explicitly_morphological(self):
        mask = np.zeros((64, 64), np.uint8)
        mask[8:56, 8:56] = 255
        self.assertEqual(peripheral_clear_fraction(np.full((64, 64, 3), 255, np.uint8), mask), 1.0)
        self.assertEqual(peripheral_clear_fraction(np.zeros((64, 64, 3), np.uint8), mask), 0.0)
        self.assertTrue(np.isnan(peripheral_clear_fraction(np.zeros((64, 64, 3), np.uint8), mask * 0)))
        mapping, _ = fit_label_mapping([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 2)
        self.assertEqual(mapping, {1: "A", 0: "B"})
        for k in (2, 4, 6):
            mapping, _ = fit_label_mapping(np.arange(k), np.arange(k), k)
            self.assertEqual(sorted(mapping.values()), list("ABCDEF"[:k]))

    def test_single_cluster_and_single_slide_are_not_zero_filled(self):
        self.assertIsNone(_metric_result(np.ones((4, 2)), np.zeros(4))["silhouette"])
        self.assertIsNone(_cramers_v(pd.DataFrame({"cluster": [0, 1], "slide_id": ["s", "s"]}))["value"])

    def test_wsi_geometry_and_physical_units_and_rectangles(self):
        crop, mask, pixel = extract_wsi_crop(
            SyntheticSlide(), [[100, 100], [500, 100], [500, 500], [100, 500]]
        )
        self.assertEqual(crop.shape, (512, 512, 3))
        self.assertEqual(mask.shape, (512, 512))
        self.assertAlmostEqual(pixel, 0.5038 * 250 / 512, places=5)
        with self.assertRaisesRegex(ValueError, "invalid_polygon"):
            extract_wsi_crop(SyntheticSlide(), [[0, 0], [1, 1], [2, 2]])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.json"
            write_json(path, {"stage": "wsi_predictions", "cascade_mode": False})
            with self.assertRaisesRegex(ValueError, "YOLO"):
                load_predictions(path)

    def test_post_nms_export_preserves_mask_provenance(self):
        from src.inference.prediction_io import export_predictions

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "weights"
            weights.write_bytes(b"weights")
            data = {
                "slide_id": "s",
                "svs_path": "s.svs",
                "cascade_mode": True,
                "level0_dimensions": [100, 100],
                "detected_boxes_l0": [[0, 0, 10, 10], [20, 20, 30, 30]],
                "detected_polygons_l0": [
                    [[0, 0], [10, 0], [10, 10]],
                    [[20, 20], [30, 20], [30, 30]],
                ],
                "detected_scores": [0.9, 0.8],
                "patch_records": [{"patch_contour": np.ones((3, 1, 2))}, {"patch_contour": None}],
                "evaluation": None,
            }
            out = export_predictions(data, root / "out.json", weights, weights)
            result = json.loads(out.read_text())
            self.assertEqual([r["mask_source"] for r in result["detections"]], ["unet", "bbox_fallback"])

    def test_whole_cycle_three_k_resume_and_frozen_wsi_assignment(self):
        rng = np.random.default_rng(55)
        ntrain, nval = 36, 8
        total = ntrain + nval
        ids = np.asarray([f"g{i:03d}" for i in range(total)])
        splits = np.asarray(["train"] * ntrain + ["val"] * nval)
        train_slides = [f"RECHERCHE-{s:03d}" for s in (3, 4, 5, 9, 10, 11)]
        slides = np.asarray([train_slides[i % 6] for i in range(ntrain)] + ["RECHERCHE-016"] * nval)
        deep = rng.normal(size=(total, 25)).astype(np.float32)
        morph = rng.normal(size=(total, 10)).astype(np.float32)
        crops = [np.full((64, 64, 3), 50 + i * 3, np.uint8) for i in range(total)]
        masks = [np.pad(np.full((48, 48), 255, np.uint8), 8) for i in range(total)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "image").write_bytes(b"image")
            (root / "manifest.csv").write_text("manifest")
            cohort = pd.DataFrame(
                {"annotation_id": ids, "slide_id": slides, "split": splits, "image_path": "image"}
            )
            cohort.to_csv(root / "cohort.csv", index=False)
            meta = {"cohort_hash": "synthetic", "inputs": {"masks": [], "xml": []}}
            cfg = yaml.safe_load(Path("configs/clustering_config.yaml").read_text())
            cfg["data"].update(manifest_csv=str(root / "manifest.csv"), cohort_path=str(root / "cohort.csv"))
            cfg["output"].update(
                run_root=str(root / "runs"),
                resampling=str(root / "resampling.json"),
                save_gallery=False,
            )
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(cfg))
            common = {
                "annotation_id": ids,
                "slide_id": slides,
                "split": splits,
                "raw_features": morph.astype(float),
                "standardized_features": morph.astype(float),
                "feature_names": np.asarray(NAMES),
                "train_mean": np.zeros(10),
                "train_scale": np.ones(10),
            }
            metadata = cohort.to_dict("records")
            validate_splits(cohort, yaml.safe_load(Path("configs/dataset_config.yaml").read_text()))
            bad = cohort.copy()
            bad.loc[0, "slide_id"] = "RECHERCHE-015"
            with self.assertRaises(ValueError):
                validate_splits(bad, yaml.safe_load(Path("configs/dataset_config.yaml").read_text()))
            with ExitStack() as stack:
                stack.enter_context(patch("src.clustering.training.ensure_cohort", return_value=(cohort, meta)))
                loader = stack.enter_context(
                    patch(
                        "src.clustering.training.load_cohort_samples",
                        return_value=(crops, masks, metadata, [1.0] * total, pd.DataFrame()),
                    )
                )
                stack.enter_context(
                    patch("src.clustering.training.GlomeruliFeatureExtractor", SyntheticExtractor)
                )
                encoder = stack.enter_context(
                    patch("src.clustering.training._cached_embeddings", return_value=deep)
                )
                stack.enter_context(
                    patch(
                        "src.clustering.training.extract_morphological_dataset",
                        return_value=(morph, NAMES),
                    )
                )
                stack.enter_context(
                    patch(
                        "src.clustering.training.load_or_create_common_space",
                        return_value=(common, "shared-common"),
                    )
                )
                output = run_training(str(config))
                self.assertEqual(encoder.call_count, 0)
                self.assertEqual(loader.call_args.args[2], ["train", "val"])
                for k in (2, 4, 6):
                    run = output / f"clustering_k{k}"
                    model = joblib.load(run / "pipeline.joblib")
                    self.assertEqual(model.feature_type, "stain_invariant_morph")
                    self.assertIsNone(model.clusterer.pca)
                    self.assertEqual(model.clusterer.scaler.n_samples_in_, ntrain)
                    self.assertEqual(
                        model.clusterer.scaler.n_features_in_, len(STAIN_INVARIANT_MORPH_FEATURES)
                    )
                    self.assertEqual(sorted(model.label_mapping.values()), list("ABCDEF"[:k]))
                    with np.load(run / "features.npz", allow_pickle=False) as arrays:
                        for key in arrays.files:
                            arrays[key]
                        self.assertEqual(
                            arrays["model_features"].shape,
                            (total, len(STAIN_INVARIANT_MORPH_FEATURES)),
                        )
                        self.assertEqual(set(arrays["split"]), {"train", "val"})
                        self.assertFalse(set(arrays["slide_id"]) & {"RECHERCHE-015", "RECHERCHE-017"})
                    for split, count in [("train", ntrain), ("val", nval)]:
                        metrics = json.loads((run / split / "metrics.json").read_text())
                        self.assertEqual(metrics["n_samples"], count)
                    self.assertTrue((run / "evaluation/metrics.json").is_file())
                    self.assertFalse((run / "test").exists())
                    evaluation = json.loads((run / "evaluation/metrics.json").read_text())
                    self.assertEqual(set(evaluation["common_space_internal_metrics"]), {"train", "val"})
                before = file_identity(output / "clustering_k2/complete.json")["sha256"]
                run_training(str(config))
                self.assertEqual(encoder.call_count, 0)
                self.assertEqual(file_identity(output / "clustering_k2/complete.json")["sha256"], before)
            predictions = root / "predictions.json"
            slide = root / "RECHERCHE-015.svs"
            slide.write_bytes(b"slide")
            poly = [[100, 100], [500, 100], [500, 500], [100, 500]]
            write_json(
                predictions,
                {
                    "stage": "wsi_predictions",
                    "slide_id": "RECHERCHE-015",
                    "svs_path": str(slide),
                    "cascade_mode": True,
                    "detections": [
                        {"prediction_id": "valid", "polygon_l0": poly, "mask_source": "unet"},
                        {"prediction_id": "rectangle", "polygon_l0": poly, "mask_source": "bbox_fallback"},
                    ],
                },
            )
            hashes = {k: file_identity(output / f"clustering_k{k}/pipeline.joblib")["sha256"] for k in (2, 4, 6)}
            with (
                patch("src.clustering.prediction.openslide.OpenSlide", return_value=SyntheticSlide()),
                patch.object(KMeans, "fit", side_effect=AssertionError("must not fit during prediction")),
                patch.object(
                    StandardScaler, "fit", side_effect=AssertionError("must not fit during prediction")
                ),
                patch.object(
                    StandardScaler,
                    "fit_transform",
                    side_effect=AssertionError("must not fit during prediction"),
                ),
                patch.object(PCA, "fit", side_effect=AssertionError("must not fit during prediction")),
                patch.object(
                    PCA, "fit_transform", side_effect=AssertionError("must not fit during prediction")
                ),
            ):
                result = run_prediction(predictions, output, root / "predictions")
            for k in (2, 4, 6):
                m = json.loads((result / f"clustering_k{k}/metrics.json").read_text())
                self.assertEqual((m["n_detected"], m["n_assigned"], m["n_excluded"]), (2, 1, 1))
                self.assertIsNone(m["clinical_accuracy"])
                self.assertEqual(file_identity(output / f"clustering_k{k}/pipeline.joblib")["sha256"], hashes[k])


if __name__ == "__main__":
    unittest.main()
