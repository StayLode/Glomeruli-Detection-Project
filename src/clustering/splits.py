"""Confirmed slide separation between clustering development and WSI testing."""
from pathlib import Path
import yaml


TRAINING_SPLITS = ("train", "val")
TEST_SLIDES = frozenset({"RECHERCHE-015", "RECHERCHE-017"})
TRAINING_PROTOCOL = {
    "workflow": "train_val_only_v1",
    "fit_splits": ["train"],
    "assign_splits": ["val"],
    "excluded_slide_ids": sorted(TEST_SLIDES),
}


def validate_dataset_splits(config):
    groups = {split: list(config["splits"][split]) for split in ("train", "val", "test")}
    if set(groups["test"]) != TEST_SLIDES:
        raise ValueError("The test split must contain RECHERCHE-015 and RECHERCHE-017")
    all_slides = [slide for group in groups.values() for slide in group]
    if len(all_slides) != len(set(all_slides)):
        raise ValueError("A slide occurs more than once or crosses dataset splits")
    if not groups["train"] or not groups["val"]:
        raise ValueError("Both train and validation must contain slides")
    return config


def load_split_config(path="configs/dataset_config.yaml"):
    return validate_dataset_splits(yaml.safe_load(Path(path).read_text()))


def validate_test_prediction(document, dataset_config="configs/dataset_config.yaml"):
    config = load_split_config(dataset_config)
    slide_id = document.get("slide_id")
    if slide_id not in config["splits"]["test"]:
        raise ValueError(f"predict_clustering accepts test slides only (015/017), received {slide_id!r}")
    if Path(document.get("svs_path", "")).stem != slide_id:
        raise ValueError("Prediction slide_id does not match the source WSI filename")
    return document
