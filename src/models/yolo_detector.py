"""
YOLO-based glomeruli detection model wrapper using Ultralytics.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLODetector:
    """Wrapper around Ultralytics YOLO for training, evaluation, and inference."""

    def __init__(self, model_name_or_path: str = "yolov8m.pt") -> None:
        """
        Initialize the YOLO detector.

        Args:
            model_name_or_path: Path to checkpoint (.pt) or official architecture (e.g., 'yolov8m.pt').
        """
        self.model_path = str(model_name_or_path)
        logger.info(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(self.model_path)

    @staticmethod
    def ensure_data_yaml_paths(data_yaml_path: str) -> Path:
        """
        Dynamically updates 'path' in data.yaml to match the current filesystem location.
        Guarantees seamless execution both locally and on any remote server.
        """
        p = Path(data_yaml_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"data.yaml not found at: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        data["path"] = str(p.parent.resolve())

        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        return p

    def train(self, config: Dict[str, Any]) -> Any:
        """
        Train the YOLO model using parameters defined in the configuration dictionary.

        Args:
            config: Dictionary containing training parameters (from yolo_config.yaml).

        Returns:
            Ultralytics training results object.
        """
        data_yaml = config.get("data_yaml", "dataset/yolo/data.yaml")
        resolved_yaml = self.ensure_data_yaml_paths(data_yaml)

        train_args = {
            "data": str(resolved_yaml),
            "device": config.get("device", 0),
            "imgsz": config.get("imgsz", 1024),
            "epochs": config.get("epochs", 60),
            "batch": config.get("batch", 16),
            "workers": config.get("workers", 8),
            "patience": config.get("patience", 15),
            "save_period": config.get("save_period", 10),
            "project": config.get("project", "runs/yolo"),
            "name": config.get("name", "yolov8m_20x"),
            "optimizer": config.get("optimizer", "AdamW"),
            "lr0": config.get("lr0", 0.001),
            "lrf": config.get("lrf", 0.01),
            "weight_decay": config.get("weight_decay", 0.0005),
            "warmup_epochs": config.get("warmup_epochs", 3.0),
            "fliplr": config.get("fliplr", 0.5),
            "flipud": config.get("flipud", 0.5),
            "degrees": config.get("degrees", 90.0),
            "hsv_h": config.get("hsv_h", 0.015),
            "hsv_s": config.get("hsv_s", 0.3),
            "hsv_v": config.get("hsv_v", 0.3),
            "mosaic": config.get("mosaic", 0.5),
            "mixup": config.get("mixup", 0.1),
        }

        logger.info(f"Starting YOLO training with arguments:\n{train_args}")
        results = self.model.train(**train_args)
        return results

    def evaluate(self, data_yaml: str = "dataset/yolo/data.yaml", split: str = "test", device: Any = 0) -> Dict[str, float]:
        """
        Evaluate the model on a designated split ('val' or 'test').

        Args:
            data_yaml: Path to data.yaml.
            split: Dataset split to evaluate on ('val' or 'test').
            device: Device to run evaluation on (default: 0).

        Returns:
            Dictionary of evaluation metrics (Precision, Recall, mAP50, mAP50-95).
        """
        resolved_yaml = self.ensure_data_yaml_paths(data_yaml)
        logger.info(f"Running evaluation on '{split}' split on device={device}...")

        metrics = self.model.val(data=str(resolved_yaml), split=split, device=device)

        summary = {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        }

        logger.info(f"Evaluation Results on [{split.upper()}]:")
        logger.info(f"   Precision : {summary['precision']:.4f}")
        logger.info(f"   Recall    : {summary['recall']:.4f}")
        logger.info(f"   mAP@50    : {summary['mAP50']:.4f}")
        logger.info(f"   mAP@50-95 : {summary['mAP50-95']:.4f}")

        return summary

    def predict(
        self,
        source: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1024,
        save: bool = True,
        project: str = "runs/yolo",
        name: str = "predict",
    ) -> List[Any]:
        """Run inference on images or a folder of images."""
        return self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=save,
            project=project,
            name=name,
        )
