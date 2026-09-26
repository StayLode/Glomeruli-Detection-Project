#!/usr/bin/env python3
"""
CLI Script to train YOLO on the extracted glomeruli dataset.

Usage:
    python scripts/train_yolo.py --config configs/yolo_config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import torch
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.yolo_detector import YOLODetector


def setup_logger() -> logging.Logger:
    """Configure structured console logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on Glomeruli Dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/yolo_config.yaml",
        help="Path to YOLO training config YAML file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model architecture (e.g. yolov8s.pt, yolov8m.pt, yolov8x.pt)."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch size (e.g. 16 or 32 for NVIDIA A40)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (e.g. 0, '0', 'cpu'). Defaults to config or GPU 0 if available."
    )
    args = parser.parse_args()

    logger = setup_logger()
    logger.info(f"Loading configuration from {args.config}...")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Allow CLI overrides
    if args.model:
        cfg["model"] = args.model
    if args.batch:
        cfg["batch"] = args.batch
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.device is not None:
        cfg["device"] = args.device

    # Hardware verification
    logger.info(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using GPU [0]: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        logger.warning("CUDA is NOT detected by PyTorch! Check nvidia driver and CUDA installation.")

    detector = YOLODetector(model_name_or_path=cfg["model"])

    # 1. Train model
    logger.info("Starting training loop...")
    results = detector.train(cfg)

    # 2. Evaluate on Test set using the best weights
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (Path(cfg.get("project", "runs/yolo")) / cfg.get("name", "yolov8m_20x"))
    best_weights = save_dir / "weights" / "best.pt"
    if best_weights.exists():
        logger.info(f"Loading best checkpoint for test evaluation: {best_weights}")
        evaluator = YOLODetector(model_name_or_path=str(best_weights))
    else:
        evaluator = detector

    test_metrics = evaluator.evaluate(
        data_yaml=cfg.get("data_yaml", "dataset/yolo/data.yaml"),
        split="test",
        device=cfg.get("device", 0)
    )

    # 3. Save test metrics
    metrics_path = save_dir / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(test_metrics, mf, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("YOLO TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Best Model Weights : {best_weights}")
    print(f"Test Precision     : {test_metrics['precision']:.4f}")
    print(f"Test Recall        : {test_metrics['recall']:.4f}")
    print(f"Test mAP@50        : {test_metrics['mAP50']:.4f}")
    print(f"Test mAP@50-95     : {test_metrics['mAP50-95']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
