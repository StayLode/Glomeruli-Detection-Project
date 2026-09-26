#!/usr/bin/env python3
"""
CLI Script to evaluate a trained YOLO checkpoint on test or validation splits.

Usage:
    python scripts/evaluate_yolo.py --weights runs/yolo/yolov8m_20x/weights/best.pt --split test
"""

import argparse
import json
import logging
import sys
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Evaluate YOLO Checkpoint on Glomeruli.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pt file)."
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        default="dataset/yolo/data.yaml",
        help="Path to data.yaml dataset config."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on."
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=True,
        help="Run inference on test images and save visual prediction overlays."
    )
    parser.add_argument(
        "--device",
        default=0,
        help="Device to run on (e.g. 0, '0', 'cpu')."
    )
    args = parser.parse_args()

    logger = setup_logger()
    logger.info(f"Evaluating checkpoint: {args.weights} on split: {args.split}")

    detector = YOLODetector(model_name_or_path=args.weights)
    metrics = detector.evaluate(data_yaml=args.data_yaml, split=args.split, device=args.device)

    if args.save_predictions:
        test_images_dir = Path("dataset/yolo/images") / args.split
        if test_images_dir.exists():
            logger.info(f"Generating visual predictions on {test_images_dir}...")
            detector.predict(
                source=str(test_images_dir),
                conf=0.25,
                iou=0.45,
                imgsz=1024,
                save=True,
                project="runs/yolo",
                name=f"predictions_{args.split}",
            )

    print("\n" + "=" * 55)
    print(f"YOLO EVALUATION RESULTS [{args.split.upper()}]")
    print("=" * 55)
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"mAP@50    : {metrics['mAP50']:.4f}")
    print(f"mAP@50-95 : {metrics['mAP50-95']:.4f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
