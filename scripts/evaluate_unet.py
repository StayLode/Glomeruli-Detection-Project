#!/usr/bin/env python3
"""
CLI Script to evaluate a trained U-Net checkpoint on the test set.

Usage:
    python scripts/evaluate_unet.py --checkpoint runs/unet/unet_20x_baseline/weights/best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet_trainer import UNetTrainer


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
    parser = argparse.ArgumentParser(description="Evaluate U-Net Checkpoint on Test Set.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained U-Net checkpoint (best.pt)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/unet_config.yaml",
        help="Path to U-Net config YAML file."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid probability threshold."
    )
    args = parser.parse_args()

    logger = setup_logger()
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")

    trainer = UNetTrainer(config_path=args.config)
    trainer.cfg["threshold"] = args.threshold

    results = trainer.evaluate_test(checkpoint_path=args.checkpoint, save_visualizations=True)

    seg_m = results["segmentation_metrics"]
    det_m = results["detection_metrics"]

    print("\n" + "=" * 65)
    print("U-NET TEST EVALUATION REPORT")
    print("=" * 65)
    print("PIXEL-LEVEL SEGMENTATION:")
    print(f"   Dice Score (F1)     : {seg_m['dice']:.4f}")
    print(f"   IoU (Jaccard Index) : {seg_m['iou']:.4f}")
    print(f"   Pixel Precision     : {seg_m['pixel_precision']:.4f}")
    print(f"   Pixel Recall        : {seg_m['pixel_recall']:.4f}")
    print("-" * 65)
    print("OBJECT-LEVEL DETECTION (Extracted BBoxes):")
    print(f"   mAP@50 (AP50)       : {det_m['ap50']:.4f}")
    print(f"   BBox Precision      : {det_m['precision']:.4f}")
    print(f"   BBox Recall         : {det_m['recall']:.4f}")
    print(f"   BBox F1-Score       : {det_m['f1']:.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
