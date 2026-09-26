#!/usr/bin/env python3
"""
CLI Script to execute Step 1: Preprocessing & Dataset Extraction.

Usage:
    python scripts/prepare_dataset.py --config configs/dataset_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataset_builder import DatasetBuilder
from src.visualization.inspect_dataset import create_inspection_grid


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
    parser = argparse.ArgumentParser(description="Extract patches from kidney biopsy WSIs.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Path to YAML dataset configuration file."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=6,
        help="Number of samples to include in the visual quality control grid."
    )
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("Initializing WSI Dataset Builder...")

    builder = DatasetBuilder(config_path=args.config)
    manifest_df = builder.build()

    logger.info("Generating visual verification grid...")
    grid_path = builder.output_dir / "inspection_grid.png"
    create_inspection_grid(
        manifest_csv=str(builder.output_dir / "manifest.csv"),
        output_png=str(grid_path),
        num_samples=args.num_samples,
    )
    logger.info(f"Visual inspection grid saved to: {grid_path}")

    # Summary table
    print("\n" + "=" * 65)
    print("DATASET EXTRACTION COMPLETE - SUMMARY STATISTICS")
    print("=" * 65)
    print(f"Total Patches Extracted : {len(manifest_df):,}")
    print(f"Positive Patches        : {(manifest_df['is_positive'] == True).sum():,}")
    print(f"Negative Patches        : {(manifest_df['is_positive'] == False).sum():,}")
    print(f"Total Glomeruli BBoxes  : {manifest_df['num_glomeruli'].sum():,}")
    print("-" * 65)
    print("Distribution by Split:")
    for split in ["train", "val", "test"]:
        sub = manifest_df[manifest_df["split"] == split]
        pos = (sub["is_positive"] == True).sum()
        neg = (sub["is_positive"] == False).sum()
        gloms = sub["num_glomeruli"].sum()
        print(f"   [{split.upper():5s}] Patches: {len(sub):5,d} (Pos: {pos:4,d}, Neg: {neg:4,d}) | Glomeruli: {gloms:4,d}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
