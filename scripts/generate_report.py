#!/usr/bin/env python3
"""
CLI Script to Generate the Final End-to-End Pipeline Report & Visualizations (Step 4).

Generates:
1. Publication-quality 6-panel composite figure (PNG & PDF).
2. Interactive, standalone HTML report with embedded visualizations.
3. Executive Markdown summary report.
4. Consolidated metrics table (CSV & JSON).

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --output_dir runs/final_report
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.pipeline_reporter import PipelineReporter


def setup_logger() -> logging.Logger:
    """Configure structured console logging."""
    logger = logging.getLogger("generate_report")
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
    parser = argparse.ArgumentParser(
        description="Generate End-to-End Glomeruli Detection & Grading Final Report (Step 4)."
    )
    parser.add_argument(
        "--yolo_run",
        type=str,
        default="runs/detect/runs/yolo/yolov8m_20x_baseline",
        help="Path to YOLO experiment directory containing test_metrics.json.",
    )
    parser.add_argument(
        "--unet_run",
        type=str,
        default="runs/unet/unet_20x_baseline",
        help="Path to U-Net experiment directory containing test_metrics.json.",
    )
    parser.add_argument(
        "--clustering_run",
        type=str,
        default="runs/clustering/unsupervised_grading_3classes",
        help="Path to clustering experiment directory containing clustering_report.json.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="dataset/manifest.csv",
        help="Path to dataset manifest.csv.",
    )
    parser.add_argument(
        "--slide",
        type=str,
        default="glomeruli_grading/RECHERCHE-015.svs",
        help="Path to a test Whole Slide Image (.svs) for overview visualization.",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default="glomeruli_grading/RECHERCHE-015.xml",
        help="Path to ground truth ASAP XML annotation file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/final_report",
        help="Directory where report artifacts will be saved.",
    )
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("Initializing End-to-End Pipeline Reporter...")

    reporter = PipelineReporter(
        yolo_run_dir=args.yolo_run,
        unet_run_dir=args.unet_run,
        clustering_run_dir=args.clustering_run,
        manifest_csv=args.manifest,
        output_dir=args.output_dir,
        slide_path=args.slide,
        xml_path=args.xml,
    )

    results = reporter.run_all()

    print("\n" + "=" * 70)
    print("END-TO-END PIPELINE REPORT GENERATED SUCCESSFULLY (STEP 4)")
    print("=" * 70)
    print(f"1. Publication Summary (PNG) : {results['summary_figure_png']}")
    print(f"2. Publication Summary (PDF) : {results['summary_figure_pdf']}")
    print(f"3. Interactive HTML Report   : {results['html_report']}")
    print(f"4. Markdown Executive Report : {results['markdown_report']}")
    print(f"5. Consolidated Metrics CSV  : {results['metrics_csv']}")
    print(f"6. Consolidated Metrics JSON : {results['metrics_json']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
