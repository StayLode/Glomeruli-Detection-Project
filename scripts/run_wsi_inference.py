#!/usr/bin/env python3
"""
CLI Script to perform Whole Slide Image (WSI) Inference and Stitching.

Usage examples:
    # Run on a single test slide:
    python scripts/run_wsi_inference.py \
        --slide glomeruli_grading/RECHERCHE-015.svs \
        --weights runs/yolo/yolov8m_20x_baseline/weights/best.pt

    # Run on all test slides defined in config:
    python scripts/run_wsi_inference.py \
        --all_test_slides \
        --weights runs/yolo/yolov8m_20x_baseline/weights/best.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.wsi_infer import WSIInferenceEngine


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
    parser = argparse.ArgumentParser(description="Run WSI Glomeruli Detection & Stitching.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained YOLO checkpoint (best.pt)."
    )
    parser.add_argument(
        "--slide",
        type=str,
        default=None,
        help="Path to a specific .svs slide file."
    )
    parser.add_argument(
        "--all_test_slides",
        action="store_true",
        help="Run inference on all test slides defined in dataset_config.yaml."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Path to dataset configuration YAML."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/wsi_inference",
        help="Directory to save WSI prediction artifacts (XML, overview image, metrics)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO predictions."
    )
    parser.add_argument(
        "--nms_iou",
        type=float,
        default=0.40,
        help="IoU threshold for Global Non-Maximum Suppression."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Inference batch size on GPU."
    )
    args = parser.parse_args()

    logger = setup_logger()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine slides to process
    slides_to_process: List[Path] = []
    if args.slide:
        slides_to_process.append(Path(args.slide))
    elif args.all_test_slides:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        raw_dir = Path(cfg["raw_data_dir"])
        for sid in cfg["splits"]["test"]:
            svs_p = raw_dir / f"{sid}.svs"
            if svs_p.exists():
                slides_to_process.append(svs_p)
    else:
        logger.error("Please specify either --slide <path_to_svs> or --all_test_slides.")
        sys.exit(1)

    # 2. Initialize Inference Engine
    engine = WSIInferenceEngine(
        model_weights_path=args.weights,
        patch_size=1024,
        stride=768,
        target_mag=20,
    )

    all_slide_evals = []

    # 3. Process each slide
    for svs_p in slides_to_process:
        sid = svs_p.stem
        xml_p = svs_p.parent / f"{sid}.xml"
        xml_arg = str(xml_p) if xml_p.exists() else None

        result = engine.predict_slide(
            svs_path=str(svs_p),
            xml_path=xml_arg,
            conf_thresh=args.conf,
            nms_iou_thresh=args.nms_iou,
            batch_size=args.batch_size,
        )

        # A. Export ASAP XML
        asap_xml_path = output_dir / f"{sid}_predicted.xml"
        engine.export_asap_xml(result["detected_boxes_l0"], str(asap_xml_path))

        # B. Export Overview Image
        overview_path = output_dir / f"{sid}_overview.png"
        engine.generate_overview_plot(
            svs_path=str(svs_p),
            pred_boxes_l0=result["detected_boxes_l0"],
            gt_boxes_l0=result["gt_boxes_l0"] if xml_arg else None,
            output_path=str(overview_path),
        )

        # C. Save Metrics JSON
        if result["evaluation"]:
            all_slide_evals.append((sid, result["evaluation"]))
            metrics_json = output_dir / f"{sid}_metrics.json"
            with open(metrics_json, "w", encoding="utf-8") as f:
                json.dump(result["evaluation"], f, indent=2)

    # Print summary table
    if all_slide_evals:
        print("\n" + "=" * 70)
        print("WHOLE SLIDE INFERENCE (WSI) CLINICAL EVALUATION REPORT")
        print("=" * 70)
        print(f"{'Slide ID':<16} | {'GT':<4} | {'Pred':<4} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<9} | {'Recall':<9} | {'F1':<9}")
        print("-" * 70)
        for sid, ev in all_slide_evals:
            print(f"{sid:<16} | {ev['gt_count']:<4} | {ev['pred_count']:<4} | {ev['tp']:<4} | {ev['fp']:<4} | {ev['fn']:<4} | {ev['precision']:<9.4f} | {ev['recall']:<9.4f} | {ev['f1']:<9.4f}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
