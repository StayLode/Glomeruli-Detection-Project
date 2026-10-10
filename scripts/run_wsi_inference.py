#!/usr/bin/env python3
"""
CLI Script to perform Whole Slide Image (WSI) Inference and Stitching.

Supports:
1. Fast Screening Mode (YOLO only):
   python scripts/run_wsi_inference.py \
       --slide glomeruli_grading/RECHERCHE-015.svs \
       --weights runs/yolo/yolov8m_20x_baseline/weights/best.pt

2. Cascade Mode (YOLO Screening + U-Net Gated Segmentation):
   python scripts/run_wsi_inference.py \
       --slide glomeruli_grading/RECHERCHE-015.svs \
       --weights runs/yolo/yolov8m_20x_baseline/weights/best.pt \
       --unet_weights runs/unet/unet_20x_resnet34/weights/best.pt

3. Batch Evaluation on All Held-Out Test Slides:
   python scripts/run_wsi_inference.py \
       --all_test_slides \
       --weights runs/yolo/yolov8m_20x_baseline/weights/best.pt \
       --unet_weights runs/unet/unet_20x_resnet34/weights/best.pt
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
    logger = logging.getLogger("wsi_runner")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WSI Glomeruli Detection & Segmentation.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained YOLO checkpoint (best.pt).",
    )
    parser.add_argument(
        "--unet_weights",
        type=str,
        default=None,
        help="Optional path to trained U-Net checkpoint (best.pt) to enable Cascade Mode.",
    )
    parser.add_argument(
        "--slide",
        type=str,
        default=None,
        help="Path to a specific .svs Whole Slide Image file.",
    )
    parser.add_argument(
        "--all_test_slides",
        action="store_true",
        help="Run inference on all test slides defined in dataset_config.yaml.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Path to dataset configuration YAML.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/wsi_inference",
        help="Directory to save WSI prediction artifacts (XML, overview image, comparison grid, metrics).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="YOLO proposal confidence threshold (default: 0.15 for high screening recall).",
    )
    parser.add_argument(
        "--unet_threshold",
        type=float,
        default=0.45,
        help="U-Net probability threshold for foreground mask (default: 0.45).",
    )
    parser.add_argument(
        "--min_glom_area",
        type=int,
        default=100,
        help="Minimum pixel area for valid glomerular tuft segmentation (default: 100).",
    )
    parser.add_argument(
        "--nms_iou",
        type=float,
        default=0.35,
        help="IoU threshold for Global Non-Maximum Suppression (default: 0.35).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Inference batch size on GPU (default: 16).",
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

    # 2. Initialize Inference Engine (Cascade Mode if unet_weights provided)
    engine = WSIInferenceEngine(
        model_weights_path=args.weights,
        unet_weights_path=args.unet_weights,
        patch_size=1024,
        stride=768,
        target_mag=20,
        unet_threshold=args.unet_threshold,
        min_glom_area=args.min_glom_area,
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

        # A. Export ASAP XML (Polygons if cascade mode, otherwise BBoxes)
        asap_xml_path = output_dir / f"{sid}_predicted.xml"
        engine.export_asap_xml(
            polygons_l0=result["detected_polygons_l0"],
            boxes_l0=result["detected_boxes_l0"],
            output_xml_path=str(asap_xml_path),
        )

        # B. Export Whole-Slide Overview Image
        overview_path = output_dir / f"{sid}_overview.png"
        engine.generate_overview_plot(
            svs_path=str(svs_p),
            pred_polygons_l0=result["detected_polygons_l0"],
            pred_boxes_l0=result["detected_boxes_l0"],
            gt_annots=result["gt_annots"],
            output_path=str(overview_path),
            cascade_mode=result["cascade_mode"],
            eval_metrics=result["evaluation"],
        )

        # C. Export Multi-Panel Comparison Grid (Raw vs GT vs Pred vs Overlap)
        grid_path = output_dir / f"{sid}_comparison_grid.png"
        engine.generate_comparison_grid(
            patch_records=result["patch_records"],
            pred_polygons_l0=result["detected_polygons_l0"],
            pred_boxes_l0=result["detected_boxes_l0"],
            gt_annots=result["gt_annots"],
            output_path=str(grid_path),
            max_samples=6,
        )

        # D. Save Metrics JSON
        if result["evaluation"]:
            all_slide_evals.append((sid, result["cascade_mode"], result["evaluation"]))
            metrics_json = output_dir / f"{sid}_metrics.json"

            def _to_json_compat(obj):
                if hasattr(obj, "item"):
                    return obj.item()
                if isinstance(obj, (tuple, set)):
                    return list(obj)
                return str(obj)

            with open(metrics_json, "w", encoding="utf-8") as f:
                json.dump(result["evaluation"], f, indent=2, default=_to_json_compat)

    # Print Clinical Benchmark Summary Table
    if all_slide_evals:
        print("\n" + "=" * 90)
        print("WHOLE SLIDE INFERENCE (WSI) CLINICAL EVALUATION BENCHMARK")
        print("=" * 90)
        header = f"{'Slide ID':<15} | {'Mode':<18} | {'GT':<4} | {'Pred':<4} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'Dice':<7}"
        print(header)
        print("-" * 90)
        for sid, is_cascade, ev in all_slide_evals:
            mode_lbl = "Cascade (Y+U)" if is_cascade else "YOLO Only"
            dice_lbl = f"{ev['mean_dice']*100:.1f}%" if ev.get("mean_dice") is not None else "N/A"
            print(
                f"{sid:<15} | {mode_lbl:<18} | {ev['gt_count']:<4} | {ev['pred_count']:<4} | "
                f"{ev['tp']:<4} | {ev['fp']:<4} | {ev['fn']:<4} | {ev['precision']*100:5.1f}% | "
                f"{ev['recall']*100:5.1f}% | {ev['f1']*100:5.1f}% | {dice_lbl:<7}"
            )
        print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
