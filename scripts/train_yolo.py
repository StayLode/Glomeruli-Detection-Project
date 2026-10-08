#!/usr/bin/env python3
"""
CLI Script to train and/or evaluate YOLOv8 on the glomeruli dataset.

Usage:
    # To run the full training and evaluation pipeline automatically:
    python scripts/train_yolo.py --config configs/yolo_config.yaml
    
    # To evaluate a previously trained model on the test set:
    python scripts/train_yolo.py --config configs/yolo_config.yaml --eval_only --weights runs/yolo/.../best.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# Add project root to sys.path to resolve internal module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logger() -> logging.Logger:
    """Configure structured console logging."""
    logger = logging.getLogger("yolo_runner")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] : %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def ensure_data_yaml_paths(data_yaml_path: str) -> str:
    """
    Dynamically updates the 'path' attribute in data.yaml to match the current filesystem location.
    This guarantees seamless execution both locally and on remote SLURM clusters.
    """
    p = Path(data_yaml_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Dataset configuration file (data.yaml) not found at: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Overwrite the root dataset path with the absolute path of the current directory
    data["path"] = str(p.parent.resolve())

    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return str(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and/or Evaluate YOLOv8 on the Glomeruli Dataset.")
    parser.add_argument("--config", type=str, default="configs/yolo_config.yaml", help="Path to YOLO configuration YAML.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training phase and run evaluation only.")
    parser.add_argument("--weights", type=str, default=None, help="Path to pre-trained weights for eval_only mode (e.g., best.pt).")
    args = parser.parse_args()

    logger = setup_logger()
    
    # 1. Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    resolved_yaml = ensure_data_yaml_paths(cfg.get("data_yaml", "dataset/yolo/data.yaml"))

    # 2. Hardware verification
    device = cfg.get("device", 0)
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA NOT detected! Training will be extremely slow on CPU.")

    # ==========================================
    # TRAINING PHASE
    # ==========================================
    project_dir = str((PROJECT_ROOT / cfg.get("project", "runs/yolo")).resolve())
    exp_name = cfg.get("name", "yolov8m_20x_baseline")

    if not args.eval_only:
        logger.info(f"Initializing YOLO model using architecture: {cfg.get('model')}")
        model = YOLO(cfg.get("model"))
        
        train_args = {
            "data": resolved_yaml,
            "device": device,
            "imgsz": cfg.get("imgsz", 1024),
            "epochs": cfg.get("epochs", 60),
            "batch": cfg.get("batch", 16),
            "workers": cfg.get("workers", 8),
            "patience": cfg.get("patience", 15),
            "project": project_dir,
            "name": exp_name,
            "optimizer": cfg.get("optimizer", "AdamW"),
            "lr0": cfg.get("lr0", 0.001),
            "lrf": cfg.get("lrf", 0.01),
            "weight_decay": cfg.get("weight_decay", 0.0005),
            # Histology-specific augmentations
            "fliplr": cfg.get("fliplr", 0.5),
            "flipud": cfg.get("flipud", 0.5),
            "degrees": cfg.get("degrees", 90.0),
            "hsv_h": cfg.get("hsv_h", 0.015),
            "hsv_s": cfg.get("hsv_s", 0.3),
            "hsv_v": cfg.get("hsv_v", 0.3),
        }
        
        logger.info("Starting YOLO training loop...")
        results = model.train(**train_args)
        
        # Dynamically resolve the path to the newly generated best weights
        save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (Path(project_dir) / exp_name)
        weights_path = save_dir / "weights" / "best.pt"
    else:
        # Evaluation-Only Mode
        if not args.weights:
            logger.error("A valid --weights path must be provided when using --eval_only flag.")
            sys.exit(1)
        weights_path = Path(args.weights)
        save_dir = weights_path.parent.parent

    # ==========================================
    # EVALUATION PHASE (Test Set)
    # ==========================================
    logger.info(f"Evaluating optimal checkpoint: {weights_path}")
    eval_model = YOLO(str(weights_path))
    
    # Run quantitative validation on the test split
    metrics = eval_model.val(
        data=resolved_yaml,
        split="test",
        device=device,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
    )
    
    # Generate visual bounding box predictions for qualitative analysis
    test_images_dir = Path("dataset/yolo/images/test")
    if test_images_dir.exists():
        logger.info("Generating visual prediction overlays on the test set...")
        eval_model.predict(
            source=str(test_images_dir),
            conf=0.25, 
            iou=0.45, 
            imgsz=cfg.get("imgsz", 1024),
            save=True, 
            project=str(save_dir.parent), 
            name="predictions_test",
            exist_ok=True,
        )

    
    summary = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }

    metrics_path = save_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(summary, mf, indent=2)

    print("\n" + "=" * 60)
    print("YOLO EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Test Precision     : {summary['precision']:.4f}")
    print(f"Test Recall        : {summary['recall']:.4f}")
    print(f"Test mAP@50        : {summary['mAP50']:.4f}")
    print(f"Test mAP@50-95     : {summary['mAP50-95']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()