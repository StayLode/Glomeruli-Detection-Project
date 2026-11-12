#!/usr/bin/env python3
"""Assign frozen PCA20 experimental classes to YOLO + U-Net WSI predictions."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from src.clustering.prediction import run_prediction, select_test_predictions

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictions",type=Path,help="A *_predictions.json post-NMS export")
    group.add_argument("--inference-dir",type=Path,help="Process both test WSI exports (RECHERCHE-015 and RECHERCHE-017)")
    parser.add_argument("--dataset-config",default="configs/dataset_config.yaml")
    parser.add_argument("--models-root",default="runs/clustering")
    parser.add_argument("--output-root",default="runs/clustering_predictions")
    parser.add_argument("--k-values",nargs="+",type=int,choices=[2,4,6],default=[2,4,6])
    parser.add_argument("--slide",type=Path,help="Override slide path; required for legacy XML")
    parser.add_argument("--allow-legacy-unet-xml",action="store_true",help="Explicitly trust a historical U-Net polygon export; rectangles remain excluded")
    args=parser.parse_args()
    try:
        paths=[args.predictions] if args.predictions else select_test_predictions(args.inference_dir,args.dataset_config)
    except ValueError as error:parser.error(str(error))
    if not paths:parser.error("No machine-readable WSI predictions found; run run_wsi_inference.py first")
    for path in paths:print(run_prediction(path,args.models_root,args.output_root,args.k_values,args.slide,args.allow_legacy_unet_xml,args.dataset_config))

if __name__=="__main__":main()
