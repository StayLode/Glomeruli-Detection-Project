#!/usr/bin/env python3
"""Fit classic PCA20 K=2/4/6 on train and assign validation; exclude test WSIs."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from src.clustering.training import run_training

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/clustering_config.yaml")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--k-values",nargs="+",type=int,choices=[2,4,6])
    parser.add_argument("--overwrite",action="store_true",help="Archive prior runs in to_cleanup and fit again")
    parser.add_argument("--rebuild-cohort",action="store_true",help="Rebuild annotation cohort from prepare_dataset manifest; requires --overwrite")
    args=parser.parse_args()
    print(run_training(args.config,args.device,args.k_values,args.overwrite,args.rebuild_cohort))

if __name__=="__main__":main()
