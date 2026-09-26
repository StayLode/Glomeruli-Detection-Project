# Glomeruli Detection & Unsupervised Grading Pipeline

> **An End-to-End Deep Learning Framework for Whole Slide Biopsies in Diabetic Kidney Disease**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-m-green.svg)](https://docs.ultralytics.com/)
[![OpenSlide](https://img.shields.io/badge/OpenSlide-WSI-brightgreen.svg)](https://openslide.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Clinical Motivation & Challenges](#clinical-motivation--challenges)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Dataset & Zero-Leakage Policy](#dataset--zero-leakage-policy)
5. [Benchmark Performance Summary](#benchmark-performance-summary)
6. [Repository Structure](#repository-structure)
7. [Environment Setup & Installation](#environment-setup--installation)
8. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
9. [Artifacts & Clinical Deliverables](#artifacts--clinical-deliverables)

---

## Project Overview

Accurate identification, boundary delineation, and morphological subtyping of **renal glomeruli** are essential steps for diagnosing and staging **Diabetic Kidney Disease (DKD)**.

This repository provides a modular, reproducible, end-to-end deep learning pipeline that addresses the three core requirements of the project:

1. **Identification (Object Detection)**: Rapid localization of all glomeruli across multi-gigapixel Whole Slide Images (WSIs) using **YOLOv8m**.
2. **Segmentation (Semantic Boundary Delineation)**: Fine pixel-level segmentation of the glomerular tuft and Bowman's capsule using a **U-Net with a pre-trained ResNet-34 encoder** trained with a composite loss function (`ComboLoss`).
3. **Unsupervised Grading (Manifold Learning & Subtyping)**: Because clinical datasets lack ground-truth severity annotations, the framework extracts **hybrid representations** (deep CNN features + 10 quantitative morphological biomarkers) and projects them into low-dimensional manifolds (PCA + t-SNE / UMAP). It then clusters glomeruli into progression grades (*Normal / Preserved* $\rightarrow$ *Segmental Sclerosis* $\rightarrow$ *Global Sclerosis / Obsoleted*).

---

## Clinical Motivation & Challenges

| Challenge | Clinical Context | Technical Solution in Pipeline |
| :--- | :--- | :--- |
| **Gigapixel Scale** | Biopsy slides (`.svs`) are typically $100,000 \times 70,000$ pixels; ~95% is empty background glass. | Automated thumbnail-based tissue segmentation (HSV thresholding + morphology) skips background glass. |
| **Class Imbalance** | Glomerular tissue occupies less than 2% of kidney biopsy area. | Balanced tiling at 20x ($1024 \times 1024$ px) with spatial index query (`STRtree`) and a 1:1 negative patch subsampling ratio. |
| **Lack of Grade Labels** | Clinical ground truth only annotates *where* a glomerulus is, not its degree of sclerosis. | Unsupervised clustering on hybrid features, automatically ordered along the first principal component (PC1) to match disease progression. |
| **Whole Slide Translation** | Models trained on local patches must translate back to whole slides for clinical use. | WSI Inference engine with coordinate projection to Level 0 (40x), Global Non-Maximum Suppression (NMS), and ASAP-compatible XML export. |

---

## Pipeline Architecture

```
                       Whole Slide Image (.svs) + ASAP Annotations (.xml)
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 0: Preprocessing & Smart Tiling (prepare_dataset.py)                                 │
│  - HSV-based tissue masking (filters out ~95% background glass)                           │
│  - 20x magnification downsampling (0.50 µm/px), 1024x1024 px patches, 768 px stride      │
│  - Spatial indexing (STRtree) matching patches with XML polygons                         │
│  - Dual export: YOLO bounding box txt + U-Net binary PNG masks + 1:1 negative sampling    │
│  - Patient-level split: 6 train, 1 validation, 2 test slides                              │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
┌──────────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│ STAGE 1: Identification (train_yolo.py)  │    │ STAGE 2: Segmentation (train_unet.py)    │
│  - YOLOv8m (1024x1024 resolution)        │    │  - U-Net with pre-trained ResNet-34      │
│  - Histology-tailored stain augmentations│    │  - ComboLoss: BCE + SoftDice + FocalLoss │
│  - Fast whole-slide screening            │    │  - Mixed Precision (AMP), AdamW, Cosine  │
│  - Test mAP@50: 89.55%                   │    │  - Test Dice: 79.88% | Pixel Prec: 85.86%│
└─────────────────────┬────────────────────┘    └────────────────────┬─────────────────────┘
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Unsupervised Manifold Learning & Disease Grading (run_clustering.py)             │
│  - Extracts standardized context-preserving crops (512x512, 25% parenchymal margin)       │
│  - Hybrid feature vector: 512-dim ResNet-34 embeddings + 10 histological biomarkers       │
│  - Dimensionality reduction: PCA (95% variance) + 2D Manifold projection (t-SNE / UMAP)   │
│  - Optimal K evaluation (Elbow & Silhouette curves across K=2..6)                         │
│  - Monotonic ordering along PC1: Grade 0 (Normal) -> Grade 1 (Segmental) -> Grade 2 (Sclerotic) │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
┌──────────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│ STAGE 4: Full WSI Stitching & Inference  │    │ STAGE 5: Executive Clinical Reporting    │
│ (run_wsi_inference.py)                   │    │ (generate_report.py)                     │
│  - Tiled GPU inference over tissue areas │    │  - Publication 6-panel figure (PNG/PDF)  │
│  - Projection back to native Level 0     │    │  - Self-contained interactive HTML report│
│  - Global NMS across patch boundaries    │    │  - Consolidated CSV & JSON metric tables │
│  - Export to ASAP-compliant XML          │    │  - Executive Markdown summary            │
└──────────────────────────────────────────┘    └──────────────────────────────────────────┘
```

---

## Dataset & Zero-Leakage Policy

The raw dataset contains 9 renal biopsy Whole Slide Images in Aperio `.svs` format and corresponding ground-truth annotations in ASAP `.xml` format:

```
glomeruli_grading/
├── RECHERCHE-003.{svs,xml}   ├── RECHERCHE-010.{svs,xml}   ├── RECHERCHE-016.{svs,xml}
├── RECHERCHE-004.{svs,xml}   ├── RECHERCHE-011.{svs,xml}   ├── RECHERCHE-017.{svs,xml}
├── RECHERCHE-005.{svs,xml}   ├── RECHERCHE-015.{svs,xml}
└── RECHERCHE-009.{svs,xml}
```

### Zero Data Leakage Guarantee
Splits are strictly established at the **patient / slide level** rather than random patch sampling. Patches from the same patient never appear across different splits:

| Split | Slides Assigned | Total Patches | Positive Patches | Negative Patches | Total Glomeruli |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train** | `RECHERCHE-003, 004, 005, 009, 010, 011` | 1,650 | 878 | 772 | 1,239 |
| **Val** | `RECHERCHE-016` | 213 | 109 | 104 | 154 |
| **Test** | `RECHERCHE-015, 017` | 244 | 122 | 122 | 172 |
| **Total** | **9 Slides** | **2,107** | **1,109** | **998** | **1,565** |

### Patch Extraction Parameters
* **Target Magnification**: 20x (downsampled from native 40x, $\approx 0.5038\ \mu\text{m/px}$).
* **Patch Size**: $1024 \times 1024$ pixels ($\approx 516 \times 516\ \mu\text{m}$ tissue field of view).
* **Stride**: $768$ pixels (25% boundary overlap to prevent truncating glomeruli).
* **Tissue Filter**: Discards patches with less than 15% tissue content.
* **Negative Ratio**: Subsampled to 1.0 (equal number of negative and positive patches per slide).

---

## Benchmark Performance Summary

All models were evaluated on the held-out test cohort (`RECHERCHE-015` and `RECHERCHE-017`):

### 1. Stage (i) — Glomerular Identification (YOLOv8m)
| Metric | Test Set Score | Description |
| :--- | :---: | :--- |
| **mAP @ IoU 0.50** | **89.55%** | Primary detection accuracy across unseen patient biopsies |
| **Precision** | **86.47%** | Minimal false positive proposals on background tubules |
| **Recall** | **84.30%** | Comprehensive detection of healthy and diseased glomeruli |
| **mAP @ IoU 0.50:0.95** | **56.96%** | High spatial localization accuracy under strict IoU thresholds |

### 2. Stage (ii) — Semantic Segmentation (U-Net)
| Model Variant | Dice (F1) | IoU (Jaccard) | Pixel Precision | Pixel Recall | Extracted Box mAP@50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **U-Net Scratch Baseline** | 48.41% | 43.93% | 76.84% | 58.12% | 21.10% |
| **U-Net ResNet-34 (Pre-trained)** | **79.88%** | **75.13%** | **85.86%** | **86.91%** | **69.80%** |

*Note: In addition to pixel metrics, segmentation masks were converted to bounding boxes using contour analysis to enable a direct, apples-to-apples comparison with YOLO.*

### 3. Stage (iii) — Unsupervised Manifold Learning & Disease Grading
* Evaluated on **1,746 segmented glomeruli** extracted from all biopsies.
* Feature space: **Hybrid** (512 deep ResNet features + 10 histological biomarkers: area, circularity, solidity, optical density, red/green ratio, stain heterogeneity).
* Dimensionality reduction: PCA ($95\%$ variance, 244 components) $\rightarrow$ 2D t-SNE / UMAP.
* Clusters automatically ordered along PC1 to reflect disease progression:

| Discovered Class | Clinical Interpretation | Count | Mean Area ($\text{px}^2$) | Mean Circularity | Optical Density |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Grade 0** | **Normal / Preserved**: Open capillary lumens, clear Bowman space | 332 | $\approx 68,400$ | 0.72 | 0.118 |
| **Grade 1** | **Segmental Sclerosis**: Mesangial expansion, focal tuft collapse | 540 | $\approx 82,100$ | 0.68 | 0.095 |
| **Grade 2** | **Global Sclerosis**: Fibrous obliterative scarring, obsolescent tuft | 874 | $\approx 94,300$ | 0.64 | 0.076 |

*(Alternative 6-class grading configuration is also available in `configs/clustering_config.yaml` matching classes A through F of the reference literature).*

---

## Repository Structure

```
Glomeruli-Detection-Project/
├── configs/                      # Centralized configuration YAMLs
│   ├── dataset_config.yaml       # Tiling, resolution, and slide split definitions
│   ├── yolo_config.yaml          # YOLOv8m training hyperparameters and augmentations
│   ├── unet_config.yaml          # U-Net architecture, loss weights, and scheduler
│   └── clustering_config.yaml    # Feature extraction, PCA, t-SNE, and K-Means settings
├── dataset/                      # Generated by scripts/prepare_dataset.py
│   ├── manifest.csv              # Master index with metadata for every patch
│   ├── dataset_summary.json      # Distribution statistics per split and slide
│   ├── inspection_grid.png       # Quality control visual verification grid
│   ├── yolo/                     # YOLO dataset: images/, labels/, data.yaml
│   └── segmentation/             # Segmentation dataset: images/, masks/
├── docs/                         # Clinical papers, project proposals, and presentations
├── glomeruli_grading/            # Raw clinical WSIs (.svs) and annotations (.xml)
├── requirements.txt              # Complete Python dependency list
├── runs/                         # Model checkpoints, evaluation logs, and visual outputs
│   ├── detect/runs/yolo/         # Trained YOLO weights (best.pt) and predictions
│   ├── unet/                     # Trained U-Net checkpoints (best.pt) and test grids
│   ├── clustering/               # Manifold scatter plots, elbow curves, galleries
│   └── final_report/             # Final 6-panel summary figure, HTML and CSV reports
├── scripts/                      # Independent CLI execution entrypoints
│   ├── prepare_dataset.py        # Step 1: Preprocessing & dual dataset generation
│   ├── train_yolo.py             # Step 2A: YOLO training
│   ├── evaluate_yolo.py          # Step 2A: YOLO checkpoint evaluation
│   ├── train_unet.py             # Step 2B: U-Net training
│   ├── evaluate_unet.py          # Step 2B: U-Net test set evaluation
│   ├── run_clustering.py         # Step 3: Unsupervised feature extraction & clustering
│   ├── run_wsi_inference.py      # Step 4: Full slide inference and ASAP XML export
│   └── generate_report.py        # Step 5: Master multi-stage report generation
└── src/                          # Reusable core modules
    ├── utils/                    # XML parser for ASAP annotations
    ├── preprocessing/            # TissueDetector and PatchExtractor
    ├── models/                   # YOLO wrapper, U-Net, ComboLoss, mask-to-bbox converter
    ├── clustering/               # Feature extractor, manifold clusterer, visualizer
    ├── inference/                # Whole slide inference engine with global NMS
    └── visualization/            # Verification grids and pipeline report generators
```

---

## Environment Setup & Installation

### 1. System Dependencies (OpenSlide C Library)
The pipeline requires OpenSlide to process `.svs` files. Install it via your package manager:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y libopenslide0 libopenslide-dev

# macOS (Homebrew)
brew install openslide
```

### 2. Python Environment Setup
We recommend Python 3.10 or 3.11 with a virtual environment:

```bash
# Clone the repository
git clone https://github.com/your-username/Glomeruli-Detection-Project.git
cd Glomeruli-Detection-Project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step-by-Step Execution Guide

Every script can be run directly from the root directory using the prepared YAML configurations or CLI flags.

### Step 1: Preprocess WSIs & Build Dataset
Processes raw `.svs` and `.xml` files, filters background glass, performs 20x tiling, and creates dual YOLO and segmentation sets:
```bash
python scripts/prepare_dataset.py --config configs/dataset_config.yaml --num_samples 6
```
*Outputs*: `dataset/yolo/`, `dataset/segmentation/`, `dataset/manifest.csv`, and `dataset/inspection_grid.png`.

---

### Step 2: Glomeruli Identification (YOLOv8m)

#### Train YOLO:
```bash
python scripts/train_yolo.py --config configs/yolo_config.yaml
```
*(CLI overrides such as `--batch 16`, `--epochs 60`, or `--model yolov8x.pt` are supported).*

#### Evaluate YOLO Checkpoint on Test Set:
```bash
python scripts/evaluate_yolo.py \
    --weights runs/detect/runs/yolo/yolov8m_20x_baseline/weights/best.pt \
    --split test \
    --save_predictions
```
*Outputs*: Confusion metrics in console, `test_metrics.json`, and visual overlays in `runs/detect/runs/yolo/predictions_test/`.

---

### Step 3: Semantic Segmentation (U-Net)

#### Train U-Net (ResNet-34 + ComboLoss):
```bash
python scripts/train_unet.py --config configs/unet_config.yaml
```
*(CLI overrides such as `--encoder resnet50`, `--loss combo`, or `--batch_size 16` are supported).*

#### Evaluate U-Net Checkpoint on Test Set:
```bash
python scripts/evaluate_unet.py \
    --checkpoint runs/unet/unet_20x_resnet34/weights/best.pt \
    --config configs/unet_config.yaml \
    --threshold 0.5
```
*Outputs*: Pixel-level metrics (Dice, IoU), contour-derived detection metrics (AP50), and qualitative comparison grid `test_predictions_grid.png`.

---

### Step 4: Unsupervised Manifold Learning & Disease Grading
Extracts context-aware crops, computes deep embeddings + morphological features, projects via t-SNE, and identifies clinical grades:

```bash
# 3-Grade Clinical Progression (Normal, Segmental, Sclerotic):
python scripts/run_clustering.py --config configs/clustering_config.yaml --n_clusters 3

# 6-Grade Progression (A through F):
python scripts/run_clustering.py --config configs/clustering_config.yaml --n_clusters 6
```
*Outputs saved in `runs/clustering/`:*
* `manifold_clusters.png`: 2D t-SNE projection colored by progression grade.
* `optimal_k_analysis.png`: Elbow inertia and Silhouette curves across $K \in [2, 6]$.
* `morphological_distributions.png`: Boxplots of circularity, area, and optical density across clusters.
* `cluster_gallery.png`: Representative glomeruli images closest to each cluster centroid.
* `cluster_assignments.csv`: Glomerulus-level dataset with assigned classes and extracted features.

---

### Step 5: Full Whole Slide Image (WSI) Inference & ASAP Export
Runs tiled inference across an entire multi-gigapixel slide, resolves patch boundary overlaps via global NMS, and exports pathologist annotations:

```bash
# Evaluate a single slide:
python scripts/run_wsi_inference.py \
    --slide glomeruli_grading/RECHERCHE-015.svs \
    --weights runs/detect/runs/yolo/yolov8m_20x_baseline/weights/best.pt \
    --output_dir runs/wsi_inference

# Process all slides in the test cohort:
python scripts/run_wsi_inference.py \
    --all_test_slides \
    --weights runs/detect/runs/yolo/yolov8m_20x_baseline/weights/best.pt \
    --output_dir runs/wsi_inference
```
*Outputs*:
* `<slide_id>_predicted.xml`: Fully compliant with **ASAP** (Automated Slide Analysis Platform).
* `<slide_id>_overview.png`: Full-slide thumbnail showing predicted boxes (green) vs ground truth (yellow).
* `<slide_id>_metrics.json`: Slide-level True Positives, False Positives, False Negatives, Precision, Recall, and F1.

---

### Step 6: Generate Master Clinical Pipeline Report
Harmonizes metrics across YOLO, U-Net, and Clustering into presentation and publication artifacts:

```bash
python scripts/generate_report.py \
    --yolo_run runs/detect/runs/yolo/yolov8m_20x_baseline \
    --unet_run runs/unet/unet_20x_resnet34 \
    --clustering_run runs/clustering/unsupervised_grading_3classes \
    --output_dir runs/final_report
```
*Outputs generated in `runs/final_report/`:*
* `pipeline_summary_figure.png` / `.pdf`: 300 DPI 6-panel summary figure suitable for presentations or papers.
* `final_pipeline_report.html`: Self-contained interactive report with embedded base64 figures.
* `final_pipeline_report.md`: Executive summary document.
* `pipeline_metrics_summary.csv` / `.json`: Consolidated numerical benchmarks.

---

## Artifacts & Clinical Deliverables

Pre-trained weights and generated artifacts are organized as follows:

| Deliverable | File Path |
| :--- | :--- |
| **YOLOv8m Best Weights** | `runs/detect/runs/yolo/yolov8m_20x_baseline/weights/best.pt` |
| **U-Net ResNet-34 Best Weights** | `runs/unet/unet_20x_resnet34/weights/best.pt` |
| **Interactive HTML Master Report** | `runs/final_report/final_pipeline_report.html` |
| **Publication Composite Figure** | `runs/final_report/pipeline_summary_figure.pdf` |
| **Clustering Morphological Table** | `runs/clustering/unsupervised_grading_3classes/cluster_assignments.csv` |
| **ASAP Reviewer Annotations** | `runs/wsi_inference/*_predicted.xml` |

---

## Hardware Specifications

The pipeline was developed and validated on an **NVIDIA A40 GPU (46 GB VRAM, CUDA 12.8)**. Thanks to configurable batch sizes, Automatic Mixed Precision (AMP), and gradient accumulation in the YAML configuration files, all scripts can be executed on standard consumer GPUs (e.g., RTX 3080/4090) or CPU (`--device cpu`).
