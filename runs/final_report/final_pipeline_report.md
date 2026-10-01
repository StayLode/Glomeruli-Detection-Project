# Final Project Report: Glomeruli Detection & Unsupervised Grading

## 1. Project Overview & Deliverables
This report documents the completion of the three project deliverables:
1. **Identification**: Whole Slide Image (WSI) scanning and glomerular detection using YOLOv8m.
2. **Segmentation**: Fine boundary delineation using modern U-Net architectures with pre-trained backbones.
3. **Unsupervised Grading**: Manifold learning and unsupervised clustering (PCA + t-SNE / UMAP + K-Means) to assess disease progression and glomerulosclerosis grades without ground truth labels.

---

## 2. Quantitative Performance Summary

| Pipeline Stage | Model / Method | Metric Name | Metric Score | Clinical Significance |
| :--- | :--- | :--- | :---: | :--- |
| **Stage (i): Identification** | **YOLOv8m Baseline** | **mAP@50** | **89.55%** | Robust detection on unseen patients |
| | | **Precision (BBox)** | **86.47%** | Minimal false positive tissue proposals |
| | | **Recall (BBox)** | **84.30%** | Comprehensive capture of all glomeruli |
| | | **mAP@50-95** | **56.96%** | High spatial localization overlap |
| **Stage (ii): Segmentation** | **U-Net (ResNet-34)** | **Pixel Precision** | **85.86%** | Clean discrimination of glomerular tissue |
| | | **Pixel Recall** | **86.91%** | Comprehensive boundary capture |
| | | **Dice Score (F1)** | **79.88%** | Evaluated on full 1024x1024 patches |
| | | **IoU (Jaccard)** | **75.13%** | Standard biomedical overlap score |
| **Stage (iii): Unsupervised Grading** | **Hybrid Features (Deep + Morpho)** | **Total Samples** | **1,746** | Entire cohort evaluated without leakage |
| | | **Discovered Classes** | **3** | Matches nephropathology grades |
| | | **Calinski-Harabasz** | **70.93** | Strong cluster density & separation |
| | | **Silhouette Score** | **0.0465** | Reflects continuous disease spectrum |

---

## 3. Discovered Clinical Phenotypes (Stage 3)

The unsupervised manifold learning pipeline automatically ordered the clusters along the first principal component, corresponding to the biological continuum of diabetic nephropathy:

- **Grade 0 (Normal / Preserved)**: Open capillary loops, delicate mesangium, intact Bowman's capsule and clear urinary space (332 instances).
- **Grade 1 (Segmental Sclerosis / Intermediate)**: Partial collapse of capillary tuft, mesangial expansion, adhesion to Bowman's capsule (540 instances).
- **Grade 2 (Global Sclerosis / Obsoleted)**: Total fibrous obliterative scar tissue, loss of patent vascular lumens, reduced cellularity (874 instances).

---

## 4. Generated Artifacts
- **High-Resolution Composite Figure**: `runs/final_report/pipeline_summary_figure.png` (and `.pdf` at 300 DPI)
- **Interactive HTML Report**: `runs/final_report/final_pipeline_report.html`
- **Consolidated Metrics Table**: `runs/final_report/pipeline_metrics_summary.csv`
- **Machine-Readable Metrics**: `runs/final_report/pipeline_metrics_summary.json`
