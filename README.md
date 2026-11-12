# Glomeruli Detection and Exploratory Clustering

Pipeline for preparing glomerular annotations, training YOLO and U-Net models,
running whole-slide inference, and assigning exploratory PCA20 clusters.
Letters are ordered from A to the final letter using a train-fitted image proxy
as a low-to-high necrosis-compatibility ordering. This is exploratory, not a
validated clinical grade. The gallery keeps a 1×2 grid for K=2, 2×2 for K=4,
and 2×3 for K=6, with captions under the images. Training galleries select
validation specimens first, falling back to train, and mark the source split.
Prediction galleries contain only detections from the selected test WSI.

## Clustering workflow

The final clustering representation is classic PCA20 + K-Means. The frozen
development cohort contains 468 annotations: 411 train and 57 validation
(RECHERCHE-016), with the existing selection thresholds. Model fitting and all
learned feature transforms use train only. Validation receives frozen assignments.
`train_clustering.py` filters the prepare_dataset CSV before opening any image,
mask or XML, writes `runs/clustering/train_val_manifest.csv`, and builds a scoped
`cohort_train_val/` with separate feature caches and resampling plan. It never
loads or assigns the test glomeruli from RECHERCHE-015/017.
XML annotations provide locations/contours, not clinical cluster labels.

Run all three K values:

```bash
./.venv/bin/python scripts/train_clustering.py \
  --config configs/clustering_config.yaml --device cpu
```

Use `--device cuda` on a machine where PyTorch has a working CUDA device. To
archive the current results in the ignored `to_cleanup/` folder and fit again:

```bash
./.venv/bin/python scripts/train_clustering.py \
  --config configs/clustering_config.yaml --device cuda --overwrite --rebuild-cohort
```

Each run is saved under `runs/clustering/clustering_k2`, `clustering_k4`, or
`clustering_k6`. Each folder contains the serialized assignment pipeline,
assignments and metrics for train/validation, a gallery, a projection, and input
and output hashes. Combined summaries are in `runs/clustering/summary.csv`,
`runs/clustering/report.md`, and `runs/clustering/k_comparison.csv`. A readable
historical report from the former train/validation/test assignment workflow is in
[`docs/clustering_final_pca20.md`](docs/clustering_final_pca20.md).
The current method and a critical discussion of the saved train/validation and
WSI inference results are documented in Italian in
[`docs/clustering_metodo_e_lettura_critica.md`](docs/clustering_metodo_e_lettura_critica.md).
Results under the new protocol are generated when you run the commands above;
earlier pipelines are rejected by prediction until regenerated.

## WSI inference and cluster assignment

Run WSI inference with both YOLO and U-Net. The inference command exports
post-NMS detections with mask provenance to JSON, plus detection/segmentation
metrics against the location/contour XML when available:

```bash
./.venv/bin/python scripts/run_wsi_inference.py \
  --all_test_slides \
  --weights runs/yolo/yolov8m_20x_baseline/weights/best.pt \
  --unet_weights runs/unet/unet_20x_resnet34/weights/best.pt \
  --output_dir runs/wsi_inference
```

Then assign each valid U-Net-segmented detection with all three saved PCA20
pipelines. `predict_clustering` processes only RECHERCHE-015/017 and never
refits on WSI predictions. Batch mode requires both test exports; it ignores
exports belonging to train or validation:

```bash
./.venv/bin/python scripts/predict_clustering.py \
  --inference-dir runs/wsi_inference \
  --models-root runs/clustering \
  --output-root runs/clustering_predictions
```

The prediction report includes assigned cluster counts, internal morphology
metrics, excluded detections and the WSI detection/segmentation metrics. It
does not report clinical cluster accuracy because those labels are not present
in the XML files.

## Dataset preparation and detector training

The raw slide and annotation files belong in the ignored `glomeruli_grading/`
directory. Prepare patch data using `configs/dataset_config.yaml`:

```bash
./.venv/bin/python scripts/prepare_dataset.py
```

Train/evaluate YOLO and U-Net with their respective configuration files:

```bash
./.venv/bin/python scripts/train_yolo.py --config configs/yolo_config.yaml
./.venv/bin/python scripts/train_unet.py --config configs/unet_config.yaml
```

The prepared manifest and split configuration are tracked; generated image and
mask data, raw slides, local checkpoints, caches and `to_cleanup/` are ignored.
Final PCA20 run outputs are included in the repository and large serialized
models are configured for Git LFS in `.gitattributes`.

## Repository layout

```text
configs/                    Dataset, detector and PCA20 configurations
docs/                       Final clustering report and project references
scripts/                    prepare_dataset, train_yolo, train_unet,
                            train_clustering, run_wsi_inference,
                            predict_clustering
src/clustering/             Cohort, feature, model, evaluation and reporting code
src/inference/              WSI inference and machine-readable exports
runs/clustering/clustering_k*/  Final model and per-split results
runs/wsi_inference/         Detection/segmentation inference exports
to_cleanup/                 Recoverable local archive, excluded from Git
```

Before publishing, install Git LFS and fetch/track the configured large files
if your GitHub workflow uses the LFS attributes. See
[`docs/publishing.md`](docs/publishing.md).
