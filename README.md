# Glomeruli Detection and Exploratory Clustering

End-to-end pipeline for preparing glomerular annotations from whole-slide images (WSIs), training YOLOv8 and U-Net models for glomerular detection and segmentation, running whole-slide inference, and assigning exploratory morphological clusters (`StandardScaler` + `K-Means` at $K \in \{2, 4, 6\}$).

Cluster letters (`A` to the final letter) are ordered on the training set using a descending train-cluster median of `peripheral_clear_fraction` as an exploratory, low-to-high necrosis-compatibility ordering. **This ordering is strictly exploratory and does not represent a measurement of Bowman's space or a validated clinical grade.**

## Clustering workflow

* **Representation & model:** `StandardScaler` -> `KMeans` (`seed=42`, `n_init=20`) evaluated at `K = 2, 4, 6` on 7 descriptors (`glom_area_um2`, `glom_perimeter_um`, `circularity`, `solidity`, `extent`, `aspect_ratio`, `stain_heterogeneity`), configured in `configs/clustering_config.yaml` and `configs/clustering_k{2,4,6}.yaml` (`feature_type: stain_invariant_morph`, `manifold.method: none`). Six features capture pure mask geometry/shape, while `stain_heterogeneity` measures raw (unnormalized) intra-mask RGB standard deviation.
* **Frozen development cohort (`N = 468`):** `411` training glomeruli across 6 WSIs (`RECHERCHE-003`, `004`, `005`, `009`, `010`, `011`) and `57` held-out validation glomeruli (`RECHERCHE-016`).
* **Strict leakage prevention:** `scripts/train_clustering.py` filters the `prepare_dataset.py` manifest before opening any image, mask, or XML file, writes `runs/clustering/train_val_manifest.csv`, and fits all preprocessing and clustering parameters on the training split only. Validation receives frozen cluster assignments, while held-out test WSIs (`RECHERCHE-015`, `RECHERCHE-017`) are never loaded during clustering training.
* **Cross-slide generalization findings:** Leave-One-Slide-Out (LOSO) experiments across the 7 development WSIs (`to_cleanup/evaluate_loso_stain_invariant.py`) show that standard hybrid features (ResNet34 embeddings + raw color/intensity descriptors) suffer from severe inter-slide stain/illumination shift (e.g., `RECHERCHE-016` collapses almost entirely into a single cluster under frozen `K = 2/4/6` models). Removing slide-level color averages and deep embeddings substantially reduces single-slide collapse in `K = 4` and `K = 6` (`max_share` on `RECHERCHE-016` drops from `96.49%` to `59.65%` at `K = 4` and `54.39%` at `K = 6`), though `K = 2` still isolates an outlier tail in train (`379` vs `32`) and assigns all `57/57` validation glomeruli to cluster `A`. See `docs/clustering_method_and_critical_discussion.md` for full metrics and critical analysis.
* **Galleries:** Training galleries maintain a `1x2` grid (`K=2`), `2x2` (`K=4`), and `2x3` (`K=6`), prioritizing validation specimens before falling back to training specimens and labeling the source split. Prediction galleries display detections from the selected test WSI only.

## Environment setup

Create the virtual environment, install pinned dependencies (`scikit-learn==1.6.1`, matching the serialized LFS models), and materialize Git LFS artifacts:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
git lfs pull
```

## Raw WSI dataset placement

Raw `.svs` whole-slide images and `.xml` ground-truth annotations are excluded from Git due to file size. Pre-computed split manifests (`dataset/manifest.csv`), clustering run outputs (`runs/clustering/`), and test inference/clustering artifacts (`runs/wsi_inference/`, `runs/clustering_predictions/`) are already tracked in the repository.

To re-run patch extraction, training, or WSI inference from scratch, place the 9 renal biopsy WSIs (`RECHERCHE-003`, `004`, `005`, `009`, `010`, `011`, `015`, `016`, `017`) and their matching XML files (Bueno et al., *Data in Brief* 2020 / course dataset `FP03`) under `glomeruli_grading/` as configured in `configs/dataset_config.yaml`:

```text
glomeruli_grading/
├── RECHERCHE-003.svs
├── RECHERCHE-003.xml
├── ...
├── RECHERCHE-017.svs
└── RECHERCHE-017.xml
```

## Dataset preparation and detector training

Prepare patches, masks, and split manifests:

```bash
./.venv/bin/python scripts/prepare_dataset.py
```

Train and evaluate the YOLOv8 detector and U-Net segmenter using their respective configuration files:

```bash
./.venv/bin/python scripts/train_yolo.py --config configs/yolo_config.yaml
./.venv/bin/python scripts/train_unet.py --config configs/unet_config.yaml
```

## Clustering training and WSI inference

Fit the frozen morphological `K-Means` models (`K = 2, 4, 6`) on the development cohort:

```bash
./.venv/bin/python scripts/train_clustering.py --config configs/clustering_config.yaml
```

Run whole-slide cascade inference (YOLOv8 detection + U-Net segmentation) and assign frozen cluster labels across both held-out test WSIs (`RECHERCHE-015` and `RECHERCHE-017`):

```bash
./.venv/bin/python scripts/run_wsi_inference.py \
  --all_test_slides \
  --weights runs/yolo/yolov8m_20x_baseline/weights/best.pt \
  --unet_weights runs/unet/unet_20x_resnet34/weights/best.pt \
  --output_dir runs/wsi_inference

./.venv/bin/python scripts/predict_clustering.py \
  --inference-dir runs/wsi_inference
```

To run inference and cluster assignment on a single WSI (e.g., `RECHERCHE-015`):

```bash
./.venv/bin/python scripts/run_wsi_inference.py \
  --slide glomeruli_grading/RECHERCHE-015.svs \
  --weights runs/yolo/yolov8m_20x_baseline/weights/best.pt \
  --unet_weights runs/unet/unet_20x_resnet34/weights/best.pt \
  --output_dir runs/wsi_inference

./.venv/bin/python scripts/predict_clustering.py \
  --predictions runs/wsi_inference/RECHERCHE-015_predictions.json
```

Run the automated unit and integration test suite:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

The prepared manifest and split configuration are tracked in Git; generated image/mask tiles, raw WSIs, local checkpoints, caches, and `to_cleanup/` are ignored. Final clustering run outputs are stored under `runs/clustering/`, and large serialized artifacts are configured for Git LFS in `.gitattributes`.

## Repository layout

```text
configs/                        Dataset, detector, segmenter, and clustering configurations
docs/                           Technical report and methodological documentation
scripts/                        prepare_dataset, train_yolo, train_unet,
                                train_clustering, run_wsi_inference,
                                predict_clustering
src/clustering/                 Cohort audit, feature extraction, modeling, evaluation, and reporting
src/inference/                  WSI sliding-window inference and structured exports
src/models/                     U-Net architecture, losses, and trainer
src/preprocessing/              XML parsing, tissue detection, and patch extraction
runs/clustering/clustering_k*/  Frozen K-Means models, metrics, and per-split reports
runs/clustering_predictions/    Frozen cluster assignments and reports on unseen test WSIs
runs/wsi_inference/             Whole-slide detection/segmentation inference outputs
tests/                          Automated unit and integration tests
to_cleanup/                     Recoverable local archive (excluded from Git)
```
