# Glomeruli Detection and Exploratory Clustering

End-to-end pipeline for preparing glomerular annotations from whole-slide images (WSIs), training YOLOv8 and U-Net models for glomerular detection and segmentation, running whole-slide inference, and assigning exploratory stain-invariant morphological clusters (`StandardScaler` + `K-Means` at $K \in \{2, 4, 6\}$).

Cluster letters (`A` to the final letter) are ordered on the training set using a descending train-cluster median of `peripheral_clear_fraction` as an exploratory, low-to-high necrosis-compatibility ordering. **This ordering is strictly exploratory and does not represent a measurement of Bowman's space or a validated clinical grade.**

## Clustering workflow

* **Representation & model:** `StandardScaler` -> `KMeans` (`seed=42`, `n_init=50`) evaluated at `K = 2, 4, 6` on 7 stain-invariant morphological descriptors (`glom_area_um2`, `glom_perimeter_um`, `circularity`, `solidity`, `extent`, `aspect_ratio`, `stain_heterogeneity`), configured in `configs/clustering_config.yaml` and `configs/clustering_k{2,4,6}.yaml` (`feature_type: stain_invariant_morph`, `manifold.method: none`).
* **Frozen development cohort (`N = 468`):** `411` training glomeruli across 6 WSIs (`RECHERCHE-003`, `004`, `005`, `009`, `010`, `011`) and `57` held-out validation glomeruli (`RECHERCHE-016`).
* **Strict leakage prevention:** `scripts/train_clustering.py` filters the `prepare_dataset.py` manifest before opening any image, mask, or XML file, writes `runs/clustering/train_val_manifest.csv`, and fits all preprocessing and clustering parameters on the training split only. Validation receives frozen cluster assignments, while held-out test WSIs (`RECHERCHE-015`, `RECHERCHE-017`) are never loaded during clustering training.
* **Cross-slide generalization findings:** Leave-One-Slide-Out (LOSO) evaluation across the 7 development WSIs shows that standard hybrid features (ResNet embeddings + raw color/intensity descriptors) are sensitive to inter-slide staining and illumination differences (for example, `RECHERCHE-016` concentrates almost entirely into a single cluster under frozen `K = 2/4/6` models). Using the 7 stain-invariant morphological descriptors (shape, boundary regularity, and normalized intra-glomerular stain heterogeneity) substantially reduces single-slide collapse in LOSO experiments and yields balanced cluster assignments on `RECHERCHE-016`.
* **Galleries:** Training galleries maintain a `1x2` grid (`K=2`), `2x2` (`K=4`), and `2x3` (`K=6`), prioritizing validation specimens before falling back to training specimens and labeling the source split. Prediction galleries display detections from the selected test WSI only.

## Environment setup

Create the virtual environment, install dependencies, and materialize Git LFS artifacts:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
git lfs pull
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

Fit the frozen stain-invariant morphological `K-Means` models (`K = 2, 4, 6`) on the development cohort:

```bash
./.venv/bin/python scripts/train_clustering.py --config configs/clustering_config.yaml
```

Run whole-slide detection/segmentation inference and assign frozen cluster labels on unseen WSIs (`RECHERCHE-015`, `RECHERCHE-017`):

```bash
./.venv/bin/python scripts/run_wsi_inference.py --slide-id RECHERCHE-015
./.venv/bin/python scripts/predict_clustering.py --slide-id RECHERCHE-015
```

Run the automated unit and integration test suite:

```bash
./.venv/bin/python -m unittest discover -s tests
```

The prepared manifest and split configuration are tracked in Git; generated image/mask tiles, raw WSIs, local checkpoints, caches, and `to_cleanup/` are ignored. Final stain-invariant clustering run outputs are stored under `runs/clustering/`, and large serialized artifacts are configured for Git LFS in `.gitattributes`.

## Repository layout

```text
configs/                        Dataset, detector, segmenter, and stain-invariant clustering configurations
docs/                           Technical report and methodological documentation
scripts/                        prepare_dataset, train_yolo, train_unet,
                                train_clustering, run_wsi_inference,
                                predict_clustering
src/clustering/                 Cohort audit, feature extraction, modeling, evaluation, and reporting
src/inference/                  WSI sliding-window inference and structured exports
src/models/                     U-Net architecture, losses, and trainer
src/preprocessing/              XML parsing, tissue detection, and patch extraction
runs/clustering/clustering_k*/  Frozen stain-invariant K-Means models, metrics, and per-split reports
runs/clustering_predictions/    Frozen cluster assignments and reports on unseen test WSIs
runs/wsi_inference/             Whole-slide detection/segmentation inference outputs
tests/                          Automated unit and integration tests
to_cleanup/                     Recoverable local archive (excluded from Git)
```
