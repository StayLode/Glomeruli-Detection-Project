### Glomeruli Detection and Exploratory Clustering

End-to-end pipeline for preparing glomerular annotations from whole-slide images (WSIs), training YOLOv8 and U-Net models for glomerular detection and segmentation, running whole-slide inference, and assigning exploratory PCA20 + K-Means morphological clusters[span_5](start_span)[span_5](end_span).

Cluster letters (`A` to the final letter) are ordered on the training set using a descending train-cluster median of `peripheral_clear_fraction` as an exploratory, low-to-high necrosis-compatibility ordering[span_6](start_span)[span_6](end_span). **This ordering is strictly exploratory and does not represent a measurement of Bowman's space or a validated clinical grade[span_7](start_span)[span_7](end_span).**

#### Clustering workflow

* **Representation & model:** Classic `StandardScaler` -> `PCA(n_components=20)` -> `KMeans` evaluated at `K = 2, 4, 6`[span_8](start_span)[span_8](end_span).
* **Frozen development cohort (`N = 468`):** `411` training glomeruli across 6 WSIs (`RECHERCHE-003`, `004`, `005`, `009`, `010`, `011`) and `57` held-out validation glomeruli (`RECHERCHE-016`)[span_9](start_span)[span_9](end_span).
* **Strict leakage prevention:** `scripts/train_clustering.py` filters the `prepare_dataset.py` manifest before opening any image, mask, or XML file, writes `runs/clustering/train_val_manifest.csv`, and fits all preprocessing and clustering parameters on the training split only[span_10](start_span)[span_10](end_span). Validation receives frozen cluster assignments, while held-out test WSIs (`RECHERCHE-015`, `RECHERCHE-017`) are never loaded during clustering training[span_11](start_span)[span_11](end_span).
* **Cross-slide generalization findings:** Leave-One-Slide-Out (LOSO) evaluation across the 7 development WSIs shows that standard hybrid features (ResNet embeddings + raw color/intensity descriptors) are sensitive to inter-slide staining and illumination differences (for example, `RECHERCHE-016` concentrates almost entirely into a single cluster under frozen `K = 2/4/6` models)[span_12](start_span)[span_12](end_span). Stain-invariant morphological descriptors (shape, boundary regularity, and normalized intra-glomerular stain heterogeneity) substantially reduce single-slide collapse in LOSO experiments[span_13](start_span)[span_13](end_span).
* **Galleries:** Training galleries maintain a `1x2` grid (`K=2`), `2x2` (`K=4`), and `2x3` (`K=6`), prioritizing validation specimens before falling back to training specimens and labeling the source split[span_14](start_span)[span_14](end_span). Prediction galleries display detections from the selected test WSI only[span_15](start_span)[span_15](end_span).

#### Dataset preparation and detector training

Prepare patches, masks, and split manifests:

```bash
./.venv/bin/python scripts/prepare_dataset.py
```

Train and evaluate the YOLOv8 detector and U-Net segmenter using their respective configuration files[span_16](start_span)[span_16](end_span):

```bash
./.venv/bin/python scripts/train_yolo.py --config configs/yolo_config.yaml
./.venv/bin/python scripts/train_unet.py --config configs/unet_config.yaml
```

#### Clustering training and WSI inference

Fit the frozen PCA20 + K-Means models (`K = 2, 4, 6`) on the development cohort:

```bash
./.venv/bin/python scripts/train_clustering.py --config configs/clustering_config.yaml
```

Run whole-slide detection/segmentation inference and assign frozen cluster labels on unseen WSIs (`RECHERCHE-015`, `RECHERCHE-017`)[span_17](start_span)[span_17](end_span):

```bash
./.venv/bin/python scripts/run_wsi_inference.py --slide-id RECHERCHE-015
./.venv/bin/python scripts/predict_clustering.py --slide-id RECHERCHE-015
```

The prepared manifest and split configuration are tracked in Git; generated image/mask tiles, raw WSIs, local checkpoints, caches, and `to_cleanup/` are ignored[span_18](start_span)[span_18](end_span). Final PCA20 run outputs are stored under `runs/clustering/`, and large serialized weights are configured for Git LFS in `.gitattributes`[span_19](start_span)[span_19](end_span).

#### Repository layout

```text
configs/                        Dataset, detector, segmenter, and PCA20 configurations
docs/                           Final technical report and methodological documentation
scripts/                        prepare_dataset, train_yolo, train_unet,
                                train_clustering, run_wsi_inference,
                                predict_clustering
src/clustering/                 Cohort audit, feature extraction, modeling, evaluation, and reporting
src/inference/                  WSI sliding-window inference and structured exports
src/models/                     U-Net architecture, losses, and trainer
src/preprocessing/              XML parsing, tissue detection, and patch extraction
runs/clustering/clustering_k*/  Frozen PCA20 models, metrics, and per-split reports
runs/wsi_inference/             Whole-slide detection/segmentation inference outputs
tests/                          Automated unit and integration tests
to_cleanup/                     Recoverable local archive (excluded from Git)
```
