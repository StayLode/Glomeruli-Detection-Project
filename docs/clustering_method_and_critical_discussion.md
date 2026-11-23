# Clustering Method and Critical Discussion of Results

## 1. Methodological evolution: from the PCA20 baseline to stain-reduced morphological clustering

The initial baseline (`PCA20 + K-Means` on ResNet34 embeddings + 10 morphological/photometric features) exhibited a strong *batch effect* associated with H&E/PAS staining intensity and illumination differences across slides (Cramér's $V = 0.4488$ at $K=4$), causing near-total collapse of out-of-sample assignments on the held-out validation slide `RECHERCHE-016` (`57/57` assigned to `A` at $K=2$; `55/57` (`96.49%`) to `D` at $K=4$; `51/57` (`89.47%`) to `F` at $K=6$).

To mitigate slide-level color confounding, the final pipeline (`feature_type: stain_invariant_morph`, `input_scaling: standard`, `manifold.method: none`, `K-Means` with `n_init=20`, `seed=42`) restricts the representation to **7 native descriptors**:

- **6 pure geometric/shape features:** `glom_area_um2`, `glom_perimeter_um`, `circularity`, `solidity`, `extent`, and `aspect_ratio` (computed solely from the binary glomerular mask and calibrated $\mu\text{m}/\text{px}$ spacing, hence strictly invariant to staining and illumination).
- **1 raw intra-mask texture feature:** `stain_heterogeneity`, computed as the unnormalized pixel-level RGB standard deviation inside the glomerular mask (`src/clustering/feature_extractor.py`). The 3 slide-sensitive photometric averages (`mean_intensity`, `mean_optical_density`, `red_green_ratio`) and non-normalized ResNet34 embeddings are excluded.
- **Leakage-prevention protocol:** `StandardScaler` and the `K-Means` centroids are fit exclusively on the **411 training glomeruli** (6 WSIs: `RECHERCHE-003`, `004`, `005`, `009`, `010`, `011`) and applied without refitting to the **57 validation glomeruli** (`RECHERCHE-016`) and to YOLO + U-Net predictions on the held-out test WSIs (`RECHERCHE-015`, `RECHERCHE-017`).
- **Exploratory `A` to final-letter ordering:** cluster letters are assigned in descending order of the training-set cluster median of `peripheral_clear_fraction` (an exploratory morphological proxy compatible with Bowman's capsule / peripheral clearing, **not** a validated clinical necrosis or sclerosis grade).

---

## 2. Exploratory LOSO benchmark vs. delivered K-Means models

A clear distinction must be drawn between the **exploratory Leave-One-Slide-Out (LOSO) model-selection benchmark** (`to_cleanup/evaluate_loso_stain_invariant.py`, evaluated across all 7 development WSIs, $N=468$) and the **final serialized production models** (`runs/clustering/clustering_k{2,4,6}/`, fit with `K-Means` on the 6 training WSIs, $N=411$, and evaluated on `RECHERCHE-016`, $N=57$).

### 2.1 Exploratory LOSO & full-cohort benchmark (`N = 468`, 7 WSIs)

In the 7-fold patient-level LOSO experiment (`k_values = [3, 4]`), fitting on 6 WSIs and predicting on the held-out WSI showed that removing global color/deep features substantially curbs single-slide collapse compared to the `PCA20` hybrid baseline:

- **Best LOSO stability (`stain_inv_gmm_tied`, $K=4$):** a Gaussian Mixture Model with tied covariance achieved **0/7 collapsed folds** (threshold $> 80\%$ in a single cluster), reducing worst-case held-out slide concentration from `96.49%` (baseline) to **59.65%** (mean **48.19%** across the 7 folds) and passing **5/5** full-cohort audit criteria (`ARI_sub80` $= 0.9335 \ge 0.75$, `Jaccard_min` $= 0.7808 \ge 0.60$, silhouette $z$-score $= 8.64 \ge 2.0$, Cramér's $V = 0.2472 \le 0.30$, `max_dom_slide` $= 38.71\% \le 80\%$).
- **Choice of `K-Means` for the delivered pipeline:** for the final reproducible CLI pipeline ($K \in \{2, 4, 6\}$), `K-Means` was retained over `GMM` to enforce deterministic Euclidean Voronoi partitioning, avoid covariance regularization sensitivity across varying $K$, and enable direct comparison across $K=2, 4, 6$ under identical preprocessing.

### 2.2 Delivered `K-Means` production runs (`runs/clustering/`)

The quantitative metrics recorded in `runs/clustering/summary.json` (and `clustering_report.json`) for the delivered `K-Means` models are summarized below:

| Model (`K-Means`) | Split ($N$) | Cluster Counts | Silhouette (Common / Own Space) | Train Criteria Passed | Subsampling `ARI` / `Jaccard_min` |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`clustering_k2`** | Train (`411`) | `A: 379, B: 32` | `0.4593` / `0.5555` | **5/5** | `0.9650` / `0.8235` |
| **`clustering_k2`** | Val `016` (`57`) | `A: 57, B: 0` | *N/A* (1 occupied cluster) | — | — |
| **`clustering_k4`** | Train (`411`) | `A: 97, B: 101, C: 180, D: 33` | `0.1736` / `0.2471` | **3/5** | `0.8316` / `0.5526` |
| **`clustering_k4`** | Val `016` (`57`) | `A: 3, B: 20, C: 34, D: 0` | `-0.0076` / — | — | — |
| **`clustering_k6`** | Train (`411`) | `A: 69, B: 71, C: 121, D: 17, E: 117, F: 16` | `0.1351` / `0.2100` | **2/5** | `0.6859` / `0.0000` |
| **`clustering_k6`** | Val `016` (`57`) | `A: 0, B: 11, C: 31, D: 0, E: 15, F: 0` | `0.1163` / — | — | — |

*(Note: "Common Space" refers to the shared 10-feature standardized evaluation space used in `summary.json` to compare heterogeneous feature sets fairly; "Own Space" refers to the 7-feature standardized model space reported in `clustering_report.json`.)*

On the held-out test slides (`runs/clustering_predictions/`), frozen `K-Means` assignments on YOLO + U-Net detections yield:

- **`RECHERCHE-015` (`36` valid detections out of `41` detected):** $K=2 \rightarrow$ `A: 25, B: 11` ($\text{Sil} = 0.4453$); $K=4 \rightarrow$ `A: 3, B: 8, C: 15, D: 10` ($\text{Sil} = 0.0247$); $K=6 \rightarrow$ `A: 0, B: 5, C: 9, D: 8, E: 12, F: 2` ($\text{Sil} = 0.0441$).
- **`RECHERCHE-017` (`56` valid detections out of `68` detected):** $K=2 \rightarrow$ `A: 46, B: 10` ($\text{Sil} = 0.5320$); $K=4 \rightarrow$ `A: 2, B: 19, C: 25, D: 10` ($\text{Sil} = 0.1227$); $K=6 \rightarrow$ `A: 1, B: 13, C: 21, D: 8, E: 10, F: 3` ($\text{Sil} = 0.1455$).

---

## 3. Critical discussion and methodological limitations

1. **Continuous morphological spectrum vs. discrete clusters:**
   Consistent with the exploratory nature of task *FP03* (which lacks ground-truth necrosis/sclerosis class labels), glomerular morphology varies along a continuous spectrum rather than forming well-separated islands. While $K=2$ achieves the highest internal separation on train (`0.4593`) and passes `5/5` train criteria, it merely separates a small minority of irregular/fragmented contours (`B: 32/411`) from the main population (`A: 379/411`), causing all `57/57` validation glomeruli on `RECHERCHE-016` to fall into `A`.
2. **Trade-off between cluster granularity and out-of-sample separation ($K=4$ and $K=6$):**
   At $K=4$, single-slide collapse on `RECHERCHE-016` is substantially reduced relative to the `PCA20` baseline (maximum cluster share `34/57 = 59.65%` vs. `96.49%`), and all 4 clusters are populated on both unseen test WSIs (`RECHERCHE-015` and `RECHERCHE-017`). However, validation silhouette in the common evaluation space is slightly negative (`-0.0076`) and cluster `D` is empty on `RECHERCHE-016`, confirming weak geometric margin between adjacent morphological phenotypes. At $K=6$, finer partitioning reduces resampling stability (`Jaccard_min = 0.0` in train, `2/5` criteria) and leaves 3 of 6 clusters empty on `RECHERCHE-016`.
3. **Unnormalized `stain_heterogeneity` feature:**
   Although global color averages and deep embeddings were removed, `stain_heterogeneity` is currently implemented as the raw RGB standard deviation inside the mask without stain deconvolution or intensity normalization. Consequently, residual inter-slide contrast or illumination differences can still influence cluster boundaries; future iterations should replace it with a normalized texture descriptor (e.g., coefficient of variation $\sigma/\mu$ or hematoxylin-channel optical density variance after stain normalization).
4. **Lack of clinical ground truth:**
   Because the XML annotations provide contour coordinates without histopathological grading, the `A`–`F` ordering via `peripheral_clear_fraction` must be interpreted strictly as an unsupervised structural stratification tool rather than a diagnostic classifier.
