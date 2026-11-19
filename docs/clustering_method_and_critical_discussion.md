# Clustering Method and Critical Discussion of Results

## 1. Methodological evolution: from the PCA20 baseline to stain-invariant clustering
The initial baseline (`PCA20 + K-Means` on ResNet34 embeddings + 10 morphological/photometric features) showed a strong *batch effect* associated with H&E/PAS staining intensity across slides (Cramér's $V = 0.4488$ at $K=4$), causing near-total collapse of out-of-sample assignments on the validation slide `RECHERCHE-016` (57/57 assigned to A at $K=2$; 55/57 to D at $K=4$; 51/57 to F at $K=6$).

The final pipeline uses a **stain-invariant** representation (`feature_type: stain_invariant_morph`, `input_scaling: standard`, `manifold.method: none`, `K-Means`):
- **7 native structural/morphological features** (`glom_area_um2`, `glom_perimeter_um`, `circularity`, `solidity`, `extent`, `aspect_ratio`, `stain_heterogeneity`), excluding the 3 confounding photometric features (`mean_intensity`, `mean_optical_density`, `red_green_ratio`) and deep embeddings that are not stain-normalized.
- **Leakage-prevention protocol:** `StandardScaler` and the `K-Means` centroids are fit exclusively on the **411 training glomeruli** (6 WSIs) and applied without refitting to the **57 validation glomeruli** (`RECHERCHE-016`) and to U-Net predictions on the test WSIs (`RECHERCHE-015`, `RECHERCHE-017`).
- **Exploratory A → F ordering:** the letters are ordered in descending order of the median `peripheral_clear_fraction`, computed on the training set alone (an exploratory morphological proxy compatible with the degree of sclerosis/necrosis, not a clinically validated diagnosis).

## 2. Leave-One-Slide-Out validation (LOSO, N=468)
Evaluating the model on all 7 development patients (fit on 6 WSIs, assignment on 1 held-out WSI):
- **No inter-slide collapse (0/7 slides >80%):** at $K=4$, the maximum out-of-sample share assigned to any single cluster falls from 96.49% (baseline) to **59.65%** (mean **48.19%**); at $K=6$, the worst case is **54.39%** (mean **43.92%**).
- **Quality criteria met:** $K=4$ passes **5/5 official criteria** ($\text{ARI}_{\text{sub80}} = 0.9335 \ge 0.75$, $\text{Jaccard}_{\min} = 0.7808 \ge 0.60$, $z\text{-score} = 8.64 \ge 2.0$, Cramér's $V = 0.2472 \le 0.30$, $\text{max\_dom\_slide} = 38.71\% \le 80\%$).
- **Comparison of $K \in \{2, 4, 6\}$:** $K=4$ provides the best balance between statistical stability and phenotypic resolution; $K=6$ remains available as a finer exploratory partition (6 stages of the FP03 trace, 4/5 criteria), while $K=2$ isolates only a small outlier tail (7/7 LOSO collapses).
