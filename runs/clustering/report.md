 **Stain-Invariant Morphological Clustering: confronto K=2/4/6** 

Experimental ordering only: A has the highest train-cluster median peripheral clear fraction and the final letter has the lowest. The letters are shown as a low-to-high necrosis-compatibility proxy for visualization; this is not a measurement of Bowman's space or a validated clinical grade.

Fit: sei slide train; assegnazioni validation: RECHERCHE-016. Le WSI 015/017 sono riservate a inference e predict_clustering. Gallery: validation, poi train.

| Modello | Split/slide | N | Conteggi lettere | Silhouette comune | Mediana distanza | Criteri train |
| --- | --- | --- | --- | --- | --- | --- |
| clustering_k2 | train | 411 | {'A': 379, 'B': 32} | 0.4593 | 1.8497 | 5/5 |
| clustering_k2 | val | 57 | {'A': 57, 'B': 0} | non valutabile | 1.7097 | None |
| clustering_k4 | train | 411 | {'A': 97, 'B': 101, 'D': 33, 'C': 180} | 0.1736 | 1.3948 | 3/5 |
| clustering_k4 | val | 57 | {'A': 3, 'B': 20, 'D': 0, 'C': 34} | -0.0076 | 1.3618 | None |
| clustering_k6 | train | 411 | {'C': 121, 'B': 71, 'A': 69, 'D': 17, 'E': 117, 'F': 16} | 0.1351 | 1.2741 | 2/5 |
| clustering_k6 | val | 57 | {'C': 31, 'B': 11, 'A': 0, 'D': 0, 'E': 15, 'F': 0} | 0.1163 | 1.2924 | None |

Le lettere vengono ordinate sul train e mantenute fisse negli altri split. Non sono una classificazione clinica validata.
I modelli ai diversi K sono indipendenti: lettere uguali non garantiscono identici sottogruppi fra K.
Un solo cluster occupato rende silhouette/DB/CH non valutabili; una sola slide rende Cramér's V non stimabile.
Le distanze appartengono allo spazio del modello e non sono probabilità. I cinque criteri sono criteri sul train.
Gli XML descrivono posizioni/contorni, non classi di salute o necrosi. Le WSI test vengono elaborate da run_wsi_inference e predict_clustering.
