**PCA20 classica: confronto K=2/4/6**

Experimental ordering only: A has the highest train-cluster median peripheral clear fraction and the final letter has the lowest. The letters are shown as a low-to-high necrosis-compatibility proxy for visualization; this is not a measurement of Bowman's space or a validated clinical grade.

Fit: sei slide train; assegnazioni validation: RECHERCHE-016. Le WSI 015/017 sono riservate a inference e predict_clustering. Gallery: validation, poi train.

| Modello | Split/slide | N | Conteggi lettere | Silhouette comune | Mediana distanza | Criteri train |
| --- | --- | --- | --- | --- | --- | --- |
| clustering_k2 | train | 411 | {'A': 381, 'B': 30} | 0.4678 | 1.0119 | 4/5 |
| clustering_k2 | val | 57 | {'A': 57, 'B': 0} | non valutabile | 1.7433 | None |
| clustering_k4 | train | 411 | {'D': 131, 'A': 118, 'C': 31, 'B': 131} | 0.2322 | 0.8875 | 4/5 |
| clustering_k4 | val | 57 | {'D': 55, 'A': 2, 'C': 0, 'B': 0} | 0.2030 | 1.2781 | None |
| clustering_k6 | train | 411 | {'F': 49, 'C': 114, 'B': 68, 'D': 31, 'E': 89, 'A': 60} | 0.2037 | 0.8188 | 3/5 |
| clustering_k6 | val | 57 | {'F': 51, 'C': 3, 'B': 0, 'D': 0, 'E': 3, 'A': 0} | 0.1030 | 1.0748 | None |

Le lettere vengono ordinate sul train e mantenute fisse negli altri split. Non sono una classificazione clinica validata.
I modelli ai diversi K sono indipendenti: lettere uguali non garantiscono identici sottogruppi fra K.
Un solo cluster occupato rende silhouette/DB/CH non valutabili; una sola slide rende Cramér's V non stimabile.
Le distanze appartengono allo spazio del modello e non sono probabilità. I cinque criteri sono criteri sul train.
Gli XML descrivono posizioni/contorni, non classi di salute o necrosi. Le WSI test vengono elaborate da run_wsi_inference e predict_clustering.
