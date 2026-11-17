**Clustering delle predizioni YOLO + U-Net**

Experimental ordering only: A has the highest train-cluster median peripheral clear fraction and the final letter has the lowest. The letters are shown as a low-to-high necrosis-compatibility proxy for visualization; this is not a measurement of Bowman's space or a validated clinical grade.

Glomeruli assegnati: 56/68; esclusioni: 12. Nessun fit sulle predizioni WSI.

| Modello | Split/slide | N | Conteggi lettere | Silhouette comune | Mediana distanza | Criteri train |
| --- | --- | --- | --- | --- | --- | --- |
| clustering_k2 | RECHERCHE-017 | 56 | {'A': 46, 'B': 10} | 0.5320 | 1.1423 | — |
| clustering_k4 | RECHERCHE-017 | 56 | {'A': 9, 'B': 1, 'C': 10, 'D': 36} | 0.1002 | 1.0230 | — |
| clustering_k6 | RECHERCHE-017 | 56 | {'A': 5, 'B': 0, 'C': 12, 'D': 9, 'E': 10, 'F': 20} | 0.0673 | 1.0027 | — |

Le lettere vengono ordinate sul train e mantenute fisse negli altri split. Non sono una classificazione clinica validata.
I modelli ai diversi K sono indipendenti: lettere uguali non garantiscono identici sottogruppi fra K.
Un solo cluster occupato rende silhouette/DB/CH non valutabili; una sola slide rende Cramér's V non stimabile.
Le distanze appartengono allo spazio del modello e non sono probabilità. I cinque criteri sono criteri sul train.
Gli XML descrivono posizioni/contorni, non classi di salute o necrosi. Le WSI test vengono elaborate da run_wsi_inference e predict_clustering.
