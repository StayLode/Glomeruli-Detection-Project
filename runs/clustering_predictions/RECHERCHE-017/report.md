 **Clustering delle predizioni YOLO + U-Net** 

Experimental ordering only: A has the highest train-cluster median peripheral clear fraction and the final letter has the lowest. The letters are shown as a low-to-high necrosis-compatibility proxy for visualization; this is not a measurement of Bowman's space or a validated clinical grade.

Glomeruli assegnati: 56/68; esclusioni: 12. Nessun fit sulle predizioni WSI.

| Modello | Split/slide | N | Conteggi lettere | Silhouette comune | Mediana distanza | Criteri train |
| --- | --- | --- | --- | --- | --- | --- |
| clustering_k2 | RECHERCHE-017 | 56 | {'A': 46, 'B': 10} | 0.5320 | 2.2983 | — |
| clustering_k4 | RECHERCHE-017 | 56 | {'A': 2, 'B': 19, 'C': 25, 'D': 10} | 0.1227 | 1.8538 | — |
| clustering_k6 | RECHERCHE-017 | 56 | {'A': 1, 'B': 13, 'C': 21, 'D': 8, 'E': 10, 'F': 3} | 0.1455 | 1.8352 | — |

Le lettere vengono ordinate sul train e mantenute fisse negli altri split. Non sono una classificazione clinica validata.
I modelli ai diversi K sono indipendenti: lettere uguali non garantiscono identici sottogruppi fra K.
Un solo cluster occupato rende silhouette/DB/CH non valutabili; una sola slide rende Cramér's V non stimabile.
Le distanze appartengono allo spazio del modello e non sono probabilità. I cinque criteri sono criteri sul train.
Gli XML descrivono posizioni/contorni, non classi di salute o necrosi. Le WSI test vengono elaborate da run_wsi_inference e predict_clustering.
