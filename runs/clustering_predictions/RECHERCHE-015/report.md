 **Clustering delle predizioni YOLO + U-Net** 

Experimental ordering only: A has the highest train-cluster median peripheral clear fraction and the final letter has the lowest. The letters are shown as a low-to-high necrosis-compatibility proxy for visualization; this is not a measurement of Bowman's space or a validated clinical grade.

Glomeruli assegnati: 36/41; esclusioni: 5. Nessun fit sulle predizioni WSI.

| Modello | Split/slide | N | Conteggi lettere | Silhouette comune | Mediana distanza | Criteri train |
| --- | --- | --- | --- | --- | --- | --- |
| clustering_k2 | RECHERCHE-015 | 36 | {'A': 25, 'B': 11} | 0.4453 | 2.0847 | — |
| clustering_k4 | RECHERCHE-015 | 36 | {'A': 3, 'B': 8, 'C': 15, 'D': 10} | 0.0247 | 1.6342 | — |
| clustering_k6 | RECHERCHE-015 | 36 | {'A': 0, 'B': 5, 'C': 9, 'D': 8, 'E': 12, 'F': 2} | 0.0441 | 1.5227 | — |

Le lettere vengono ordinate sul train e mantenute fisse negli altri split. Non sono una classificazione clinica validata.
I modelli ai diversi K sono indipendenti: lettere uguali non garantiscono identici sottogruppi fra K.
Un solo cluster occupato rende silhouette/DB/CH non valutabili; una sola slide rende Cramér's V non stimabile.
Le distanze appartengono allo spazio del modello e non sono probabilità. I cinque criteri sono criteri sul train.
Gli XML descrivono posizioni/contorni, non classi di salute o necrosi. Le WSI test vengono elaborate da run_wsi_inference e predict_clustering.
