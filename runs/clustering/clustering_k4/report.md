 **clustering_k4: train e validation** 

Experimental ordering only: A has the highest train-cluster median peripheral clear fraction and the final letter has the lowest. The letters are shown as a low-to-high necrosis-compatibility proxy for visualization; this is not a measurement of Bowman's space or a validated clinical grade.

Fit solo sul train; mapping delle lettere congelato prima delle assegnazioni held-out. La rim fraction è un proxy esplorativo, non una diagnosi.

| Modello | Split/slide | N | Conteggi lettere | Silhouette comune | Mediana distanza | Criteri train |
| --- | --- | --- | --- | --- | --- | --- |
| clustering_k4 | train | 411 | {'A': 97, 'B': 101, 'D': 33, 'C': 180} | 0.1736 | 1.3948 | 3/5 |
| clustering_k4 | val | 57 | {'A': 3, 'B': 20, 'D': 0, 'C': 34} | -0.0076 | 1.3618 | None |

Le lettere vengono ordinate sul train e mantenute fisse negli altri split. Non sono una classificazione clinica validata.
I modelli ai diversi K sono indipendenti: lettere uguali non garantiscono identici sottogruppi fra K.
Un solo cluster occupato rende silhouette/DB/CH non valutabili; una sola slide rende Cramér's V non stimabile.
Le distanze appartengono allo spazio del modello e non sono probabilità. I cinque criteri sono criteri sul train.
Gli XML descrivono posizioni/contorni, non classi di salute o necrosi. Le WSI test vengono elaborate da run_wsi_inference e predict_clustering.
