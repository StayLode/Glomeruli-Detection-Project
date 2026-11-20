 **clustering_k6: train e validation** 

Experimental ordering only: A has the highest train-cluster median peripheral clear fraction and the final letter has the lowest. The letters are shown as a low-to-high necrosis-compatibility proxy for visualization; this is not a measurement of Bowman's space or a validated clinical grade.

Fit solo sul train; mapping delle lettere congelato prima delle assegnazioni held-out. La rim fraction è un proxy esplorativo, non una diagnosi.

| Modello | Split/slide | N | Conteggi lettere | Silhouette comune | Mediana distanza | Criteri train |
| --- | --- | --- | --- | --- | --- | --- |
| clustering_k6 | train | 411 | {'C': 121, 'B': 71, 'A': 69, 'D': 17, 'E': 117, 'F': 16} | 0.1351 | 1.2741 | 2/5 |
| clustering_k6 | val | 57 | {'C': 31, 'B': 11, 'A': 0, 'D': 0, 'E': 15, 'F': 0} | 0.1163 | 1.2924 | None |

Le lettere vengono ordinate sul train e mantenute fisse negli altri split. Non sono una classificazione clinica validata.
I modelli ai diversi K sono indipendenti: lettere uguali non garantiscono identici sottogruppi fra K.
Un solo cluster occupato rende silhouette/DB/CH non valutabili; una sola slide rende Cramér's V non stimabile.
Le distanze appartengono allo spazio del modello e non sono probabilità. I cinque criteri sono criteri sul train.
Gli XML descrivono posizioni/contorni, non classi di salute o necrosi. Le WSI test vengono elaborate da run_wsi_inference e predict_clustering.
