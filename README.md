# Glomeruli-Detection-Project
## Description
### Outline
Accurate identification and classification of glomeruli are essential for diagnosing
diabetic kidney disease and guiding clinical decision-making.
### Aim
This project aims to develop a deep learning pipeline capable of:
1. Identifying glomeruli.
2. Segmenting the identified glomeruli.
3. Classifying the glomeruli into distinct categories using an unsupervised
approach.

Given the lack of ground truth data for glomeruli classification, the project will focus
on exploring the separability of glomeruli classes based on levels of necrotization
(see below figure). The unsupervised method will help reveal potential patterns or
clusters within the data.

### Dataset
Dataset includes:
1. WSIs of biopsies from kidney tissue (.svs)
2. The corresponding annotation showing the glomeruli (.xml)
