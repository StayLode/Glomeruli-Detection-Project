"""Serializable classic PCA20 pipeline and its frozen experimental label mapping."""

import numpy as np
from src.clustering.feature_extractor import extract_morphological_dataset
from src.clustering.labels import peripheral_clear_fraction, LABEL_LIMITATION


class ClusterAssignmentPipeline:
    def __init__(self, extractor, clusterer, feature_type="hybrid", morph_units="native", block_transformer=None):
        self.extractor, self.clusterer = extractor, clusterer
        self.feature_type, self.morph_units, self.block_transformer = feature_type, morph_units, block_transformer
        self.label_mapping = {i:chr(65+i) for i in range(clusterer.n_clusters)}
        self.label_limitation = LABEL_LIMITATION

    def _features_for_new(self, crops, masks, pixel_sizes_um):
        if pixel_sizes_um is None:
            raise ValueError("Physical pixel sizes are required")
        morph, _ = extract_morphological_dataset(crops, masks, "native", pixel_sizes_um)
        deep = self.extractor.extract_deep_embeddings(crops)
        return self.block_transformer.transform(deep, morph)

    def assign_new(self, crops, masks, pixel_sizes_um=None):
        features = self._features_for_new(crops, masks, pixel_sizes_um)
        return self.clusterer.predict(features), self.clusterer.distances_to_centers(features)

    def class_labels(self, labels):
        return np.asarray([self.label_mapping[int(label)] for label in labels])

    def assign_new_details(self, crops, masks, pixel_sizes_um=None):
        features = self._features_for_new(crops, masks, pixel_sizes_um)
        ids = self.clusterer.predict(features)
        return {"cluster":ids, "class_label":self.class_labels(ids),
                "distance_to_center":self.clusterer.distances_to_centers(features),
                "peripheral_clear_fraction":np.asarray([peripheral_clear_fraction(c,m) for c,m in zip(crops,masks)])}
