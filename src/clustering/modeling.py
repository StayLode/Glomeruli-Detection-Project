"""Refit PCA20 and equal-variance feature blocks on explicit train subsets."""
from src.clustering.feature_extractor import FeatureBlockTransformer
from src.clustering.manifold_clustering import GlomeruliClusterer


class FittedFeatureModel:
    def __init__(self, transformer, clusterer):
        self.block_transformer, self.clusterer = transformer, clusterer

    def predict(self, features=None, deep_features=None, morph_features=None):
        return self.clusterer.predict(self.block_transformer.transform(deep_features, morph_features))


def fit_feature_model(n_clusters, algorithm, seed, pca_variance, feature_type, block_weighting,
                      input_scaling, features=None, deep_features=None, morph_features=None):
    if feature_type != "hybrid" or block_weighting != "equal":
        raise ValueError("Only the classic hybrid representation with equal block variance is supported")
    transformer = FeatureBlockTransformer("equal")
    values = transformer.fit_transform(deep_features, morph_features)
    clusterer = GlomeruliClusterer(n_clusters, pca_variance, random_state=seed, input_scaling=input_scaling)
    return FittedFeatureModel(transformer, clusterer.fit(values, algorithm))
