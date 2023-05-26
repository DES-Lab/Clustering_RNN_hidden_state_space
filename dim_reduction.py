from sklearn.decomposition import PCA, SparsePCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection

dim_reduction_functions = {
    'pca': PCA,
    'pca_sparse': SparsePCA,
    'factor_analysis': FactorAnalysis,
    'random_projection': GaussianRandomProjection,
    'lda': LinearDiscriminantAnalysis,
    'tsne': TSNE,
}


def get_available_dim_reduction_functions():
    return list(dim_reduction_functions.keys())


def reduce_dimensions(data, dim_reduction_function, target_dimensions, **kwargs):
    assert dim_reduction_function in dim_reduction_functions.keys()
    fun = dim_reduction_functions[dim_reduction_function](n_components=target_dimensions, **kwargs)
    return fun.fit_transform(data)

