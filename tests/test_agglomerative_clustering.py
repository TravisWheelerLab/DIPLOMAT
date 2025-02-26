from diplomat.utils.clustering import hierarchical_clustering, get_components, ClusteringMethod
from scipy.cluster.hierarchy import linkage, fcluster

def test_vs_scipy():
    for method in ClusteringMethod:
        print(str(method.name))

