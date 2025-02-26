from diplomat.utils.clustering import nn_chain, get_components, ClusteringMethod
from diplomat.utils.graph_ops import to_valid_graph
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd
import numpy as np


def assert_all_close(arr1, arr2):
    if(np.all(arr1 == arr2)):
        return True
    print(arr1)
    print(arr2)
    print()
    return False


def test_vs_scipy():
    np.random.seed(0)

    for method in ClusteringMethod:
        scipy_name = method.name.lower()

        for i in range(100):
            n = np.random.randint(5, 50)
            random_graph = np.random.random((n, n)) + 0.1
            random_graph = to_valid_graph(random_graph, 0)

            n1, n2, d, s = linkage(ssd.squareform(random_graph), scipy_name).T
            n1 = n1.astype(int)
            n2 = n2.astype(int)
            s = s.astype(int)
            merges, td = nn_chain(random_graph, method)
            tn1, tn2, ts = merges.T

            try:
                assert assert_all_close(n1, tn1)
                assert assert_all_close(n2, tn2)
                assert assert_all_close(s, ts)
                assert np.allclose(d, td)
            except:
                print(n1, "\n", tn1, "\n", sep="")
                print(n2, "\n", tn2, "\n", sep="")
                print(s, "\n", ts, "\n", sep="")
                print(d, "\n", td, "\n", sep="")
                raise


