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


def remove_cluster_indexes(n1, n2):
    n1res = n1.copy()
    n2res = n2.copy()
    lowest_node = np.arange((len(n1) + 1) + len(n1))
    next_clust = len(n1)

    for i in range(len(n1)):
        a = n1[i]
        b = n2[i]
        n1res[i], n2res[i] = sorted([lowest_node[a], lowest_node[b]])
        low_node = min(a, b)
        lowest_node[a] = low_node
        lowest_node[b] = low_node
        lowest_node[next_clust] = low_node
        next_clust += 1

    return n1res, n2res



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


