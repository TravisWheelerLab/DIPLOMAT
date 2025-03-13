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


# Converts scipy format to our format for merges...
def remove_cluster_indexes(n1, n2):
    n1res = n1.copy()
    n2res = n2.copy()
    lowest_node = np.arange((len(n1) + 1) + len(n1))

    for i in range(len(n1)):
        a = n1[i]
        b = n2[i]
        n1res[i], n2res[i] = sorted([lowest_node[a], lowest_node[b]])
        low_node = min(lowest_node[a], lowest_node[b])
        lowest_node[a] = low_node
        lowest_node[b] = low_node
        lowest_node[len(n1) + 1 + i] = low_node

    return n1res, n2res


# Converts our format to scipy...
def add_cluster_indexes(n1, n2):
    n1res = n1.copy()
    n2res = n2.copy()
    current_cluster = np.arange(len(n1) + 1)
    next_cluster = len(n1) + 1

    for i in range(len(n1)):
        a = n1[i]
        b = n2[i]
        n1res[i], n2res[i] = sorted([current_cluster[a], current_cluster[b]])
        current_cluster[a] = next_cluster
        current_cluster[b] = next_cluster
        next_cluster += 1

    return n1res, n2res


def order_components(x):
    vals, idx, inv = np.unique(x, return_index=True, return_inverse=True)
    idx_order = np.argsort(idx)
    idx[idx_order] = np.arange(len(idx))
    return idx[inv], len(vals)


def test_vs_scipy():
    """
    Tests internal nn_chain algorithm against scipy's implementation...
    """
    np.random.seed(0)

    for method in ClusteringMethod:
        scipy_name = method.name.lower()

        for i in range(200):
            n = np.random.randint(2, 50)
            random_graph = np.random.random((n, n))
            random_graph = to_valid_graph(random_graph, 0)

            z = linkage(ssd.squareform(random_graph), scipy_name)
            n1, n2, d, s = z.T
            n1 = n1.astype(int)
            n2 = n2.astype(int)
            s = s.astype(int)
            merges, td = nn_chain(random_graph, method)
            tn1, tn2, ts = merges.T
            tn1_conv, tn2_conv = add_cluster_indexes(tn1, tn2)

            try:
                assert assert_all_close(n1, tn1_conv)
                assert assert_all_close(n2, tn2_conv)
                assert assert_all_close(s, ts)
                assert np.allclose(d, td)
            except:
                print(n1, "\n", tn1_conv, "\n", sep="")
                print(n2, "\n", tn2_conv, "\n", sep="")
                print(s, "\n", ts, "\n", sep="")
                print(d, "\n", td, "\n", sep="")
                raise

            for num_comps in range(1, n + 1):
                # Scipy components start at 1...
                components, ac_comps = order_components(fcluster(z, num_comps, "maxclust"))
                test_components, returned_comps = get_components(merges, td, num_comps)
                actual_comps = len(np.unique(test_components))
                assert actual_comps == returned_comps
                assert len(np.unique(test_components)) == num_comps
                if(ac_comps != num_comps):
                    # Scipy failed to actually give the number of clusters we asked for, continue...
                    continue
                try:
                    assert assert_all_close(components, test_components)
                except:
                    print(f"Component count: {num_comps}")
                    raise





