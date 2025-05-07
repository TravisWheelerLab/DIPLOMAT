from typing import Union, Tuple
import numba
import numpy as np
from enum import IntEnum


_numeric = Union[np.ndarray, int]


class ClusteringMethod(IntEnum):
    """
    The set of agglomerate clustering methods supported by this module.
    Determines how weights are adjusted after a merge.
    """
    SINGLE = 0
    """ Single linkage clustering, the distance between two clusters is the distance between the two closest nodes to each. """
    COMPLETE = 1
    """ Complete linkage clustering, the distance between two clusters is the distance between the two farthest points in each. """
    AVERAGE = 2
    """ Average clustering. The distance between two clusters is the distance between their points averaged. """
    WEIGHTED = 3
    """ Weighted average clustering. Like average, but ignores cluster size when determining the average point for a new cluster. """
    WARD = 4
    """ Ward clustering. Minimizes the total within-cluster variance. """


_CLUSTERING_METHOD_MAX = max(ClusteringMethod)
_CLUSTERING_METHOD_MIN = min(ClusteringMethod)


dist_update_func_sig = numba.float64(
    numba.int64,
    numba.int64,
    numba.int64,
    numba.int64,
    numba.float64,
    numba.float64,
    numba.float64
)


@numba.njit(dist_update_func_sig)
def distance_update(mode: int, sx: int, sy: int, si: int, dxy: float, dxi: float, dyi: float) -> float:
    if(mode == ClusteringMethod.SINGLE):
        return min(dxi, dyi)
    elif(mode == ClusteringMethod.COMPLETE):
        return max(dxi, dyi)
    elif(mode == ClusteringMethod.AVERAGE):
        return (sx * dxi + sy * dyi) / (sx + sy)
    elif(mode == ClusteringMethod.WEIGHTED):
        return 0.5 * (dxi + dyi)
    elif(mode == ClusteringMethod.WARD):
        denom = 1.0 / (sx + sy + si)
        return np.sqrt(
            ((sx + si) * denom) * np.square(dxi)
            + ((sy + si) * denom) * np.square(dyi)
            - (si * denom) * np.square(dxy)
        )
    else:
        return min(dxi, dyi)



@numba.njit("types.UniTuple(i8, 2)(i8, i8)")
def dist_index(node1: _numeric, node2: _numeric):
    if(node1 > node2):
        node1, node2 = node2, node1
    return node1, node2


@numba.njit("types.Tuple((i8[:, :], f8[:]))(f8[:, :], i8)")
def nn_chain(dists: np.ndarray, linkage_mode: int):
    """
    Use the nearest neighbor chain algorithm to perform hierarchical clustering.
    """
    assert dists.ndim == 2
    assert dists.shape[0] == dists.shape[1]
    assert _CLUSTERING_METHOD_MIN <= linkage_mode <= _CLUSTERING_METHOD_MAX

    dists = dists.copy()
    node_count = len(dists)
    sizes = np.ones(node_count)

    stack = np.zeros(node_count, dtype=np.int64)
    chain_length: int = 0

    nodes_and_merge = np.zeros((node_count - 1, 3), dtype=np.int64)
    merge_distances = np.zeros(node_count - 1, dtype=np.float64)

    for i in range(node_count - 1):
        # Chain is empty, add the first next valid cluster...
        if(chain_length == 0):
            for j in range(node_count):
                if sizes[j] != 0:
                    stack[0] = j
                    chain_length = 1
                    break

        nn_current: int = 0
        nn_next: int = 0

        # Grow the nearest neighbor chain...
        while(True):
            nn_current = stack[chain_length - 1]
            # Set to the past link if chain is long enough, this prevents cycles...
            nn_next = stack[chain_length - 2] if(chain_length >= 2) else nn_current
            current_min = dists[dist_index(nn_current, nn_next)] if(chain_length >= 2) else np.inf

            # Find the nearest neighbor for this cluster...
            for k in range(node_count):
                # Check if a valid cluster index...
                if(sizes[k] == 0 or k == nn_current):
                    continue

                # If distance is closer, update next nearest neighbor...
                dist = dists[dist_index(nn_current, k)]
                if(dist < current_min):
                    current_min = dist
                    nn_next = k

            # If the next nearest neighbor is second one back on the stack, we've found the next set of
            # clusters to merge...
            if(chain_length >= 2 and nn_next == stack[chain_length - 2]):
                break

            stack[chain_length] = nn_next
            chain_length += 1

        # Pop next 2 nodes off the stack...
        chain_length -= 2
        # Node 1 should be the smaller of the two, this is to make merging encoding consistent...
        if(nn_current > nn_next):
            nn_current, nn_next = nn_next, nn_current

        n1_size = sizes[nn_current]
        n2_size = sizes[nn_next]
        # Write next cluster merge (solution) for this step...
        nodes_and_merge[i, 0] = nn_current
        nodes_and_merge[i, 1] = nn_next
        merge_distances[i] = current_min
        nodes_and_merge[i, 2] = n1_size + n2_size
        # Merge the two clusters by updating sizes...
        sizes[nn_current] = n1_size + n2_size
        sizes[nn_next] = 0

        # Update distances for the new merged cluster...
        for l in range(node_count):
            l_size = sizes[l]
            if(l_size == 0 or l == nn_current):
                continue

            dists[dist_index(nn_current, l)] = distance_update(
                linkage_mode,
                n1_size,
                n2_size,
                l_size,
                current_min,
                dists[dist_index(nn_current, l)],
                dists[dist_index(nn_next, l)]
            )

    # Reorder by distance, stably...
    idx_order = np.argsort(merge_distances, kind="mergesort")
    return nodes_and_merge[idx_order], merge_distances[idx_order]


UnionFindType = np.ndarray


@numba.njit("i8[:, :](i8)")
def _new_union_find(size: int) -> UnionFindType:
    res = np.ones((2, size), dtype=np.int64)
    res[0, :] = np.arange(size, dtype=np.int64)
    return res


@numba.njit("i8(i8[:, :], i8)")
def _uf_compress(uf: UnionFindType, n: int) -> int:
    parents = uf[0]

    if(n == parents[n]):
        return n
    else:
        root = _uf_compress(uf, parents[n])
        parents[n] = root
        return root


@numba.njit("i8(i8[:, :], i8)")
def _uf_find(uf: UnionFindType, n: int):
    return _uf_compress(uf, n)


@numba.njit("i8(i8[:, :], i8, i8)")
def _uf_union(uf: UnionFindType, n1: int, n2: int) -> int:
    parents, sizes = uf

    n1 = _uf_find(uf, n1)
    n2 = _uf_find(uf, n2)

    if(n1 == n2):
        return sizes[n1]

    n1_size = sizes[n1]
    n2_size = sizes[n2]
    if(n1_size < n2_size):
        n1, n2 = n2, n1

    merged_size = n1_size + n2_size
    sizes[n1] = merged_size
    sizes[n2] = merged_size
    parents[n2] = n1

    return merged_size


@numba.njit("types.Tuple((i8[:], i8))(i8[:, :], f8[:], i8)")
def get_components(merge_list: np.ndarray, distances: np.ndarray, num_components: int):
    """
    Get the components or clusters of set of nodes after performing hierarchical clustering, given a specific number
    of desired components to be returned. Returns clustering solution at that level.
    """
    assert merge_list.ndim == 2
    assert distances.ndim == 1
    assert merge_list.shape[1] == 3
    assert merge_list.shape[0] >= 1 and merge_list.shape[0] == distances.shape[0]

    size = merge_list.shape[0] + 1
    assert 0 < num_components
    num_components = min(num_components, size)

    components = np.full(size, fill_value=-1, dtype=np.int64)
    iters = size - num_components
    uf = _new_union_find(size)

    # Merge nodes until number of components matches desired amount...
    for i in range(iters):
        _uf_union(uf, merge_list[i, 0], merge_list[i, 1])

    # For every node, find it's root. If it's root
    # has not been assigned a component index, assign it one
    # and set the component to the same as the root.
    component_index = 0
    for j in range(size):
        root = _uf_find(uf, j)
        if(components[root] < 0):
            components[root] = component_index
            component_index += 1
        components[j] = components[root]

    return components, num_components

