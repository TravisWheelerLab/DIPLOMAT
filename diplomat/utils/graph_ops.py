import numba
import numpy as np
from typing import Tuple

from .scipy_hungarian import linear_sum_assignment

@numba.experimental.jitclass([
    ["_stack_ptr", numba.int64],
    ["_stack", numba.int64[:]]
])
class Stack:
    def __init__(self, max_size: int):
        self._stack = np.zeros(max_size, dtype=np.int64)
        self._stack_ptr = 0

    def push(self, val: int):
        self._stack[self._stack_ptr] = val
        self._stack_ptr += 1

    def pop(self) -> int:
        self._stack_ptr -= 1
        return self._stack[self._stack_ptr]

    def size(self):
        return self._stack_ptr


@numba.njit
def _connected_components(graph: np.ndarray) -> np.ndarray:
    node_count = graph.shape[-1]
    stack = Stack(node_count * 2)
    component = np.full(node_count, -1, dtype=np.int64)

    for i in range(node_count):
        stack.push(i)

        while(stack.size() > 0):
            current_node = stack.pop()
            if(component[current_node] < 0):
                component[current_node] = i

            for j in range(node_count):
                if((graph[current_node, j] != 0) and (component[j] < 0)):
                    stack.push(j)

    return component


def connected_components(graph: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Compute the connected components of a graph.

    :param graph: An adjacency matrix representing a graph, stored as a 2D numpy array.

    :returns: A tuple of 2 values, the first value being the number of components (integer), and the second value
              being an array of identifiers, specifying which component each node belongs to.
    """
    graph = ~np.isinf(graph)
    np.fill_diagonal(graph, 0)
    components = _connected_components(graph)
    vals, indexes = np.unique(components, return_inverse=True)
    return (len(vals), indexes)


def min_spanning_tree(graph: np.ndarray) -> np.ndarray:
    """
    Finds the minimum spanning tree of a graph encoded as an adjacency matrix. Algorithm is node-centric Prim's, which
    guarantees a runtime of O(n^2) for all graphs, and is ideal for dense graphs.

    :param graph: An adjacency matrix, assumes disconnected nodes have distance value of infinity.

    :returns: A new adjacency matrix, representing the minimum spanning tree covering the provided graph.
    """
    return _min_spanning_tree(graph.astype(np.float32))


@numba.njit
def _min_spanning_tree(graph: np.ndarray) -> np.ndarray:
    tree = np.full(graph.shape, np.inf, dtype=np.float32)
    num_nodes = graph.shape[-1]
    np.fill_diagonal(graph, np.inf)

    explore_node = 0
    min_source = np.zeros(num_nodes, np.int64)
    min_links = np.full(num_nodes, np.inf, np.float32)
    in_tree = np.zeros(num_nodes, np.uint8)
    in_tree[0] = True

    for __ in range(num_nodes - 1):
        for j in range(num_nodes):
            if(not in_tree[j] and graph[explore_node, j] < min_links[j]):
                min_source[j] = explore_node
                min_links[j] = graph[explore_node, j]

        best_idx = 0
        best_val = np.inf
        for j in range(num_nodes):

            if(not in_tree[j] and min_links[j] < best_val):
                best_idx = j
                best_val = min_links[j]

        best_source = min_source[best_idx]

        tree[best_source, best_idx] = graph[best_source, best_idx]
        tree[best_idx, best_source] = graph[best_idx, best_source]

        in_tree[best_idx] = True
        explore_node = best_idx

    return tree


@numba.njit
def _min_row_subtract(g: np.ndarray) -> np.ndarray:
    """
    Subtract the minimum value for each row from a graph. Used by the min_cost_assignment algorithm.
    """
    for i in range(g.shape[0]):
        min_row_val = np.inf

        for j in range(g.shape[1]):
            min_row_val = g[i, j] if(g[i, j] < min_row_val) else min_row_val

        for j in range(g.shape[1]):
            g[i, j] -= min_row_val

    return g

def min_cost_matching(cost_matrix: np.ndarray, mode="scipy") -> Tuple[np.ndarray, np.ndarray]:
    if mode == "scipy":
        return linear_sum_assignment(cost_matrix)
    else:
        return _min_cost_matching(cost_matrix)

@numba.njit
def _min_cost_matching(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the hungarian algorithm for solving the minimum assignment problem. Given a cost matrix,
    find the optimal assignment of workers (rows) to jobs (columns). Algorithm is O(N^3).

    :param cost_matrix: The cost matrix to find an optimal matching for.

    :returns: A tuple of 2 numpy arrays, first representing row assignments (row -> column) and second column
              assignments (column -> row). Note, these are inversions of each other.
    """

    graph_copy = cost_matrix.copy()

    while(True):
        # Subtract current minimum row/column values to get current greedy solution...
        graph_copy = _min_row_subtract(graph_copy)
        graph_copy = _min_row_subtract(graph_copy.T).T

        # Get number of row and column zeros for each row and column
        zeros = (graph_copy == 0).astype(np.int64)
        row_zeros = np.sum(zeros, axis=1)
        col_zeros = np.sum(zeros, axis=0)

        # Cover the graph with the minimum number of rows and columns.
        marked_rows = np.zeros(len(row_zeros), np.uint8)
        marked_cols = np.zeros(len(col_zeros), np.uint8)

        for i in range(len(row_zeros)):
            for j in range(len(col_zeros)):
                # If we find a zero, either it's row or column MUST be marked to cover it.
                # We select the one which covers more zeros, this guarantees an optimal assignment...
                if(graph_copy[i, j] <= 0 and not marked_rows[i] and not marked_cols[j]):
                    if(row_zeros[i] > col_zeros[j]):
                        marked_rows[i] = 1
                    else:
                        marked_cols[j] = 1

        # Compute the assignment coverage...
        coverage = np.sum(marked_rows) + np.sum(marked_cols)

        if(coverage >= graph_copy.shape[0]):
            break

        min_uncovered = np.inf

        for i in range(graph_copy.shape[0]):
            for j in range(graph_copy.shape[1]):
                val = graph_copy[i, j]
                if((not marked_rows[i]) and (not marked_cols[j]) and (val < min_uncovered)):
                    min_uncovered = val

        for i in range(graph_copy.shape[0]):
            for j in range(graph_copy.shape[1]):
                if(not marked_rows[i] and not marked_cols[j]):
                    graph_copy[i, j] -= min_uncovered
                elif(marked_rows[i] and marked_cols[j]):
                    graph_copy[i, j] += min_uncovered

    row_solution = np.full(graph_copy.shape[0], -1, np.int64)
    col_solution = np.full(graph_copy.shape[1], -1, np.int64)

    for _ in range(len(row_solution)):
        min_row = 0

        for i2 in range(len(row_solution)):
            if(row_zeros[i2] < row_zeros[min_row]):
                min_row = i2

        for j in range(len(col_solution)):
            if(graph_copy[min_row, j] == 0 and row_solution[min_row] < 0 and col_solution[j] < 0):
                row_solution[min_row] = j
                col_solution[j] = min_row
                row_zeros[min_row] = np.iinfo(row_zeros.dtype).max
                for k in range(len(row_zeros)):
                    if(graph_copy[k, j] == 0 and row_zeros[k] != np.iinfo(row_zeros.dtype).max):
                        row_zeros[k] -= 1
                break

    return (np.arange(cost_matrix.shape[0], dtype=np.int64), col_solution)


def to_valid_graph(g: np.ndarray) -> np.ndarray:
    """
    Convert an arbitrary adjacency matrix to a valid undirected graph. Diagonal is set to positive infinity
    and graph is made symmetric by copying the lower triangle of the graph to the upper triangle.
    """
    g = g.copy()
    np.fill_diagonal(g, np.inf)
    i_upper = np.triu_indices(g.shape[0], 1)
    g[i_upper] = g.T[i_upper]
    return g


if(__name__ == "__main__"):
    inf = np.inf

    # print(_min_spanning_tree(np.array([
    #     [inf, 1, 2, 6],
    #     [1, inf, 5, 3],
    #     [2, 5, inf, 7],
    #     [6, 3, 7, inf]
    # ])))

    # print(connected_components(np.array([
    #     [inf, inf, inf, inf],
    #     [inf, inf, inf, 6],
    #     [inf, inf, inf, 7],
    #     [inf, 6, 7, inf]
    # ])))

    # print(min_cost_matching(np.array([
    #     [1, 3, 5, 7],
    #     [3, 5, 7, 1],
    #     [21, 2, 6, 4],
    #     [99, 91, 8, 1]
    # ])))

    print(min_cost_matching(np.array([[9.92690898e+04, 1.07946068e+01, 2.50553569e-01],
        [3.14225839e+01, 1.54683226e+00, 1.06039718e+05],
        [3.12269539e+01, 9.64991177e+04, 1.47355721e-03]])))