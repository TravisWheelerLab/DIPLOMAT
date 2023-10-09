from typing import Tuple
import numpy as np
from diplomat.utils.graph_ops import min_cost_matching


def greedy(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix).copy()
    rows, cols = np.zeros(matrix.shape[0], dtype=np.int64), np.zeros(matrix.shape[1], dtype=np.int64)

    for i in range(matrix.shape[0]):
        row_max, col_max = np.unravel_index(np.argmin(matrix), matrix.shape)
        rows[row_max] = row_max
        cols[row_max] = col_max
        matrix[row_max, :] = np.inf
        matrix[:, col_max] = np.inf

    return rows, cols


def hungarian(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return min_cost_matching(matrix)


ASSIGNMENT_ALGORITHMS = {
    "greedy": greedy,
    "hungarian": hungarian
}
