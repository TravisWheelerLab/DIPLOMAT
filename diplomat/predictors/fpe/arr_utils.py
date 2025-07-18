from typing import Sequence, List, Tuple
import numpy as np


def intersect_coords_indexes(coord_list: Sequence[np.ndarray]) -> List[np.ndarray]:
    comb_coords = np.concatenate(coord_list)
    indexes = np.concatenate([np.arange(len(c)) for c in coord_list])
    masks = np.repeat(np.repeat(np.arange(coord_list), [len(c) for c in coord_list]))

    sorted_locs = np.lexsort((comb_coords[:, 0], comb_coords[:, 1]))
    sorted_comb_coords = comb_coords[sorted_locs]

    intersection_locs = np.where(
        np.all(sorted_comb_coords[:-1] == sorted_comb_coords[1:], axis=1)
    )[0]
    idxs_intersect = indexes[sorted_locs][intersection_locs]
    masks_intersect = masks[sorted_locs][intersection_locs]

    return [idxs_intersect[masks_intersect == i] for i in range(len(coord_list))]


def intersect_coords(coord_list: Sequence[np.ndarray]) -> np.ndarray:
    comb_coords = np.concatenate(coord_list)

    sorted_locs = np.lexsort((comb_coords[:, 0], comb_coords[:, 1]))
    sorted_comb_coords = comb_coords[sorted_locs]

    intersection_locs = np.where(
        np.all(sorted_comb_coords[:-1] == sorted_comb_coords[1:], axis=1)
    )[0]

    return sorted_comb_coords[intersection_locs]


def union_coords(coord_list: Sequence[np.ndarray]) -> np.ndarray:
    return np.unique(np.concatenate(coord_list), axis=0)


ndlist = List[np.ndarray]


def pad_coordinates_and_probs(
    probs: Sequence[np.ndarray], coord_list: Sequence[np.ndarray], fill_value: float = 0
) -> Tuple[ndlist, ndlist, ndlist]:
    def flattify(arr, max_row_val):
        return arr[:, 0] * (max_row_val + 1) + arr[:, 1]

    def intersect_idx(arr1, arr2):
        frm_arr = np.repeat([0, 1], [len(arr1), len(arr2)])
        arr_idxs = np.concatenate([np.arange(len(arr1)), np.arange(len(arr2))], None)
        comb_coords = np.concatenate([arr1, arr2], None)

        sort_idxs = np.argsort(comb_coords)
        comb_coords = comb_coords[sort_idxs]
        arr_idxs = arr_idxs[sort_idxs]
        frm_arr = frm_arr[sort_idxs]

        intersect_locs = np.append(comb_coords[1:] == comb_coords[:-1], [False])
        intersect_locs |= np.roll(intersect_locs, 1)

        return (
            arr_idxs[(~frm_arr & intersect_locs).astype(bool)],
            arr_idxs[(frm_arr & intersect_locs).astype(bool)],
        )

    all_coords_float = np.concatenate(coord_list)
    all_coords, coord_index = np.unique(
        all_coords_float.astype(np.int64), return_index=True, axis=0
    )
    all_coords_float = all_coords_float[coord_index]

    max_row2_val = np.max(all_coords[:, 1])
    all_coords_flat = flattify(all_coords, max_row2_val)

    new_probs = []
    new_coords = []
    resolve_idxs = []

    for c, prob in zip(coord_list, probs):
        c_flat = flattify(c.astype(np.int64), max_row2_val)
        idx_c, idx_all = intersect_idx(c_flat, all_coords_flat)

        res = np.full(len(all_coords_flat), fill_value)
        res_c = all_coords_float.copy()

        res[idx_all] = prob[idx_c]
        res_c[idx_all] = c[idx_c]

        resolve_idxs.append(idx_all[np.argsort(idx_c)])
        new_probs.append(res)
        new_coords.append(res_c)

    return (new_probs, new_coords, resolve_idxs)


class _NumpyDict:
    def __init__(self, keys: np.ndarray, values: np.ndarray, default_val: float = 0):
        self._keys = keys
        self._values = values
        self._sorted_key_indexes = np.argsort(keys)
        self._default_value = default_val

    def __getitem__(self, query):
        into_sorted_indexes = np.searchsorted(
            self._keys, query, sorter=self._sorted_key_indexes
        )
        out_of_bounds = into_sorted_indexes >= len(self._values)
        into_sorted_indexes[out_of_bounds] = 0
        indexes = self._sorted_key_indexes[into_sorted_indexes]
        vals = self._values[indexes]
        vals[(self._keys[indexes] != query) | out_of_bounds] = self._default_value
        return vals


def find_peaks(x: np.ndarray, y: np.ndarray, prob: np.ndarray, fill_value: float = 0):
    """
    Finds the peaks of a sparse frame, or locations where all neighboring values are less than the value at this
    location (including diagonals).

    :param x: The x indexes of each cell.
    :param y: The y indexes of each cell.
    :param prob: The probability, or score of each cell.
    :param fill_value: The fill value to fill non-existent cells with. Defaults to 0.

    :returns: An array of indexes, being the locations of peaks.
    """
    if not np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.int64)
    if not np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.int64)

    keep_arr = np.ones(prob.shape, dtype=np.uint8)
    x_max = np.max(x)

    def to_keys(_x, _y):
        return _y * (x_max + 1) + _x

    lookup_table = _NumpyDict(to_keys(x, y), prob, fill_value)

    # We perform a 3x3 max-convolution to find peaks.
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            neighbor = lookup_table[to_keys(x + j, y + i)]
            below_to_right = (i >= 0) & (j >= 0)
            keep_arr = keep_arr & (
                neighbor <= prob if (below_to_right) else neighbor < prob
            )

    return np.flatnonzero(keep_arr)
