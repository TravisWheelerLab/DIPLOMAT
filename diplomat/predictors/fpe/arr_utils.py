from typing import Sequence, List, Tuple
import numpy as np

def intersect_coords_indexes(coord_list: Sequence[np.ndarray]) -> List[np.ndarray]:
    comb_coords = np.concatenate(coord_list)
    indexes = np.concatenate([np.arange(len(c)) for c in coord_list])
    masks = np.repeat(
        np.repeat(np.arange(coord_list), [len(c) for c in coord_list])
    )

    sorted_locs = np.lexsort((comb_coords[:, 0], comb_coords[:, 1]))
    sorted_comb_coords = comb_coords[sorted_locs]

    intersection_locs = np.where(
        np.all(sorted_comb_coords[:-1] == sorted_comb_coords[1:], axis=1)
    )[0]
    idxs_intersect = indexes[sorted_locs][intersection_locs]
    masks_intersect = masks[sorted_locs][intersection_locs]

    return [
        idxs_intersect[masks_intersect == i] for i in range(len(coord_list))
    ]

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
    probs: Sequence[np.ndarray],
    coord_list: Sequence[np.ndarray],
    fill_value: float = 0
) -> Tuple[ndlist, ndlist, ndlist]:
    def flattify(arr, max_row_val):
        return arr[:, 0] * (max_row_val + 1) + arr[:, 1]

    def intersect_idx(arr1, arr2):
        frm_arr = np.repeat([0, 1], [len(arr1), len(arr2)])
        arr_idxs = np.concatenate(
            [np.arange(len(arr1)), np.arange(len(arr2))], None
        )
        comb_coords = np.concatenate([arr1, arr2], None)

        sort_idxs = np.argsort(comb_coords)
        comb_coords = comb_coords[sort_idxs]
        arr_idxs = arr_idxs[sort_idxs]
        frm_arr = frm_arr[sort_idxs]

        intersect_locs = np.append(comb_coords[1:] == comb_coords[:-1], [False])
        intersect_locs |= np.roll(intersect_locs, 1)

        return (
            arr_idxs[(~frm_arr & intersect_locs).astype(bool)],
            arr_idxs[(frm_arr & intersect_locs).astype(bool)]
        )

    all_coords = np.unique(np.concatenate(coord_list), axis=0)
    max_row2_val = np.max(all_coords[:, 1])
    all_coords_flat = flattify(all_coords, max_row2_val)

    new_probs = []
    resolve_idxs = []

    for c, prob in zip(coord_list, probs):
        c = flattify(c, max_row2_val)
        idx_c, idx_all = intersect_idx(c, all_coords_flat)
        res = np.full(len(all_coords_flat), fill_value)
        res[idx_all] = prob[idx_c]
        resolve_idxs.append(idx_all[np.argsort(idx_c)])
        new_probs.append(res)

    return (
        new_probs,
        [all_coords for __ in range(len(coord_list))],
        resolve_idxs
    )
