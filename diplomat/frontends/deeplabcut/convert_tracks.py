import numpy as np
import pandas as pd

def _load_dlc_tracks_h5(path_or_buffer):
    return pd.read_hdf(path_or_buffer)

def _dlc_hdf_to_diplomat_pose(path_or_buffer):
    table = pd.read_hdf(path_or_buffer)

    if(not isinstance(table, pd.DataFrame)):
        raise ValueError("HDF file did not contain table data.")

    np_table_data = table.to_numpy(np.float32)
    full_table_data = np.zeros((np.max(table.index), np_table_data.shape[1]), np.float32)
    full_table_data[table.index] = np_table_data

    table = pd.DataFrame(full_table_data, columns=table.columns)

    if(table.nlevels == 4):
        # This is DLC's latest multi-animal format...

        # Remove the scorer...
        table.columns = table.columns.droplevel(0)

        # Remove single exclusive part predictions...
        for col in table:
            if(col[0] == "single"):
                del table[col]

        bodies = np.unique()
        part_counts = {}
