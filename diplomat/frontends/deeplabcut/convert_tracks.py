from collections import defaultdict
from pathlib import Path
import diplomat.processing.type_casters as tc
import numpy as np
import pandas as pd


@tc.typecaster_function
def _dlc_hdf_to_diplomat_table(path: tc.PathLike) -> pd.DataFrame:
    if isinstance(path, Path):
        path = str(path)
    table = pd.read_hdf(path)

    if(not isinstance(table, pd.DataFrame)):
        raise ValueError("HDF file did not contain table data.")

    np_table_data = table.to_numpy(np.float32)
    full_table_data = np.zeros((np.max(table.index) + 1, np_table_data.shape[1]), np.float32)
    full_table_data[table.index] = np_table_data

    table = pd.DataFrame(full_table_data, columns=table.columns)
    attr_types = ("x", "y", "likelihood")

    if(table.columns.nlevels == 4):
        # This is DLC's latest multi-animal format...

        # Remove the scorer...
        table.columns = table.columns.droplevel(0)

        # Remove single exclusive part predictions...
        for col in table:
            if(col[0] == "single"):
                del table[col]

        # Track how many bodies each part belongs to.
        # we only keep parts that belong to all bodies...
        bodies = {}
        part_belonging = defaultdict(set)

        for col in table:
            body, part, attr = col
            if attr not in attr_types:
                raise ValueError(f"Found unsupported column: {col}")
            bodies[body] = None
            part_belonging[part].add(body)

        final_parts = []

        for part, bodies_for_part in part_belonging.items():
            if(len(bodies_for_part) == len(bodies)):
                final_parts.append(part)

        new_index = pd.MultiIndex.from_product([bodies.keys(), final_parts, attr_types])
        new_table = table[new_index].copy()
    elif(table.columns.nlevels == 3):
        # DLC's single animal, or old multi-animal format...
        # Remove the scorer...
        table.columns = table.columns.droplevel(0)

        # Get all the parts...
        parts = table.columns.unique(0)
        attrs = table.columns.unique(1)
        for attr in attrs:
            for exp_attr in attr_types:
                if attr.startswith(exp_attr):
                    break
            else:
                raise ValueError(f"Unsupported attribute in table: {attr}")

        num_outputs = len(attrs) // 3
        new_index = pd.MultiIndex.from_product([
            [f"Body{i + 1}" for i in range(num_outputs)],
            parts,
            attr_types
        ])
        new_table = pd.DataFrame(columns=new_index)
        for output in range(num_outputs):
            for part in parts:
                for attr in attr_types:
                    ext = output + 1 if output != 0 else ''
                    new_table[f"Body{output + 1}", part, attr] = table[part, f"{attr}{ext}"]
    else:
        raise ValueError(f"Unknown deeplabcut table format, has {table.nlevels} header rows.")

    return new_table

