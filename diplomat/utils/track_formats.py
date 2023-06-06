from typing import List, Union, Tuple
import pandas as pd
from io import BufferedWriter, BufferedReader
from diplomat.processing import Pose


def to_diplomat_table(
    num_outputs: int,
    body_parts: List[str],
    poses: Pose
) -> pd.DataFrame:
    """
    Convert a diplomat Pose object to a diplomat pandas table, which can be saved to a CSV.

    :param num_outputs: The number of outputs in the pose object (number of bodies)
    :param body_parts: The list of body part names describing the parts in the pose object, in order.
    :param poses: The Pose object, storing x, y, and likelihood data for each body part.

    :return: A pd.DataFrame, being a table storing the original pose data.
             Indexing is by body (Body1, Body2, then body part, then value (x, y, or likelihood).
    """
    columns = pd.MultiIndex.from_tuples([
        (f"Body{i + 1}", body_part, sub_val)
        for i in range(num_outputs)
        for body_part in body_parts
        for sub_val in ("x", "y", "likelihood")
    ])

    pose_results = poses.get_all().copy()
    pose_results = pose_results.reshape(
        (pose_results.shape[0], len(body_parts), num_outputs, 3)
    ).transpose((0, 2, 1, 3)).reshape(pose_results.shape)

    return pd.DataFrame(data=pose_results, columns=columns)


def save_diplomat_table(table: pd.DataFrame, path_or_buf: Union[str, BufferedWriter]):
    """
    Save a diplomat-pandas pose table to a CSV file.

    :param table: The table to save.
    :param path_or_buf: The file path, or file handle to store results to.
    """
    table.to_csv(path_or_buf, index=False)


def load_diplomat_table(path_or_buf: Union[str, BufferedReader]) -> pd.DataFrame:
    """
    Load a diplomat-pandas pose table from a CSV file.

    :param path_or_buf: The file path or file handle to load the CSV from.

    :return: A pd.DataFrame, being the diplomat-pandas pose table stored at the given location.
    """
    return pd.read_csv(path_or_buf, index_col=None, header=[0, 1, 2])


def to_diplomat_pose(table: pd.DataFrame) -> Tuple[Pose, List[str], int]:
    """
    Convert a diplomat pandas table to a diplomat Pose object.

    :param table: A diplomat pandas table, typically as loaded from a CSV.

    :return: A tuple containing a diplomat Pose object, a list of string giving the body part names in order,
             and an integer giving the number of bodies, our outputs.
    """
    num_outputs = len(table.columns.unique(0))
    names = table.columns.unique(1)

    poses_enc = table.to_numpy()
    poses_enc = poses_enc.reshape(
        (table.shape[0], num_outputs, len(names), 3)
    ).transpose((0, 2, 1, 3)).reshape(table.shape)

    poses = Pose.empty_pose(table.shape[0], len(names) * num_outputs)
    poses.get_all()[:] = poses_enc

    return (poses, names, num_outputs)
