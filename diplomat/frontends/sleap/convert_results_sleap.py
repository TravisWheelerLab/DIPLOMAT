from pathlib import Path

from .sleap_importer import sleap

import diplomat.processing.type_casters as tc
from diplomat.utils.track_formats import to_diplomat_table, save_diplomat_table

from .run_utils import (
    _paths_to_str,
    _to_diplomat_poses,
    _load_config
)


@tc.typecaster_function
def convert_results(
    config: tc.PathLike,
    videos: tc.Union[tc.List[tc.PathLike], tc.PathLike]
):
    """
    Convert diplomat generated results for SLEAP (in .slp format) to a csv based table format for easier analysis
    of results. The csv will have the same name and file path as the original .slp file, just a different extension.

    :param config: The path (or list of paths) to the sleap model(s) used for inference, each as either as a folder or
                   zip file.
    :param videos: Paths to the sleap label files, or .slp files, to convert to csv files, NOT the video files.
    """
    # Load config just to verify it's valid...
    _load_config(_paths_to_str(config))

    label_paths = _paths_to_str(videos)
    label_paths = [label_paths] if(isinstance(label_paths, str)) else label_paths

    for label_path in label_paths:
        _convert_results_single(label_path)


def _convert_results_single(label_path: str):
    print(f"Converting '{label_path}' to a CSV file...")

    sleap_labels = sleap.load_file(label_path)
    num_outputs, pose_obj, video, skeleton = _to_diplomat_poses(sleap_labels)

    label_path = Path(label_path).resolve()
    save_path = label_path.parent / (label_path.stem + ".csv")
    print(f"Saving to: '{save_path}'")

    table = to_diplomat_table(num_outputs, skeleton.node_names, pose_obj)
    save_diplomat_table(table, str(save_path))
