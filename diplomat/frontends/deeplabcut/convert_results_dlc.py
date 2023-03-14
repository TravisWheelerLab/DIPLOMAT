from typing import MutableMapping, Any
import pandas as pd
from .dlc_importer import auxiliaryfunctions
from .label_videos_dlc import _to_str_list, _get_video_info
import diplomat.processing.type_casters as tc
from pathlib import Path
from diplomat.processing import Pose
from diplomat.utils.track_formats import to_diplomat_table, save_diplomat_table


@tc.typecaster_function
def convert_results(
    config: tc.PathLike,
    videos: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    shuffle: int = 1,
    training_set_index: int = 0,
    tracker: str = "",
    video_type: str = "",
):
    """
    Convert DeepLabCut tracking results (.h5 file) to a csv file for easier introspection and analysis.

    :param config: The path to the config.yaml file for the project.
    :param videos: A single video or list of videos to convert the tracking results of, passed in order to modify.
    :param shuffle: The shuffle index of the model used to track the video. Defaults to 1.
    :param training_set_index: The training index of the model used. Defaults to 0.
    :param tracker: String, the extension of the deeplabcut tracker used, used to find the h5 file. Doesn't need to be
                    set for DIPLOMAT files.
    :param video_type: An optional video extension to limit the videos to videos with the specified extension if a
                       directory is passed to the 'videos' parameter.
    """
    cfg = auxiliaryfunctions.read_config(config)
    train_frac = cfg["TrainingFraction"][training_set_index]
    dlc_scorer, __ = auxiliaryfunctions.get_scorer_name(
        cfg, shuffle, train_frac
    )

    video_list = auxiliaryfunctions.get_list_of_videos(_to_str_list(videos), video_type)

    for video in video_list:
        try:
            loc_data, metadata, out_path, h5_path = _get_video_info(video, dlc_scorer, tracker)
        except FileNotFoundError:
            print(f"Unable to find h5 file for video {Path(video).name}. Make sure to run analysis first!")
            continue

        _convert_single_video(Path(video), Path(h5_path), loc_data)


def _convert_single_video(
    video_path: Path,
    h5_path: Path,
    h5_data: pd.DataFrame
):
    print(f"Converting tracks for video '{video_path}'")
    bp_names = []
    counts = {}

    for model, part, attr in h5_data:
        if(attr.startswith("x")):
            counts[part] = counts.get(part, 0) + 1
            if(counts[part] <= 1):
                bp_names.append(part)

    num_outputs = max(counts.values())

    h5_arr = h5_data.to_numpy()
    pose_data = Pose(h5_arr[:, 0::3], h5_arr[:, 1::3],  h5_arr[:, 2::3])

    h5_path = Path(h5_path).resolve()
    csv_path = h5_path.parent / (h5_path.stem + ".csv")

    print(f"Saving converted results to: '{csv_path}'")

    table = to_diplomat_table(num_outputs, bp_names, pose_data)
    save_diplomat_table(table, str(csv_path))
