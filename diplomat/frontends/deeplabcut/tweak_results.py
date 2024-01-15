from typing import MutableMapping, Any
import pandas as pd
from .dlc_importer import auxiliaryfunctions
from .label_videos_dlc import _to_str_list, _get_video_info
import diplomat.processing.type_casters as tc
from pathlib import Path
from diplomat.utils.tweak_ui import TweakUI
from diplomat.processing import Pose
from diplomat.utils.shapes import shape_iterator


@tc.typecaster_function
def tweak_videos(
    config: tc.PathLike,
    videos: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    shuffle: int = 1,
    training_set_index: int = 0,
    tracker: str = "",
    video_type: str = "",
):
    """
    Make minor modifications and tweaks to tracked results produced by DEEPLABCUT (or DIPLOMAT) using the interactive
    UI.

    :param config: The path to the config.yaml file for the project.
    :param videos: A single video or list of videos to tweak, passed in order to modify.
    :param shuffle: The shuffle index of the model used to track the video. Defaults to 1.
    :param training_set_index: The training index of the model used. Defaults to 0.
    :param tracker: String, the extension of the deeplabcut tracker used, used to find the h5 file. Doesn't need to be set for DIPLOMAT files.
    :param video_type: An optional video extension to limit the videos to videos with the specified extension if a directory is passed to the
                       'videos' parameter.
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

        _tweak_single_video(cfg, Path(video), metadata, Path(h5_path), loc_data)


def _tweak_single_video(
    config: MutableMapping[str, Any],
    video_path: Path,
    pickle_info: MutableMapping[str, Any],
    h5_path: Path,
    h5_data: pd.DataFrame
):
    ui_manager = TweakUI()
    cropping = [int(v) for v in pickle_info["cropping_parameters"]] if(pickle_info["cropping"]) else None

    bp_names = []
    counts = {}

    for model, part, attr in h5_data:
        if(attr.startswith("x")):
            # Part + Number...
            bp_names.append(part + attr[1:])
            counts[part] = counts.get(part, 0) + 1

    num_outputs = max(counts.values())
    video_metadata = {
        "fps": pickle_info["fps"],
        "output-file-path": str(h5_path),
        "orig-video-path": str(video_path),
        "duration": pickle_info["nframes"] / pickle_info["fps"],
        "size": tuple(int(v) for v in pickle_info["frame_dimensions"]),
        "cropping-offset": None if(cropping is None) else (cropping[0], cropping[2]),
        "dotsize": config["dotsize"],
        "colormap": config.get("diplomat_colormap", config["colormap"]),
        "shape_list": shape_iterator(config.get("shape_list", None), num_outputs),
        "alphavalue": config["alphavalue"],
        "pcutoff": config["pcutoff"],
        "line_thickness": config.get("line_thickness", 1),
        "skeleton": config.get("skeleton", None)
    }

    h5_arr = h5_data.to_numpy()

    def on_save(want_save, poses):
        if(want_save):
            results = pd.DataFrame(poses.get_all(), columns=h5_data.columns, index=h5_data.index)
            csv_path = h5_path.parent / (h5_path.stem + ".csv")
            if(csv_path.exists()):
                results.to_csv(str(csv_path))
            results.to_hdf(str(h5_path), "df_with_missing", format="table", mode="w")

    ui_manager.tweak(
        None,
        video_path,
        Pose(h5_arr[:, 0::3], h5_arr[:, 1::3],  h5_arr[:, 2::3]),
        bp_names,
        video_metadata,
        num_outputs,
        None if(cropping is None) else (cropping[0], cropping[2], int(cropping[1] - cropping[0]), int(cropping[3] - cropping[2])),
        on_save
    )
