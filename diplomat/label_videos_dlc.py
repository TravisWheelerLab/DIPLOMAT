import pickle
from os import PathLike
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple

import cv2
import numpy as np
import tqdm
from deeplabcut import auxiliaryfunctions

import pandas as pd

from diplomat.processing import *
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors

def cv2_fourcc_string(val) -> int:
    return int(cv2.VideoWriter_fourcc(*val))

Pathy = Union[PathLike, str]

LABELED_VIDEO_SETTINGS = {
    "skeleton_color": ("black", mpl_colors.to_rgba, "Color of the skeleton."),
    "pcutoff": (0.1, type_casters.RangedFloat(0, 1), "The probability to cutoff results below."),
    "dotsize": (4, int, "The size of the dots."),
    "alphavalue": (0.7, type_casters.RangedFloat(0, 1), "The alpha value of the dots."),
    "colormap": (plt.get_cmap(), plt.get_cmap, "The colormap to use for tracked points in the video."),
    "line_thickness": (1, int, "Thickness of lines drawn."),
    "antialiasing": (True, bool, "Use antialiasing when drawing points."),
    "draw_hidden_tracks": (True, bool, "Whether or not to draw locations under the pcutoff value."),
    "output_codec": ("mp4v", cv2_fourcc_string, "The codec to use for the labeled video...")
}

class EverythingSet:
    def __contains__(self, item):
        return True

def create_labeled_videos(
    config: Pathy,
    videos: List[Pathy],
    body_parts_to_plot: Optional[List[str]] = None,
    shuffle: int = 1,
    training_set_index: int = 0,
    video_type: str = "",
    **kwargs
) -> None:
    cfg = auxiliaryfunctions.read_config(config)
    train_frac = cfg["TrainingFraction"][training_set_index]

    dlc_scorer, __ = auxiliaryfunctions.GetScorerName(
        cfg, shuffle, train_frac
    )

    plotting_settings = Config({}, LABELED_VIDEO_SETTINGS)
    plotting_settings.extract(cfg)
    plotting_settings.update(kwargs)

    video_list = auxiliaryfunctions.get_list_of_videos(videos, video_type)

    for video in video_list:
        try:
            loc_data, metadata, out_path = _get_video_info(video, dlc_scorer)

            if(Path(out_path).exists()):
                print(f"Labeled video {Path(video).name} already exists...")
                continue
        except FileNotFoundError:
            print(f"Unable to find h5 file for video {Path(video).name}. Make sure to run analysis first!")
            continue


        _create_video_single(
            Path(video),
            loc_data,
            metadata,
            out_path,
            plotting_settings,
            body_parts_to_plot
        )

def _get_video_info(video: Pathy, dlc_scorer: str) -> Tuple[pd.DataFrame, Dict[str, Any], PathLike]:
    video = Path(video)
    parent_folder = video.resolve().parent

    df_data = pd.read_hdf(str(parent_folder / (video.stem + dlc_scorer + ".h5")), "df_with_missing")

    with (parent_folder / (video.stem + dlc_scorer + "_meta.pickle")).open("rb") as f:
        metadata = pickle.load(f)["data"]

    final_video_path = parent_folder / (video.stem + "_labeled.mp4")

    return (df_data, metadata, final_video_path)

def _to_cv2_color(color: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    r, g, b, a = [min(255, max(0, int(val * 256))) for val in color]
    return (b, g, r, a)

def _create_video_single(
    video_path: Pathy,
    location_data: pd.DataFrame,
    pickle_info: Dict[str, Any],
    export_path: Pathy,
    plotting_settings: Config,
    body_parts_to_plot: Optional[List[str]] = None,
) -> None:
    print(f"Creating labeled video: {Path(video_path).name}")
    unlabeled_video = cv2.VideoCapture(str(video_path))
    x_off, y_off = 0, 0
    x_off2 = int(unlabeled_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_off2 = int(unlabeled_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if(pickle_info["cropping"]):
        x_off, x_off2, y_off, y_off2 = [int(v) for v in pickle_info["cropping_parameters"]]

    labeled_video = cv2.VideoWriter(
        str(export_path),
        plotting_settings.output_codec,
        float(unlabeled_video.get(cv2.CAP_PROP_FPS)),
        (x_off2 - x_off, y_off2 - y_off)
    )

    if(not labeled_video.isOpened()):
        raise IOError("Unable to create labeled video with the specified location and codec.")

    # Compute the body parts to look up...
    body_parts_to_plot = EverythingSet() if(body_parts_to_plot is None) else set(body_parts_to_plot)
    body_part_data = []
    for col in location_data:
        scorer, body_part, coord = col
        if(body_part in body_parts_to_plot and coord.startswith("x")):
            # Views into the data_frame...
            body_part_data.append([
                location_data[scorer, body_part, coord],
                location_data[scorer, body_part, "y" + coord[1:]],
                location_data[scorer, body_part, "likelihood" + coord[1:]]
            ])

    progress = tqdm.tqdm(total=location_data.shape[0])
    i = 0

    # Now begin writing frames...
    while(unlabeled_video.isOpened() and labeled_video.isOpened() and i < location_data.shape[0]):
        got_frame, frame = unlabeled_video.read()

        if(not got_frame):
            break

        frame = frame[y_off:y_off2, x_off:x_off2]
        overlay = frame.copy()

        for bp_idx, (bp_x, bp_y, bp_p) in enumerate(body_part_data):
            if(bp_p[i] > plotting_settings.pcutoff):
                color = plotting_settings.colormap(bp_idx / max(1, len(body_part_data) - 1))
                cv2.circle(
                    overlay,
                    (int(bp_x[i]), int(bp_y[i])),
                    int(plotting_settings.dotsize),
                    _to_cv2_color(color[:3] + (1,)),
                    -1,
                    cv2.LINE_AA if(plotting_settings.antialiasing) else None
                )
            elif(plotting_settings.draw_hidden_tracks):
                color = plotting_settings.colormap(bp_idx / max(1, len(body_part_data) - 1))
                cv2.circle(
                    overlay,
                    (int(bp_x[i]), int(bp_y[i])),
                    int(plotting_settings.dotsize),
                    _to_cv2_color(color[:3] + (1,)),
                    plotting_settings.line_thickness,
                    cv2.LINE_AA if(plotting_settings.antialiasing) else None
                )

        labeled_video.write(cv2.addWeighted(
            overlay, plotting_settings.alphavalue, frame, 1 - plotting_settings.alphavalue, 0
        ))

        progress.update()
        i += 1

    unlabeled_video.release()
    labeled_video.release()
