import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Type, Optional, Tuple, Callable

import cv2
import numpy as np
import diplomat.processing.type_casters as tc
from diplomat.utils.cli_tools import extra_cli_args
from diplomat.processing import get_predictor, Config, TQDMProgressBar, Predictor, Pose, TrackingData
from diplomat.utils.video_info import is_video
from diplomat.utils import frame_store_fmt
from diplomat.core_ops.shared_commands.utils import (
    _get_video_metadata,
    _paths_to_str,
    _get_predictor_settings, Timer
)
from .visual_settings import VISUAL_SETTINGS
from diplomat.frontends import ModelInfo, ModelLike
from diplomat.utils.track_formats import to_diplomat_table, save_diplomat_table
from diplomat.utils.video_io import ContextVideoCapture


@extra_cli_args(VISUAL_SETTINGS, auto_cast=False)
@tc.typecaster_function
def analyze_frames(
    frame_stores: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    batch_size: tc.Optional[int] = None,
    num_outputs: tc.Optional[int] = None,
    predictor: tc.Optional[str] = None,
    predictor_settings: tc.Optional[tc.Dict[str, tc.Any]] = None,
    output_suffix: str = "",
    **kwargs
):
    """
    Run DIPLOMAT tracking on a set of frame store files.

    :param frame_stores: A path or list of paths to frame store ('.dlfs') files to run DIPLOMAT on.
    :param batch_size: An integer, the number of frames to read in from the frame store at a time.
    :param num_outputs: An integer, the number of individuals in the video defaults to 1.
    :param predictor: A string, the name of the predictor to use to perform the task of tracking.
    :param predictor_settings: A dictionary of strings to any values, the settings to use for the predictor. Each predictor offers different settings,
                               see :py:cli:`diplomat predictors list_settings` or :py:func:`~diplomat.predictor_ops.get_predictor_settings` to get
                               the settings a predictor plugin supports.
    :param output_suffix: A string, the suffix to append onto the output .slp file. Defaults to an empty string.
    :param kwargs: The following additional arguments are supported:

                   {extra_cli_args}
    """
    batch_size = 4 if (batch_size is None) else batch_size
    if(num_outputs is None):
        raise ValueError("'num_outputs' is not set! Please set it to the number of bodies you are tracking.")
    else:
        num_outputs = int(num_outputs)

    predictor_cls = get_predictor("SegmentedFramePassEngine" if (predictor is None) else predictor)
    print(f"Using predictor: '{predictor_cls.get_name()}'")

    if (not predictor_cls.supports_multi_output() and num_outputs > 1):
        raise ValueError(f"Predictor '{predictor_cls.get_name()}' doesn't support multiple outputs!")

    visual_settings = Config(kwargs, VISUAL_SETTINGS)

    frame_stores = _paths_to_str(frame_stores)
    if(isinstance(frame_stores, str)):
        frame_stores = [frame_stores]

    for frame_store_path in frame_stores:
        print(f"Running analysis on frame store: '{frame_store_path}'")
        _analyze_frame_store(
            frame_store_path,
            predictor_cls,
            num_outputs,
            visual_settings,
            predictor_settings,
            batch_size,
            output_suffix
        )


def _analyze_frame_store(
    frame_store_path: str,
    predictor_cls: Type[Predictor],
    num_outputs: int,
    visual_settings: Config,
    predictor_settings: Optional[dict],
    batch_size: int,
    output_suffix: str
):
    frame_store_path = Path(frame_store_path).resolve()
    if(not is_video(frame_store_path)):
        raise ValueError("Frame store is missing video data! Please generate frame store with more recent version of diplomat.")
    video_path = frame_store_path
    output_path = frame_store_path.parent / (frame_store_path.name + f".diplomat_{predictor_cls.get_name()}{output_suffix}.csv")

    # LOAD the frame store file...
    with frame_store_path.open("rb") as frame_store:
        with frame_store_fmt.DLFSReader(frame_store) as frame_reader:
            (
                num_f,
                f_h,
                f_w,
                f_rate,
                stride,
                vid_h,
                vid_w,
                off_y,
                off_x,
                bp_lst,
                skeleton
            ) = frame_reader.get_header().to_list()

            video_metadata, __ = _get_video_metadata(
                "diplomat_frame_store",
                video_path,
                output_path,
                num_outputs,
                visual_settings,
                skeleton,
                None if(off_x is None) else (off_y, off_x)
            )

            predictor = predictor_cls(
                bp_lst,
                num_outputs,
                num_f,
                _get_predictor_settings(predictor_cls, predictor_settings),
                video_metadata
            )

            bp_to_idx = {val: i for (i, val) in enumerate(bp_lst)}
            idx_to_keep = [bp_to_idx.get(val, None) for val in bp_lst]

            if(any(i is None for i in idx_to_keep)):
                raise ValueError(
                    f"Unable to use frame store, body part(s) "
                    f"{[i for i, val in enumerate(idx_to_keep) if(val is None)]} "
                    f"are missing from the frame store."
                )

            labels = np.zeros((num_f, 3 * len(bp_lst) * num_outputs))
            frames_done = 0
            read_frames = 0

            with predictor as pred:
                with Timer() as timer:
                    with TQDMProgressBar(total=num_f) as prog_bar:
                        print("Running the predictor on frames...")
                        while(frame_reader.has_next()):
                            batch = frame_reader.read_frames(min(batch_size, num_f - read_frames))

                            # Fix the batch to only have what is needed...
                            batch.set_source_map(batch.get_source_map()[:, :, :, idx_to_keep])
                            if(batch.get_offset_map() is not None):
                                batch.set_offset_map(batch.get_offset_map()[:, :, :, idx_to_keep])

                            pose = pred.on_frames(batch)
                            if(pose is not None):
                                labels[frames_done:frames_done + pose.get_frame_count()] = pose.get_all()
                                frames_done += pose.get_frame_count()
                            read_frames += batch.get_frame_count()
                            prog_bar.update(batch.get_frame_count())

                    with TQDMProgressBar(total=num_f - frames_done) as post_pbar:
                        print(f"Running post-processing algorithms...")
                        poses = pred.on_end(post_pbar)
                        if(poses is not None):
                            labels[frames_done:frames_done + pose.get_frame_count()] = pose.get_all()
                            frames_done += pose.get_frame_count()

            if (frames_done != len(labels)):
                raise ValueError(
                    f"The predictor algorithm did not return the same amount of frames as are in the video.\n"
                    f"Expected Amount: {frames_done}, Actual Amount Returned: {len(labels)}"
                )

            print(f"Saving output to: {output_path}")
            p = Pose.empty_pose(labels.shape[0], labels.shape[1] // 3)
            p.get_all()[:] = labels
            save_diplomat_table(to_diplomat_table(num_outputs, bp_lst, p), str(output_path))

            print(f"Finished inference at: {timer.end_date}")
            print(f"Total runtime: {timer.duration} secs")
            print()


@extra_cli_args(VISUAL_SETTINGS, auto_cast=False)
@tc.typecaster_function
def analyze_videos(
    model: ModelLike,
    model_info: ModelInfo,
    videos: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    predictor: tc.Optional[str] = None,
    predictor_settings: tc.Optional[tc.Dict[str, tc.Any]] = None,
    output_suffix: str = "",
    **kwargs
):
    batch_size = model_info["batch_size"]
    num_outputs = model_info["num_outputs"]
    if(batch_size is None):
        batch_size = 4
    if(num_outputs is None):
        raise ValueError("'num_outputs' is not set! Please set it to the number of bodies you are tracking.")

    predictor_cls = get_predictor("SegmentedFramePassEngine" if (predictor is None) else predictor)
    print(f"Using predictor: '{predictor_cls.get_name()}'")

    if (not predictor_cls.supports_multi_output() and num_outputs > 1):
        raise ValueError(f"Predictor '{predictor_cls.get_name()}' doesn't support multiple outputs!")

    visual_settings = Config(kwargs, VISUAL_SETTINGS)

    videos = _paths_to_str(videos)
    if(isinstance(videos, str)):
        videos = [videos]

    for video in _paths_to_str(videos):
        _analyze_single_video(
            video,
            model_info,
            model,
            predictor_cls,
            num_outputs,
            visual_settings,
            predictor_settings,
            batch_size,
            output_suffix
        )



def _analyze_single_video(
    video_path: str,
    model_info: ModelInfo,
    model: Callable[[np.ndarray], TrackingData],
    predictor_cls: Type[Predictor],
    num_outputs: int,
    visual_settings: Config,
    predictor_settings: Optional[dict],
    batch_size: int,
    output_suffix: str
):
    video_path = Path(video_path).resolve()
    if (not is_video(video_path)):
        raise ValueError(f"Passed file: {video_path} is not a video!")
    output_path = video_path.parent / (video_path.name + f".diplomat_{predictor_cls.get_name()}{output_suffix}.csv")
    bp_lst = model_info["bp_names"]

    video_metadata, num_f = _get_video_metadata(
        model_info["frontend"],
        video_path,
        output_path,
        num_outputs,
        visual_settings,
        model_info["skeleton"],
        None
    )

    # LOAD the frame store file...
    with ContextVideoCapture(str(video_path)) as video:
        predictor = predictor_cls(
            bp_lst,
            num_outputs,
            num_f,
            _get_predictor_settings(predictor_cls, predictor_settings),
            video_metadata
        )

        bp_to_idx = {val: i for (i, val) in enumerate(bp_lst)}
        idx_to_keep = [bp_to_idx.get(val, None) for val in bp_lst]

        if (any(i is None for i in idx_to_keep)):
            raise ValueError(
                f"Unable to use frame store, body part(s) "
                f"{[i for i, val in enumerate(idx_to_keep) if (val is None)]} "
                f"are missing from the frame store."
            )

        labels = np.zeros((num_f, 3 * len(bp_lst) * num_outputs))
        frames_done = 0
        read_frames = 0
        buffer = None

        with predictor as pred:
            with Timer() as timer:
                with TQDMProgressBar(total=num_f) as prog_bar:
                    print("Running the predictor on frames...")
                    while True:
                        frm_count, buffer = _read_frames(video, batch_size, buffer)
                        if(frm_count == 0):
                            break

                        batch = model(buffer[:frm_count])
                        # Fix the batch to only have what is needed (remove filtered body parts)...
                        batch.set_source_map(batch.get_source_map()[:, :, :, idx_to_keep])
                        if (batch.get_offset_map() is not None):
                            batch.set_offset_map(batch.get_offset_map()[:, :, :, idx_to_keep])

                        pose = pred.on_frames(batch)
                        if (pose is not None):
                            labels[frames_done:frames_done + pose.get_frame_count()] = pose.get_all()
                            frames_done += pose.get_frame_count()
                        read_frames += batch.get_frame_count()
                        prog_bar.update(batch.get_frame_count())

                with TQDMProgressBar(total=num_f - frames_done) as post_pbar:
                    print(f"Running post-processing algorithms...")
                    poses = pred.on_end(post_pbar)
                    if (poses is not None):
                        labels[frames_done:frames_done + poses.get_frame_count()] = poses.get_all()
                        frames_done += poses.get_frame_count()

        if (frames_done != len(labels)):
            raise ValueError(
                f"The predictor algorithm did not return the same amount of frames as are in the video.\n"
                f"Expected Amount: {frames_done}, Actual Amount Returned: {len(labels)}"
            )

        print(f"Saving output to: {output_path}")
        p = Pose.empty_pose(labels.shape[0], labels.shape[1] // 3)
        p.get_all()[:] = labels
        save_diplomat_table(to_diplomat_table(num_outputs, bp_lst, p), str(output_path))

        print(f"Finished inference at: {timer.end_date}")
        print(f"Total runtime: {timer.duration} secs")
        print()


def _read_frames(video: cv2.VideoCapture, batch_size: int, buffer: np.ndarray = None):
    i = 0

    while(i < batch_size):
        got, frm = video.read()
        if(not got):
            break
        if(buffer is None):
            buffer = np.zeros((batch_size, *frm.shape), dtype=frm.dtype)
        # BGR to RGB...
        buffer[i] = frm[..., ::-1]
        i += 1

    return i, buffer
