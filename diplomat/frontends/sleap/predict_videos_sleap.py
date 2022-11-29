import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Type, List

import numpy as np
import diplomat.processing.type_casters as tc
from diplomat.utils.cli_tools import extra_cli_args, Flag
from diplomat.processing.progress_bar import TQDMProgressBar
from diplomat.processing import get_predictor, Config, Predictor, Pose
from .sleap_providers import PredictorExtractor
from .visual_settings import VISUAL_SETTINGS
import time

import sleap
import tensorflow as tf
from inspect import signature

from ...utils.shapes import shape_iterator


def _paths_to_str(paths):
    if(isinstance(paths, list)):
        return [str(p) for p in paths]
    else:
        return str(paths)


def _get_default_value(func, attr, fallback):
    param = signature(func).parameters.get(attr, None)
    return fallback if(param is None) else param.default


def _setup_gpus(use_cpu: bool, gpu_index: Optional[int]):
    # GPU setup...
    if(use_cpu or not sleap.nn.system.is_gpu_system()):
        sleap.nn.system.use_cpu_only()
    else:
        if(gpu_index is None):
            print("Selecting GPU Automatically")
            gpu_free_mem = sleap.nn.system.get_gpu_memory()
            if(len(gpu_free_mem) == 0):
                print("Unable to get gpu info, selecting first GPU.")
                gpu_index = 0
            else:
                gpu_index = np.argmax(gpu_free_mem)
                print(f"Selecting GPU {gpu_index} with {gpu_free_mem[gpu_index]} MiB of free memory.")
        sleap.nn.system.use_gpu(gpu_index)

    sleap.disable_preallocation()

    # Make sure tensorflow is running in the correct mode for evaluation (this issue can happen when the DLC frontend is loaded)...
    if(not tf.executing_eagerly()):
        tf.compat.v1.enable_eager_execution()


def _get_predictor_settings(predictor_cls: Type[Predictor], user_passed_settings) -> Optional[Config]:
    settings_backing = predictor_cls.get_settings()

    if(settings_backing is None):
        return None

    return Config(user_passed_settings, settings_backing)


def _get_video_metadata(
    video_path: Path,
    output_path: Path,
    num_outputs: int,
    video: sleap.Video,
    visual_settings: Config,
    mdl_metadata: dict
) -> Config:
    fps = getattr(video, "fps", 30)

    return Config({
        "fps": fps,
        "duration": video.num_frames / fps,
        "size": tuple(video.shape[1:3]),
        "output-file-path": str(output_path),
        "orig-video-path": str(video_path),  # This may be None if we were unable to find the video...
        "cropping-offset": None,
        "dotsize": visual_settings.dotsize,
        "colormap": visual_settings.colormap,
        "shape_list": shape_iterator(visual_settings.shape_list, num_outputs),
        "alphavalue": visual_settings.alphavalue,
        "pcutoff": visual_settings.pcutoff,
        "line_thickness": visual_settings.get("line_thickness", 1),
        "skeleton": mdl_metadata["skeleton"]
    })


def _pose_to_instance(pose: Pose, frame: int, num_outputs: int, skeleton: sleap.Skeleton) -> List[sleap.PredictedInstance]:
    all_xs = pose.get_x_at(frame, slice(None))
    all_ys = pose.get_y_at(frame, slice(None))
    all_scores = pose.get_prob_at(frame, slice(None))

    instances = []

    for body_i in range(num_outputs):
        xs = all_xs[body_i::num_outputs]
        ys = all_ys[body_i::num_outputs]
        scores = all_scores[body_i::num_outputs]

        instances.append(sleap.PredictedInstance.from_arrays(
            points=np.stack([xs, ys]),
            point_confidences=scores,
            instance_score=np.average(scores),
            skeleton=skeleton,
            track=sleap.Track(0, f"Animal{body_i}"))
        )

    return instances


class PoseLabels:
    def __init__(
        self,
        video: sleap.Video,
        num_outputs: int,
        skeleton: sleap.Skeleton,
        start_frame: int = 0
    ):
        self._frame_list = []
        self._num_outputs = num_outputs
        self._skeleton = skeleton
        self._current_frame = start_frame
        self._video = video

    def append(self, pose: Optional[Pose]):
        if(pose is not None):
            for frame_i in range(pose.get_frame_count()):
                self._frame_list.append(sleap.LabeledFrame(
                    self._video,
                    self._current_frame,
                    _pose_to_instance(pose, frame_i, self._num_outputs, self._skeleton)
                ))
                self._current_frame += 1

    def __len__(self) -> int:
        return self._current_frame

    def to_sleap(self) -> sleap.Labels:
        return sleap.Labels(labeled_frames=self._frame_list)


class Timer:
    def __init__(self):
        self._start_time = None
        self._end_time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.time()

    @property
    def start(self) -> float:
        return self._start_time

    @property
    def end(self) -> float:
        return self._end_time

    @property
    def start_date(self) -> datetime:
        return datetime.fromtimestamp(self._start_time)

    @property
    def end_date(self) -> datetime:
        return datetime.fromtimestamp(self._end_time)

    @property
    def duration(self) -> float:
        return self._end_time - self._start_time


def _attach_run_info(
    labels: sleap.Labels,
    timer: Timer,
    video_path: str,
    output_path: str,
    command: List[str]
) -> sleap.Labels:
    import diplomat

    labels.provenance.update({
        "sleap_version": sleap.__version__,
        "diplomat_version": diplomat.__version__,
        "platform": platform.platform(),
        "command": " ".join(command),
        "data_path": video_path,
        "output_path": output_path,
        "total_elapsed": timer.duration,
        "start_timestamp": str(timer.start_date),
        "finish_timestamp": str(timer.end_date)
    })

    return labels


@extra_cli_args(VISUAL_SETTINGS, auto_cast=False)
@tc.typecaster_function
def analyze_videos(
    config: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    videos: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    batch_size: tc.Optional[int] = None,
    num_outputs: tc.Optional[int] = None,
    predictor: tc.Optional[str] = None,
    predictor_settings: tc.Optional[tc.Dict[str, tc.Any]] = None,
    gpu_index: tc.Optional[int] = None,
    use_cpu: Flag = False,
    **kwargs
):
    """


    :param config:
    :param videos:
    :param gpu_index:
    :param batch_size:
    :param cropping:
    :param num_outputs:
    :param predictor:
    :param predictor_settings:
    :param kwargs:

    :return:
    """
    _setup_gpus(use_cpu, gpu_index)

    batch_size = _get_default_value(sleap.load_model, "batch_size", 4) if(batch_size is None) else batch_size
    num_outputs = 1 if(num_outputs is None) else num_outputs

    print("Loading Model...")
    model = sleap.load_model(_paths_to_str(config), batch_size=batch_size)
    # Get the model extractor...
    mdl_extractor = PredictorExtractor(model)
    mdl_metadata = mdl_extractor.get_metadata()

    predictor_cls = get_predictor("SegmentedFramePassEngine" if(predictor is None) else predictor)
    print(f"Using predictor: '{predictor_cls.get_name()}'")

    if(not predictor_cls.supports_multi_output() and num_outputs > 1):
        raise ValueError(f"Predictor '{predictor_cls.get_name()}' doesn't support multiple outputs!")

    visual_settings = Config(kwargs, VISUAL_SETTINGS)

    if(isinstance(videos, str)):
        videos = [videos]

    for video_path in videos:
        print(f"Running analysis on video: '{video_path}'")
        _analyze_single_video(
            video_path,
            predictor_cls,
            mdl_extractor,
            num_outputs,
            visual_settings,
            mdl_metadata,
            predictor_settings
        )


def _analyze_single_video(
    video_path: str,
    predictor_cls: Type[Predictor],
    mdl_extractor: PredictorExtractor,
    num_outputs: int,
    visual_settings: Config,
    mdl_metadata: dict,
    predictor_settings: Optional[dict]
):
    video_path = Path(video_path).resolve()
    video = sleap.load_video(str(video_path))
    output_path = video_path.parent / (video_path.name + f".diplomat_{predictor_cls.get_name()}.slp")

    video_metadata = _get_video_metadata(video_path, output_path, num_outputs, video, visual_settings, mdl_metadata)
    pred = predictor_cls(
        mdl_metadata["bp_names"], num_outputs, video.num_frames, _get_predictor_settings(predictor_cls, predictor_settings), video_metadata
    )

    labels = PoseLabels(video, num_outputs, mdl_metadata["orig_skeleton"])
    total_frames = 0

    with Timer() as timer:
        with TQDMProgressBar(total=video.num_frames) as prog_bar:
            print("Running the model...")
            for batch in mdl_extractor.extract(video):
                result = pred.on_frames(batch)
                labels.append(result)
                prog_bar.update(batch.get_frame_count())
                total_frames += batch.get_frame_count()

        with TQDMProgressBar(total=video.num_frames - len(labels)) as post_pbar:
            print(f"Running post-processing algorithms...")
            result = pred.on_end(post_pbar)
            labels.append(result)

    if (total_frames != len(labels)):
        raise ValueError(
            f"The predictor algorithm did not return the same amount of frames as are in the video.\n"
            f"Expected Amount: {total_frames}, Actual Amount Returned: {len(labels)}"
        )

    print(f"Saving output to: {output_path}")
    labels = _attach_run_info(labels.to_sleap(), timer, str(video_path), str(output_path), sys.argv)
    labels.save(str(output_path))

    print(f"Finished inference at: {timer.end_date}")
    print(f"Total runtime: {timer.duration} secs")
    print()






















