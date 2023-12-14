from .sleap_importer import tf
import platform
import time
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Optional, Type, List, Tuple, Iterable, Dict
from .sleap_importer import sleap
import numpy as np
from diplomat.processing import Predictor, Config, Pose
from diplomat.utils.shapes import shape_iterator


def _frame_iter(
    frame: sleap.LabeledFrame,
    skeleton: sleap.Skeleton,
    track_to_idx: Dict[sleap.Track, int]
) -> Iterable[sleap.PredictedInstance]:
    import sleap
    for inst in frame.instances:
        if((inst.track is not None) and isinstance(inst, sleap.PredictedInstance) and (inst.skeleton == skeleton)):
            yield track_to_idx[inst.track], inst


def _to_diplomat_poses(labels: sleap.Labels) -> Tuple[int, Pose, sleap.Video, sleap.Skeleton]:
    video = labels.video
    skeleton = labels.skeleton

    tracks_to_idx = {track: i for i, track in enumerate(labels.tracks)}
    num_outputs = len(labels.tracks)

    pose_obj = Pose.empty_pose(labels.get_labeled_frame_count(), len(skeleton.node_names) * num_outputs)

    for f_i, frame in enumerate(labels.frames(video)):
        for i_i, inst in _frame_iter(frame, skeleton, tracks_to_idx):
            inst_data = inst.points_and_scores_array

            pose_obj.set_x_at(f_i, slice(i_i, None, num_outputs), inst_data[:, 0])
            pose_obj.set_y_at(f_i, slice(i_i, None, num_outputs), inst_data[:, 1])
            pose_obj.set_prob_at(f_i, slice(i_i, None, num_outputs), inst_data[:, 2])

    return num_outputs, pose_obj, video, skeleton


def _paths_to_str(paths):
    if(isinstance(paths, (list, tuple))):
        return [str(p) for p in paths]
    else:
        return str(paths)


def _load_config(paths):
    try:
        paths = [paths] if(isinstance(paths, str)) else paths

        if(len(paths) < 1):
            raise ValueError(f"No configuration files passed to open!")

        return [sleap.load_config(p) for p in paths]
    except IOError as e:
        raise type(e)(f"Unable to load provided sleap config: '{repr(e)}'")


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

    # Make sure tensorflow is running in the correct mode for evaluation
    # (this issue can happen when the DLC frontend is loaded)...
    if(not tf.executing_eagerly()):
        tf.compat.v1.enable_eager_execution()


def _get_predictor_settings(predictor_cls: Type[Predictor], user_passed_settings) -> Optional[Config]:
    settings_backing = predictor_cls.get_settings()

    if(settings_backing is None):
        return None

    return Config(user_passed_settings, settings_backing)


def _skeleton_conv(skeleton, fallback_skeleton, part_list):
    if(skeleton is False):
        return None
    if(skeleton is None):
        return fallback_skeleton

    if(skeleton is True):
        skeleton = list(part_list)

    part_set = set(part_list)
    def _validate_part(part):
        if(part not in part_set):
            raise ValueError(f"Part {part} not a valid body part! (Valid parts are: {part_set})")
        return part

    if(isinstance(skeleton, dict)):
        skel_list = []
        # convert to list of tuples of strings...
        for key, val in skeleton.items():
            if(isinstance(val, str)):
                skel_list.append((_validate_part(key), _validate_part(val)))
            else:
                key = _validate_part(key)
                skel_list.extend([(key, _validate_part(sub_val)) for sub_val in val])

        return skel_list
    else:
        # Force into one of two forms...
        for val in skeleton:
            if(isinstance(val, str)):
                return [
                    (_validate_part(a), _validate_part(b))
                    for i, a in enumerate(skeleton)
                    for j, b in enumerate(skeleton[i + 1:], i + 1)
                ]
            else:
                return [(_validate_part(a), _validate_part(b)) for a, b in skeleton]

        return None  # No skeleton if we made it through the loop...


def _get_video_metadata(
    video_path: Optional[Path],
    output_path: Path,
    num_outputs: int,
    video: sleap.Video,
    visual_settings: Config,
    mdl_metadata: dict,
    crop_loc: Optional[Tuple[int, int]] = None
) -> Config:
    fps = getattr(video, "fps", 30)
    skel = _skeleton_conv(visual_settings.skeleton, mdl_metadata["skeleton"], mdl_metadata["bp_names"])

    return Config({
        "fps": fps,
        "duration": video.num_frames / fps,
        "size": tuple(video.shape[1:3]),
        "output-file-path": str(output_path),
        "orig-video-path": str(video_path) if(video_path is not None) else None,  # This may be None if we were unable to find the video...
        "cropping-offset": crop_loc,
        "dotsize": visual_settings.dotsize,
        "colormap": visual_settings.colormap,
        "shape_list": shape_iterator(visual_settings.shape_list, num_outputs),
        "alphavalue": visual_settings.alphavalue,
        "pcutoff": visual_settings.pcutoff,
        "line_thickness": visual_settings.get("line_thickness", 1),
        "skeleton": skel,
        "frontend": "sleap"
    })


def _pose_to_instance(
    pose: Pose,
    frame: int,
    num_outputs: int,
    skeleton: sleap.Skeleton,
    track_objs: List[sleap.Track]
) -> List[sleap.PredictedInstance]:
    all_xs = pose.get_x_at(frame, slice(None))
    all_ys = pose.get_y_at(frame, slice(None))
    all_scores = pose.get_prob_at(frame, slice(None))

    instances = []

    for body_i in range(num_outputs):
        xs = all_xs[body_i::num_outputs]
        ys = all_ys[body_i::num_outputs]
        scores = all_scores[body_i::num_outputs]

        instances.append(sleap.PredictedInstance.from_arrays(
            points=np.stack([xs, ys]).T,
            point_confidences=scores,
            instance_score=np.sum(scores),
            skeleton=skeleton,
            track=track_objs[body_i]
        ))

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
        self._track_objs = [sleap.Track(0, f"track_{i}") for i in range(num_outputs)]

    def append(self, pose: Optional[Pose]):
        if(pose is not None):
            for frame_i in range(pose.get_frame_count()):
                self._frame_list.append(sleap.LabeledFrame(
                    self._video,
                    self._current_frame,
                    _pose_to_instance(pose, frame_i, self._num_outputs, self._skeleton, self._track_objs)
                ))
                self._current_frame += 1

    def __len__(self) -> int:
        return self._current_frame

    def to_sleap(self) -> sleap.Labels:
        return sleap.Labels(labeled_frames=self._frame_list)


class Timer:
    def __init__(self, start_time: Optional[float] = None, end_time: Optional[float] = None):
        self._start_time = start_time
        self._end_time = end_time

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
    import sleap

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
