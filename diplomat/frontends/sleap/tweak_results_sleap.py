from pathlib import Path
from typing import Iterable

import sleap

import diplomat.processing.type_casters as tc
from diplomat.utils.cli_tools import extra_cli_args
from diplomat.processing import Config, Pose
from diplomat.utils.tweak_ui import TweakUI


from .visual_settings import VISUAL_SETTINGS
from .run_utils import (
    _paths_to_str,
    _get_video_metadata,
    PoseLabels,
)
from .sleap_providers import SleapMetadata


@extra_cli_args(VISUAL_SETTINGS, auto_cast=False)
@tc.typecaster_function
def tweak_videos(
    config: tc.PathLike,
    videos: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    **kwargs
):
    label_paths = _paths_to_str(videos)
    label_paths = [label_paths] if(isinstance(label_paths, str)) else label_paths

    visual_cfg = Config(kwargs, VISUAL_SETTINGS)

    for label_path in label_paths:
        _tweak_video_single(str(config), label_path, visual_cfg)


def _frame_iter(
    frame: sleap.LabeledFrame,
    skeleton: sleap.Skeleton
) -> Iterable[sleap.PredictedInstance]:
    for inst in frame.instances:
        if(isinstance(inst, sleap.PredictedInstance) and (inst.skeleton == skeleton)):
            yield inst


def _tweak_video_single(
    config: str,
    label_file: str,
    visual_cfg: Config
):
    print(f"Making modifications to: '{label_file}'")
    model = sleap.load_model(config)
    labels = sleap.load_file(label_file)

    video = labels.video
    skeleton = labels.skeleton
    mdl_metadata = SleapMetadata(bp_names=skeleton.node_names, skeleton=skeleton.edge_names, orig_skeleton=skeleton)
    num_outputs = max(sum(1 for _ in _frame_iter(frame, skeleton)) for frame in labels.frames(video))

    pose_obj = Pose.empty_pose(labels.get_labeled_frame_count(), len(mdl_metadata["bp_names"]) * num_outputs)

    for f_i, frame in enumerate(labels.frames(video)):
        for i_i, inst in zip(range(num_outputs), _frame_iter(frame, skeleton)):
            inst_data = inst.points_and_scores_array

            pose_obj.set_x_at(f_i, slice(i_i, None, num_outputs), inst_data[:, 0])
            pose_obj.set_y_at(f_i, slice(i_i, None, num_outputs), inst_data[:, 1])
            pose_obj.set_prob_at(f_i, slice(i_i, None, num_outputs), inst_data[:, 2])

    video_meta = _get_video_metadata(Path(video.filename), Path(label_file), num_outputs, video, visual_cfg, mdl_metadata, None)

    ui_manager = TweakUI()

    def on_end(save: bool, poses: Pose):
        if(save):
            print("Saving results...")
            pose_conv = PoseLabels(video, num_outputs, skeleton)
            pose_conv.append(poses)
            pose_conv.to_sleap().save(label_file)
            print("Results saved!")
        else:
            print("Operation canceled...")

    ui_manager.tweak(None, video.filepath, pose_obj, mdl_metadata["bp_names"], dict(video_meta), num_outputs, None, on_end)
