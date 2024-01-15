from pathlib import Path

from .sleap_importer import sleap

import diplomat.processing.type_casters as tc
from diplomat.utils.cli_tools import extra_cli_args
from diplomat.processing import Config, Pose
from diplomat.utils.tweak_ui import TweakUI


from .visual_settings import VISUAL_SETTINGS
from .run_utils import (
    _paths_to_str,
    _get_video_metadata,
    _to_diplomat_poses,
    PoseLabels, _load_config,
)
from .sleap_providers import SleapMetadata


@extra_cli_args(VISUAL_SETTINGS, auto_cast=False)
@tc.typecaster_function
def tweak_videos(
    config: tc.PathLike,
    videos: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    **kwargs
):
    """
    Make minor modifications and tweaks to sleap results (stored in .slp files) using DIPLOMAT's interactive UI.

    :param config: The path (or list of paths) to the sleap model(s) used for inference, each as either as a folder or zip file.
    :param videos: Paths to the sleap label files, or .slp files, to make minor modifications to, NOT the video files.
    :param kwargs: The following additional arguments are supported:

                   {extra_cli_args}
    """
    _load_config(_paths_to_str(config))

    label_paths = _paths_to_str(videos)
    label_paths = [label_paths] if(isinstance(label_paths, str)) else label_paths

    visual_cfg = Config(kwargs, VISUAL_SETTINGS)

    for label_path in label_paths:
        _tweak_video_single(label_path, visual_cfg)


def _tweak_video_single(
    label_file: str,
    visual_cfg: Config
):
    print(f"Making modifications to: '{label_file}'")
    labels = sleap.load_file(label_file)
    num_outputs, pose_obj, video, skeleton = _to_diplomat_poses(labels)
    mdl_metadata = SleapMetadata(bp_names=skeleton.node_names, skeleton=skeleton.edge_names, orig_skeleton=skeleton)
    video_meta = _get_video_metadata(Path(video.filename), Path(label_file), num_outputs, video, visual_cfg, mdl_metadata, None)
    print(video_meta)

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

    names = [name if(i == 0) else f"{name}{i}" for name in mdl_metadata["bp_names"] for i in range(num_outputs)]
    ui_manager.tweak(None, video.filename, pose_obj, names, dict(video_meta), num_outputs, None, on_end)
