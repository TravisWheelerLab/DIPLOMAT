import cv2
from diplomat.frontends.sleap.visual_settings import VISUAL_SETTINGS
from diplomat.processing import Config, Pose
from diplomat.utils.cli_tools import extra_cli_args
from diplomat.utils.tweak_ui import TweakUI
import diplomat.processing.type_casters as tc
from diplomat.utils.track_formats import load_diplomat_table, to_diplomat_pose, save_diplomat_table, to_diplomat_table
from diplomat.utils.video_io import ContextVideoCapture
from diplomat.utils.shapes import shape_iterator

from .csv_utils import _fix_paths


@extra_cli_args(VISUAL_SETTINGS, auto_cast=False)
@tc.typecaster_function
def tweak_videos(
    config: tc.PathLike,
    videos: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    **kwargs
):
    """
    Make minor modifications and tweaks to arbitrary csv files using DIPLOMAT's light interactive UI.

    :param config: The path (or list of paths) to the csv file(s) to edit.
    :param videos: Paths to video file(s) corresponding to the provided csv files.
    :param kwargs: The following additional arguments are supported:

                   {extra_cli_args}
    """
    config, videos = _fix_paths(config, videos)
    visual_cfg = Config(kwargs, VISUAL_SETTINGS)

    for c, v in zip(config, videos):
        _tweak_video_single(str(c), str(v), visual_cfg)


def _get_video_meta(
    video: str,
    num_frames: int,
    visual_settings: Config,
    output_file: str,
    num_outputs: int
):
    with ContextVideoCapture(str(video)) as vid_cap:
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        w, h = vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH), vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    return {
        "fps": fps,
        "duration": num_frames / fps,
        "size": (h, w),
        "output-file-path": str(output_file),
        "orig-video-path": video,
        "cropping-offset": None,
        "dotsize": visual_settings.dotsize,
        "colormap": visual_settings.colormap,
        "shape_list": shape_iterator(visual_settings.shape_list, num_outputs),
        "alphavalue": visual_settings.alphavalue,
        "pcutoff": visual_settings.pcutoff,
        "line_thickness": visual_settings.get("line_thickness", 1),
        "skeleton": visual_settings.skeleton
    }


def _tweak_video_single(
    csv: str,
    video: str,
    visual_cfg: Config
):
    print(f"Making modifications to: '{csv}' (video: '{video}')")
    pose_table = load_diplomat_table(csv)
    poses, bp_names, num_outputs = to_diplomat_pose(pose_table)

    ui_manager = TweakUI()

    def on_end(save: bool, p: Pose):
        if(save):
            print("Saving results...")
            save_diplomat_table(to_diplomat_table(num_outputs, bp_names, p), csv)
            print("Results saved!")
        else:
            print("Operation canceled...")

    all_names = [name if(i == 0) else f"{name}{i}" for name in bp_names for i in range(num_outputs)]
    video_meta = _get_video_meta(video, poses.get_frame_count(), visual_cfg, csv, num_outputs)
    ui_manager.tweak(None, video, poses, all_names, video_meta, num_outputs, None, on_end)
