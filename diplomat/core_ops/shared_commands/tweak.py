import cv2
from diplomat.core_ops.shared_commands.visual_settings import VISUAL_SETTINGS
from diplomat.processing import Config, Pose
from diplomat.utils.cli_tools import extra_cli_args
from diplomat.utils.tweak_ui import TweakUI
import diplomat.processing.type_casters as tc
from diplomat.utils.track_formats import to_diplomat_pose, save_diplomat_table, to_diplomat_table
from diplomat.utils.video_info import get_frame_count_robust_fast
from diplomat.utils.video_io import ContextVideoCapture
from diplomat.utils.shapes import shape_iterator
from diplomat.core_ops.shared_commands.utils import _fix_path_pairs, _get_track_loaders, _load_tracks_from_loaders


@extra_cli_args(VISUAL_SETTINGS, auto_cast=False)
@tc.typecaster_function
def tweak_videos(
    videos: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    csvs: tc.PathLike,
    **kwargs
):
    """
    Make minor modifications and tweaks to arbitrary csv files using DIPLOMAT's light interactive UI.

    :param videos: Paths to video file(s) corresponding to the provided csv files.
    :param csvs: The path (or list of paths) to the csv file(s) to edit. If not csv files, will attempt to detect if
                 file is frontend specific and convert it to a csv.
    :param kwargs: The following additional arguments are supported:

                   {extra_cli_args}
    """
    csvs, videos = _fix_path_pairs(csvs, videos)
    visual_cfg = Config(kwargs, VISUAL_SETTINGS)

    for c, v in zip(csvs, videos):
        _tweak_video_single(str(c), str(v), visual_cfg)


def _get_video_meta(
    video: str,
    visual_settings: Config,
    output_file: str,
    num_outputs: int
):
    with ContextVideoCapture(str(video)) as vid_cap:
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        w, h = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = get_frame_count_robust_fast(vid_cap)

    return frame_count, {
        "fps": fps,
        "duration": frame_count / fps,
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
    pose_table = _load_tracks_from_loaders(_get_track_loaders(True), csv)
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
    frame_count, video_meta = _get_video_meta(video, visual_cfg, csv, num_outputs)
    if poses.get_frame_count() != frame_count:
        print(f"Warning: Passed CSV doesn't have same number of frames as video (csv: {poses.get_frame_count()}, video: {frame_count}), adjusting to match video length.")
        min_count = min(poses.get_frame_count(), frame_count)
        new_poses = Pose.empty_pose(frame_count, poses.get_bodypart_count())
        new_poses.get_all()[:] = float("nan")
        new_poses.get_all()[:min_count] = poses.get_all()[:min_count]
        poses = new_poses
    ui_manager.tweak(None, video, poses, all_names, video_meta, num_outputs, None, on_end)
