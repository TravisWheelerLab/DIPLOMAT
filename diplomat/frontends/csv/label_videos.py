from pathlib import Path
from typing import Tuple

from diplomat.utils.cli_tools import extra_cli_args
from diplomat.frontends.sleap.visual_settings import FULL_VISUAL_SETTINGS
import diplomat.processing.type_casters as tc
from diplomat.utils.track_formats import load_diplomat_table, to_diplomat_pose
from diplomat.utils.video_io import ContextVideoWriter, ContextVideoCapture
from diplomat.utils.shapes import CV2DotShapeDrawer, shape_iterator
from diplomat.processing import Config, TQDMProgressBar
from diplomat.utils.colormaps import iter_colormap

import cv2

from .csv_utils import _fix_paths


@extra_cli_args(FULL_VISUAL_SETTINGS, auto_cast=False)
@tc.typecaster_function
def label_videos(
    config: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    videos: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    body_parts_to_plot: tc.Optional[tc.List[str]] = None,
    video_extension: str = "mp4",
    **kwargs
):
    """
    Labeled videos with arbitrary csv files in diplomat's csv format.

    :param config: The path (or list of paths) to the csv file(s) to label the videos with.
    :param videos: Paths to video file(s) corresponding to the provided csv files.
    :param body_parts_to_plot: A set or list of body part names to label, or None, indicating to label all parts.
    :param video_extension: The file extension to use on the created labeled video, excluding the dot.
                            Defaults to 'mp4'.
    :param kwargs: The following additional arguments are supported:

                   {extra_cli_args}
    """
    config, videos = _fix_paths(config, videos)
    visual_settings = Config(kwargs, FULL_VISUAL_SETTINGS)

    for c, v in zip(config, videos):
        _label_videos_single(str(c), str(v), body_parts_to_plot, video_extension, visual_settings)


class EverythingSet:
    def __contains__(self, item):
        return True


def _to_cv2_color(color: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    r, g, b, a = [min(255, max(0, int(val * 256))) for val in color]
    return (b, g, r, a)


def _label_videos_single(
    csv: str,
    video: str,
    body_parts_to_plot: tc.Optional[tc.List[str]],
    video_extension: str,
    visual_settings: Config
):
    pose_data = load_diplomat_table(csv)
    poses, bp_names, num_outputs = to_diplomat_pose(pose_data)
    video_extension = video_extension if(video_extension.startswith(".")) else f".{video_extension}"
    video_path = Path(video)

    # Create the output path...
    output_path = video_path.parent / (video_path.stem + "_labeled" + video_extension)

    body_parts_to_plot = EverythingSet() if(body_parts_to_plot is None) else set(body_parts_to_plot)
    upscale = 1 if(visual_settings.upscale_factor is None) else visual_settings.upscale_factor

    with ContextVideoCapture(video) as in_video:
        out_w, out_h = tuple(
            int(dim * upscale) for dim in [
                in_video.get(cv2.CAP_PROP_FRAME_WIDTH),
                in_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            ]
        )

        with ContextVideoWriter(
            str(output_path),
            visual_settings.output_codec,
            in_video.get(cv2.CAP_PROP_FPS),
            (out_w, out_h)
        ) as writer:
            with TQDMProgressBar(total=poses.get_frame_count()) as p:
                for f_i in range(poses.get_frame_count()):
                    retval, frame = in_video.read()

                    if(not retval):
                        continue

                    if (visual_settings.upscale_factor is not None):
                        frame = cv2.resize(
                            frame,
                            (out_w, out_h),
                            interpolation=cv2.INTER_NEAREST
                        )

                    overlay = frame.copy()

                    colors = iter_colormap(visual_settings.colormap, poses.get_bodypart_count())
                    shapes = shape_iterator(visual_settings.shape_list, num_outputs)

                    part_iter = zip(
                        [name for name in bp_names for _ in range(num_outputs)],
                        poses.get_x_at(f_i, slice(None)),
                        poses.get_y_at(f_i, slice(None)),
                        poses.get_prob_at(f_i, slice(None)),
                        colors,
                        shapes
                    )

                    for (name, x, y, prob, color, shape) in part_iter:
                        if (x != x or y != y):
                            continue

                        if (name not in body_parts_to_plot):
                            continue

                        shape_drawer = CV2DotShapeDrawer(
                            overlay,
                            _to_cv2_color(tuple(color[:3]) + (1,)),
                            -1 if (prob > visual_settings.pcutoff) else visual_settings.line_thickness,
                            cv2.LINE_AA if (visual_settings.antialiasing) else None
                        )[shape]

                        if (prob > visual_settings.pcutoff or visual_settings.draw_hidden_tracks):
                            shape_drawer(int(x * upscale), int(y * upscale), int(visual_settings.dotsize * upscale))

                    writer.write(cv2.addWeighted(
                        overlay, visual_settings.alphavalue, frame, 1 - visual_settings.alphavalue, 0
                    ))
                    p.update()
