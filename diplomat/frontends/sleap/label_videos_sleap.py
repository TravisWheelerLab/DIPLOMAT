from pathlib import Path
from typing import Tuple
import cv2
from .sleap_importer import sleap

import diplomat.processing.type_casters as tc
from diplomat.utils.cli_tools import extra_cli_args
from diplomat.processing import Config, TQDMProgressBar
from diplomat.utils.colormaps import iter_colormap
from diplomat.utils.video_io import ContextVideoWriter
from diplomat.utils.shapes import shape_iterator, CV2DotShapeDrawer

from .visual_settings import FULL_VISUAL_SETTINGS
from .run_utils import (
    _paths_to_str,
    _to_diplomat_poses,
    _load_config
)


def _to_cv2_color(color: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    r, g, b, a = [min(255, max(0, int(val * 256))) for val in color]
    return (b, g, r, a)


class EverythingSet:
    def __contains__(self, item):
        return True


@extra_cli_args(FULL_VISUAL_SETTINGS, auto_cast=False)
@tc.typecaster_function
def label_videos(
    config: tc.PathLike,
    videos: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    body_parts_to_plot: tc.Optional[tc.List[str]] = None,
    video_extension: str = "mp4",
    **kwargs
):
    """
    Label videos tracked using the SLEAP frontend.

    :param config: The path (or list of paths) to the sleap model(s) used for inference, each as either as a folder or zip file.
    :param videos: Paths to the sleap label files, or .slp files, to make minor modifications to, NOT the video files.
    :param body_parts_to_plot: An optional list of body part names to plot in the labeled video. Defaults to None, meaning plot all body parts.
    :param video_extension: The file extension to use on the created labeled video, excluding the dot. Defaults to 'mp4'.
    :param kwargs: The following additional arguments are supported:

                   {extra_cli_args}
    """
    _load_config(_paths_to_str(config))

    videos = _paths_to_str(videos)
    videos = [videos] if(isinstance(videos, str)) else videos

    visual_settings = Config(kwargs, FULL_VISUAL_SETTINGS)

    for video in videos:
        _label_video_single(video, visual_settings, body_parts_to_plot, video_extension)


def _label_video_single(
    label_path: str,
    visual_settings: Config,
    body_parts_to_plot: tc.Optional[tc.List[str]],
    video_extension: str
):
    print(f"Labeling Video Associated with Labels '{label_path}'...")

    # Grab video and pose info from labels...
    labels = sleap.load_file(label_path)
    label_path = Path(label_path)
    num_outputs, poses, video, skeleton = _to_diplomat_poses(labels)
    video_extension = video_extension if(video_extension.startswith(".")) else f".{video_extension}"

    # Create the output path...
    output_path = label_path.parent / (label_path.stem + "_labeled" + video_extension)

    body_parts_to_plot = EverythingSet() if(body_parts_to_plot is None) else set(body_parts_to_plot)
    bp_names = [name for name in skeleton.node_names for _ in range(num_outputs)]

    upscale = 1 if(visual_settings.upscale_factor is None) else visual_settings.upscale_factor
    out_w, out_h = tuple(int(dim * upscale) for dim in video.shape[1:3][::-1])

    print(f"Writing output to: '{output_path}'")

    with ContextVideoWriter(
        str(output_path),
        visual_settings.output_codec,
        getattr(video, "fps", 30),
        (out_w, out_h)
    ) as writer:
        with TQDMProgressBar(total=poses.get_frame_count()) as p:
            for f_i in range(poses.get_frame_count()):
                frame = video.get_frame(f_i)[..., ::-1]

                if(visual_settings.upscale_factor is not None):
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
                    if(x != x or y != y):
                        continue

                    if(name not in body_parts_to_plot):
                        continue

                    shape_drawer = CV2DotShapeDrawer(
                        overlay,
                        _to_cv2_color(tuple(color[:3]) + (1,)),
                        -1 if (prob > visual_settings.pcutoff) else visual_settings.line_thickness,
                        cv2.LINE_AA if (visual_settings.antialiasing) else None
                    )[shape]

                    if(prob > visual_settings.pcutoff or visual_settings.draw_hidden_tracks):
                        shape_drawer(int(x * upscale), int(y * upscale), int(visual_settings.dotsize * upscale))

                writer.write(cv2.addWeighted(
                    overlay, visual_settings.alphavalue, frame, 1 - visual_settings.alphavalue, 0
                ))
                p.update()





