from os import PathLike
from typing import List, Union, Optional

from deeplabcut.pose_estimation_tensorflow.config import load_config
from diplomat.processing import *
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors

Pathy = Union[PathLike, str]

LABELED_VIDEO_SETTINGS = {
    "skeleton_color": ("black", mpl_colors.to_rgba, "Color of the skeleton."),
    "pcutoff": (0.1, type_casters.RangedFloat(0, 1), "The probability to cutoff results below."),
    "dotsize": (4, float, "The size of the dots."),
    "alphavalue": (0.7, type_casters.RangedFloat(0, 1), "The alpha value of the dots."),
    "colormap": (plt.get_cmap(), plt.get_cmap, "The colormap to use for tracked points in the video.")
}

def create_labeled_videos(
    config: Pathy,
    videos: List[Pathy],
    body_parts_to_plot: Optional[List[str]] = None,
    **kwargs
) -> None:
    # TODO: Write...
    config = load_config(config)






