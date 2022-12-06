from diplomat.processing.containers import ConfigSpec
import diplomat.processing.type_casters as tc
import matplotlib.colors as mpl_colors
from diplomat.utils.colormaps import to_colormap
from sleap.prefs import prefs
import cv2

def cv2_fourcc_string(val) -> int:
    return int(cv2.VideoWriter_fourcc(*val))


VISUAL_SETTINGS: ConfigSpec = {
    "pcutoff": (0.1, tc.RangedFloat(0, 1), "The probability to cutoff results below."),
    "dotsize": (int(prefs["marker size"]), int, "The size of the dots."),
    "alphavalue": (0.7, tc.RangedFloat(0, 1), "The alpha value of the dots."),
    "colormap": (None, to_colormap, "The colormap to use for tracked points in the video. Can be a matplotlib colormap or a list of matplotlib colors."),
    "shape_list": (None, tc.Optional(tc.List(str)), "A list of shape names, shapes to use for drawing each individual's dots."),
    "line_thickness": (1, int, "Thickness of lines drawn.")
}


FULL_VISUAL_SETTINGS: ConfigSpec = {
    **VISUAL_SETTINGS,
    "skeleton_color": ("black", mpl_colors.to_rgba, "Color of the skeleton."),
    "output_codec": ("mp4v", cv2_fourcc_string, "The codec to use for the labeled video..."),
    "draw_hidden_tracks": (True, bool, "Whether or not to draw locations under the pcutoff value."),
    "antialiasing": (True, bool, "Use antialiasing when drawing points.")
}

