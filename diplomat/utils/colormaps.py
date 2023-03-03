"""
Provides utility functions for colormap conversion and iteration.
"""
import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap, ListedColormap, to_rgba
from typing import Union, Tuple, Sequence
import itertools


def to_colormap(cmap: Union[None, str, list, Colormap] = None) -> Colormap:
    """
    Convert any colormap like object to a matplotlib Colormap.

    :param cmap: The colormap-like object, can be a list of colors, the name of a matplotlib colormap, a matplotlib colormap, or None. None
                 indicates that the default matplotlib colormap should be returned.

    :return: A matplotlib Colormap object.
    """
    if(isinstance(cmap, Colormap)):
        return cmap
    if(cmap is None):
        return mpl.colormaps[mpl.rcParams["image.cmap"]]
    if(isinstance(cmap, str)):
        return mpl.colormaps[cmap]
    if(isinstance(cmap, list)):
        return ListedColormap(cmap)
    else:
        raise ValueError("Unable to provided colormap argument to a colormap!")


# Threshold for allowing colormaps to be treated as listed...
_MAX_LISTED_THRESHOLD = 0.05


def iter_colormap(cmap: Colormap, count: int, bytes: bool = False) -> Sequence[Tuple[float, float, float, float]]:
    """
    Iterate a matplotlib colormap, returning a sequence of colors sampled from it.

    :param cmap: The matplotlib Colormap to draw colors from.
    :param count: The number of colors to be sampled from the colormap.
    :param bytes: If True, returned colors are tuples of integers between 0 and 255, if False, they are tuples of floats between 0 and 1

    :return: A list of colors. Each color is a tuple of 4 numbers, representing the red, green, blue, and alpha channels of the color.
    """
    if(isinstance(cmap, ListedColormap)):
        colors = np.asarray(cmap.colors)
        # If the colormap's largest jump in color difference is small, this is likely not a qualitative map, skip treating it like one...
        if(_MAX_LISTED_THRESHOLD < np.max(np.sqrt(np.sum((colors[1:] - colors[:-1]) ** 2, axis=-1)))):
            return [
                to_rgba(c) if(not bytes) else tuple((np.asarray(to_rgba(c)) * 255).astype(int))
                for i, c in zip(range(count), itertools.cycle(cmap.colors))
            ]

    return cmap(np.linspace(0, 1, count), bytes=bytes)