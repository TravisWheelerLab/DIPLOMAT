import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap, ListedColormap, to_rgba
from typing import Union, Tuple, Sequence
import itertools


def to_colormap(cmap: Union[None, str, list, Colormap] = None) -> Colormap:
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


def iter_colormap(cmap: Colormap, count: int, bytes: bool = False) -> Sequence[Tuple[int, int, int, int]]:
    if(isinstance(cmap, ListedColormap)):
        return [
            to_rgba(c) if(not bytes) else tuple((np.asarray(to_rgba(c)) * 255).astype(int))
            for i, c in zip(range(count), itertools.cycle(cmap.colors))
        ]
    else:
        return cmap(np.linspace(0, 1, count), bytes=bytes)