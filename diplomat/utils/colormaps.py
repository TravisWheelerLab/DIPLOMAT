"""
Provides utility functions for colormap conversion and iteration.
"""

import base64

import matplotlib as mpl
import numpy as np
import matplotlib.colors as mpl_colors
from typing import Union, Tuple, Sequence, Optional, List, Dict

import diplomat.processing.type_casters as tc
import itertools


class DiplomatColormap:
    def __init__(
        self,
        name: str,
        r_values: np.ndarray,
        g_values: np.ndarray,
        b_values: np.ndarray,
        under: Optional[Sequence[float]] = None,
        over: Optional[Sequence[float]] = None,
        bad: Optional[Sequence[float]] = None,
        count_hint: Optional[int] = None,
    ):
        """
        Create a new DIPLOMAT Colormap, which maps values from 0 to 1 to colors.

        :param name: The name of the colormap.
        :param r_values: A Nx2 numpy array, mapping offsets (0-1) to red channel intensity values (0-1).
        :param g_values: A Nx2 numpy array, mapping offsets (0-1) to green channel intensity values (0-1).
        :param b_values: A Nx2 numpy array, mapping offsets (0-1) to blue channel intensity values (0-1).
        :param under: A sequence of 3 floats, rgb color used when a passed offset is under 0. If not set, uses color
                      at offset 0.
        :param over: A sequence of 3 floats, rgb color used when a passed offset is over 1. If not set, uses color
                     at offset 1.
        :param bad: A sequence of 3 floats, rgb color used when a passed offset is nan. If not set, uses color
                    at offset 1.
        :param count_hint: Optional integer, if set provides a count hint for the number of colors in the colormap.
                           This should only be passed for colormaps that are meant to be listed colormaps.
        """
        self._r = self._normalize_mapper(r_values)
        self._g = self._normalize_mapper(g_values)
        self._b = self._normalize_mapper(b_values)
        self._under = under if under is None else np.asarray(under)
        self._over = over if over is None else np.asarray(over)
        self._bad = bad if bad is None else np.asarray(bad)
        self._name = name
        self._count_hint = count_hint

    @property
    def is_listed(self) -> bool:
        """
        Indicates if the colormap is a listed colormap, or just meant to represent a list of colors, and
        not interpolate between them in any way.
        """
        return self._count_hint is not None

    def get_colors(
        self, alpha: Optional[float] = None, bytes: bool = False
    ) -> np.ndarray:
        """
        Get the list of colors represented by this colormap. This is only valid for listed colormaps.

        :param alpha: Optional float, value to use for alpha channel for each color.
        :param bytes: If true, return the colors as unsigned bytes between 0-255 instead of floats between 0-1.

        :returns: Numpy array of Nx4, list of rgba colors. Type of elements depends on bytes parameter.
        """
        if not self.is_listed:
            raise ValueError(
                "This colormap is not a listed colormap, so it does not have a fixed list of colors."
            )

        offsets = (np.arange(self._count_hint) + 0.5) / self._count_hint
        return self(offsets, alpha, bytes)

    @classmethod
    def to_rgba_optional(cls, color):
        """
        Convert a color to a tuple of 4 floats, in rgba format, unless it's not None, in which case it returns None.
        """
        return color if color is None else mpl_colors.to_rgba(color)

    @classmethod
    def from_list(
        cls,
        name: str,
        colors: list,
        n: Optional[int] = None,
        under=None,
        over=None,
        bad=None,
    ) -> "DiplomatColormap":
        """
        Create a diplomat colormap from a list of colors.

        :param name: The name of the colormap.
        :param colors: A list 'matplotlib' colors. Can be strings, or tuples of integers or floats.
        :param n: Number of colors in the colormap. If None, use the length of the list of colors. colors are truncated
                  or repeated to match this value.
        :param under: The underflow color.
        :param over: The overflow color.
        :param bad: The bad (for NaN inputs) color.

        :return: A diplomat colormap.
        """
        colors = list(
            itertools.islice(itertools.cycle(colors), n if (n is None) else len(colors))
        )
        colors = mpl_colors.to_rgba_array(colors)[:, :3]
        offsets = np.linspace(0, 1, len(colors) + 1)
        offsets = np.stack([np.nextafter(offsets, -np.inf), offsets], -1).reshape(-1)[
            1:-1
        ]
        offsets[-1] = 1.0

        colors = [
            np.stack([offsets, np.repeat(channel, 2)], -1) for channel in colors.T
        ]

        return cls(
            name,
            colors[0],
            colors[1],
            colors[2],
            cls.to_rgba_optional(under),
            cls.to_rgba_optional(over),
            cls.to_rgba_optional(bad),
            n,
        )

    @classmethod
    def from_linear_segments(
        cls,
        name: str,
        segmentdata: Dict[str, Sequence[Tuple[float, float, float]]],
        gamma: float = 1.0,
        under=None,
        over=None,
        bad=None,
    ) -> "DiplomatColormap":
        """
        Create a diplomat colormap from a colormap segment data.

        :param name: The name of the colormap.
        :param segmentdata: A dictionary of channel ['r', 'g', 'b'] to a Nx3 array. See matplotlib's segment data format.
        :param gamma: Gamma correction to apply to offsets before mapping to colors, default to 1.0, or no gamma
                      correction.
        :param under: The underflow color.
        :param over: The overflow color.
        :param bad: The bad (for NaN inputs) color.

        :return: A diplomat colormap.
        """
        def _from_segments(d):
            if callable(d):
                xs = np.linspace(0, 1, 255)
                return np.stack([xs, np.clip(d(xs**gamma), 0.0, 1.0)], -1)
            else:
                d = np.asarray(d)
                if d.shape[0] == 1:
                    d[:, 1] = d[:, 2]
                    d = np.repeat(d, 2, 0)
                xs = d[:, 0] ** gamma
                offsets = np.stack([np.nextafter(xs, -np.inf), xs], -1).reshape(-1)
                return np.stack([offsets, d[:, 1:].reshape(-1)], -1)[1:-1]

        red = segmentdata["red"]
        green = segmentdata["green"]
        blue = segmentdata["blue"]

        return cls(
            name,
            _from_segments(red),
            _from_segments(green),
            _from_segments(blue),
            under,
            over,
            bad,
        )

    # noinspection PyUnresolvedReferences
    @classmethod
    def from_matplotlib_colormap(
        cls, colormap: mpl_colors.Colormap
    ) -> "DiplomatColormap":
        """
        Create a DIPLOMAT colormap from a matplotlib colormap.

        :param colormap: A matplotlib colormap.

        :return: A diplomat colormap.
        """
        if isinstance(colormap, mpl_colors.ListedColormap):
            return cls.from_list(colormap.name, list(colormap.colors), colormap.N)
        if isinstance(colormap, mpl_colors.LinearSegmentedColormap):
            return cls.from_linear_segments(
                colormap.name, colormap._segmentdata, colormap._gamma
            )

        raise ValueError(f"Unsupported matplotlib colormap type: {type(colormap)}")

    def to_matplotlib_colormap(self):
        """
        Convert the DIPLOMAT colormap to a matplotlib colormap.

        :return: A matplotlib colormap that matches this DIPLOMAT colormap.
        """
        if self.is_listed:
            return mpl_colors.ListedColormap(self.get_colors(1.0, False), self.name)
        else:

            def _to_mpl_segments(seg):
                lutmap = np.stack([seg[:, 0], seg[:, 1], seg[:, 1]], axis=-1)
                to_stack = []
                if lutmap[0, 0] != 0.0:
                    to_stack.append([[0.0, *lutmap[0, 1:]]])
                to_stack.append(lutmap)
                if lutmap[-1, 0] != 1.0:
                    to_stack.append([[1.0, *lutmap[-1, 1:]]])
                return np.concatenate(to_stack, axis=0)

            return mpl_colors.LinearSegmentedColormap(
                self.name,
                {
                    "red": _to_mpl_segments(self._r),
                    "green": _to_mpl_segments(self._g),
                    "blue": _to_mpl_segments(self._b),
                },
            )

    @staticmethod
    def _normalize_mapper(v):
        v = v[np.argsort(v[:, 0])]
        v[:, 0] = np.clip(v[:, 0], 0.0, 1.0)
        return v

    @property
    def name(self) -> str:
        """
        The name of the colormap.
        """
        return self._name

    def __call__(
        self, data: np.ndarray, alpha: Optional[float] = None, bytes: bool = False
    ):
        """
        Apply this colormap to some data.

        :param data: The data, an any dimensional array (shape ...) of floats between 0 and 1.
        :param alpha: Optional float, the value for the alpha channel in the colors. Defaults to 1.0.
        :param bytes: If true, return color data as unsigned bytes between 0 and 255, otherwise return as floats
                      between 0 and 1.

        :return: An ...x4 array, the last added dimension being the color channels, being red, green, blue, and alpha
                 in order. Data type of channels depends on the bytes argument.
        """
        if alpha is None:
            alpha = 1.0

        alpha = max(0.0, min(1.0, alpha))
        mult = 255 if bytes else 1.0
        colors = np.zeros(data.shape + (4,), dtype=np.uint8 if bytes else np.float32)
        colors[..., -1] = alpha * mult

        for i, mapper in enumerate([self._r, self._g, self._b]):
            xs, ys = mapper.T
            under = None if self._under is None else self._under[i]
            over = None if self._over is None else self._over[i]
            bad = 0 if self._bad is None else self._bad[i]
            colors[..., i] = (
                np.clip(
                    np.nan_to_num(np.interp(data, xs, ys, under, over), nan=bad), 0, 1
                )
                * mult
            )

        return colors

    def __tojson__(self):
        to_string = lambda arr: (
            base64.b64encode(arr.astype("<f8").tobytes()).decode()
            if arr is not None
            else None
        )

        return {
            "name": self._name,
            "r_values": to_string(self._r),
            "g_values": to_string(self._g),
            "b_values": to_string(self._b),
            "under": to_string(self._under),
            "over": to_string(self._over),
            "bad": to_string(self._bad),
            "count_hint": self._count_hint,
        }

    @classmethod
    def __fromjson__(cls, data: dict):
        from_string = lambda s: (
            np.frombuffer(base64.b64decode(s.encode()), "<f8")
            if s is not None
            else None
        )

        return cls(
            data["name"],
            from_string(data["r_values"]).reshape((-1, 2)),
            from_string(data["g_values"]).reshape((-1, 2)),
            from_string(data["b_values"]).reshape((-1, 2)),
            from_string(data["under"]),
            from_string(data["over"]),
            from_string(data["bad"]),
            data["count_hint"],
        )

    def __str__(self):
        return f"{type(self).__name__}(name={self._name})"


@tc.attach_hint(
    Union[
        None,
        str,
        List[Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]],
    ]
)
def to_colormap(
    cmap: Union[None, str, list, mpl_colors.Colormap, DiplomatColormap] = None,
) -> DiplomatColormap:
    """
    Convert any colormap like object to a :py:class:`~diplomat.utils.colormaps.DiplomatColormap`.

    :param cmap: The colormap-like object, can be a list of colors, the name of a matplotlib colormap,
                 a matplotlib colormap, a :py:class:`~diplomat.utils.colormaps.DiplomatColormap`, or None. None
                 indicates that the default matplotlib colormap should be converted to a
                 :py:class:`~diplomat.utils.colormaps.DiplomatColormap` and returned.

    :return: A :py:class:`~diplomat.utils.colormaps.DiplomatColormap` object.
    """
    if isinstance(cmap, DiplomatColormap):
        return cmap
    if isinstance(cmap, mpl_colors.Colormap):
        return DiplomatColormap.from_matplotlib_colormap(cmap)
    if cmap is None:
        return DiplomatColormap.from_matplotlib_colormap(
            mpl.colormaps[mpl.rcParams["image.cmap"]]
        )
    if isinstance(cmap, str):
        return DiplomatColormap.from_matplotlib_colormap(mpl.colormaps[cmap])
    if isinstance(cmap, list):
        return DiplomatColormap.from_list("_from_list", cmap)
    else:
        raise ValueError("Unable to provided colormap argument to a colormap!")


# Threshold for allowing colormaps to be treated as listed...
_MAX_LISTED_THRESHOLD = 0.05


def iter_colormap(
    cmap: DiplomatColormap, count: int, bytes: bool = False
) -> Sequence[Tuple[float, float, float, float]]:
    """
    Iterate a :py:class:`~diplomat.utils.colormaps.DiplomatColormap`, returning a sequence of colors sampled from it.

    :param cmap: The :py:class:`~diplomat.utils.colormaps.DiplomatColormap` to draw colors from.
    :param count: The number of colors to be sampled from the colormap.
    :param bytes: If True, returned colors are tuples of integers between 0 and 255, if False, they are tuples of floats between 0 and 1

    :return: A list of colors. Each color is a tuple of 4 numbers, representing the red, green, blue, and alpha channels of the color.
    """
    # If listed colormap with actual unique colors, cycle colors instead of just uniformly sampling colors
    # across the colormap...
    if cmap.is_listed:
        colors = cmap.get_colors()
        # If the colormap's largest jump in color difference is small, this is likely not a qualitative map, skip treating it like one...
        if _MAX_LISTED_THRESHOLD < np.max(
            np.sqrt(np.sum((colors[1:] - colors[:-1]) ** 2, axis=-1))
        ):
            reps = int(np.ceil(count / len(colors)))
            colors = np.tile(colors, [reps, 1])[:count]
            return (colors * 255).astype(np.uint8) if bytes else colors

    return cmap(np.linspace(0, 1, count), bytes=bytes)
