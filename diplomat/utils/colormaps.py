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
        return self._count_hint is not None

    def get_colors(
        self, alpha: Optional[float] = None, bytes: bool = False
    ) -> np.ndarray:
        if not self.is_listed:
            raise ValueError(
                "This colormap is not a listed colormap, so it does not have a fixed list of colors."
            )

        offsets = (np.arange(self._count_hint) + 0.5) / self._count_hint
        return self(offsets, alpha, bytes)

    @classmethod
    def to_rgba_optional(cls, color):
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
        if isinstance(colormap, mpl_colors.ListedColormap):
            return cls.from_list(colormap.name, list(colormap.colors), colormap.N)
        if isinstance(colormap, mpl_colors.LinearSegmentedColormap):
            return cls.from_linear_segments(
                colormap.name, colormap._segmentdata, colormap._gamma
            )

        raise ValueError(f"Unsupported matplotlib colormap type: {type(colormap)}")

    def to_matplotlib_colormap(self):
        if self.is_listed:
            return mpl_colors.ListedColormap(self.get_colors(1.0, False), self.name)
        else:
            def _to_mpl_segments(seg):
                lutmap = np.stack([
                    seg[:, 0], seg[:, 1], seg[:, 1]
                ], axis=-1)
                to_stack = []
                if lutmap[0, 0] != 0.0:
                    to_stack.append([[0.0, *lutmap[0, 1:]]])
                to_stack.append(lutmap)
                if lutmap[-1, 0] != 1.0:
                    to_stack.append([[1.0, *lutmap[-1, 1:]]])
                return np.concatenate(to_stack, axis=0)

            return mpl_colors.LinearSegmentedColormap(self.name, {
                "red": _to_mpl_segments(self._r),
                "green": _to_mpl_segments(self._g),
                "blue": _to_mpl_segments(self._b)
            })

    @staticmethod
    def _normalize_mapper(v):
        v = v[np.argsort(v[:, 0])]
        v[:, 0] = np.clip(v[:, 0], 0.0, 1.0)
        return v

    @property
    def name(self) -> str:
        return self._name

    def __call__(
        self, data: np.ndarray, alpha: Optional[float] = None, bytes: bool = False
    ):
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
    Convert any colormap like object to a matplotlib Colormap.

    :param cmap: The colormap-like object, can be a list of colors, the name of a matplotlib colormap, a matplotlib colormap, or None. None
                 indicates that the default matplotlib colormap should be returned.

    :return: A matplotlib Colormap object.
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
    Iterate a matplotlib colormap, returning a sequence of colors sampled from it.

    :param cmap: The matplotlib Colormap to draw colors from.
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
