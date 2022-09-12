from typing import Callable, Iterable, Tuple, Iterator
import numpy as np
import cv2
from abc import ABC, abstractmethod


class DotShapeDrawer(ABC):
    # Unit polygons for certain types of items that for which built-in drawing functions don't exist in most programming languages...
    _TRIANGLE_POLY = np.array([[0, -1], [-0.8660254037844386, 0.5], [0.8660254037844386, 0.5]])
    _STAR_POLY = np.array([
        [0, -1.0],
        [-0.5877852522924732, 0.8090169943749473],
        [0.9510565162951536, -0.3090169943749472],
        [-0.9510565162951535, -0.3090169943749475],
        [0.5877852522924729, 0.8090169943749476],
    ])
    _INSIDE_SQUARE_RADIUS_RATIO = 0.9

    SHAPE_TYPES = tuple()

    def __getitem__(self, item: str) -> Callable:
        return getattr(self, "_draw_" + item)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, "_draw_" + item)

    def __len__(self):
        len(self.SHAPE_TYPES)

    def __iter__(self):
        return self.SHAPE_TYPES

    @abstractmethod
    def _draw_circle(self, x: float, y: float, r: float):
        pass

    @abstractmethod
    def _draw_square(self, x: float, y: float, r: float):
        pass

    @abstractmethod
    def _draw_triangle(self, x: float, y: float, r: float):
        pass

    @abstractmethod
    def _draw_star(self, x: float, y: float, r: float):
        pass

DotShapeDrawer.SHAPE_TYPES = tuple(["_".join(val.split("_")[2:]) for val in dir(DotShapeDrawer) if(val.startswith("_draw_"))])

def shape_str(shape: str) -> str:
    shape = str(shape)
    if(shape not in DotShapeDrawer.SHAPE_TYPES):
        raise ValueError(f"Shape name '{shape}' not valid, supported shape names are: {list(DotShapeDrawer.SHAPE_TYPES)}")
    return shape


class shape_iterator:
    def __new__(cls, sequence: Iterable[str], rep_count: int = None):
        if(sequence is None):
            return cls.__new__(cls, ("circle", "square", "triangle", "star"), 1 if(rep_count is None) else rep_count)
        if(isinstance(sequence, cls)):
            return cls.__new__(cls, sequence._seq, sequence._rep if(rep_count is None) else rep_count)

        inst = super().__new__(cls)
        inst._seq = sequence
        inst._rep = 1 if(rep_count is None) else rep_count
        return inst

    def __iter__(self) -> Iterator[str]:
        self._count = 0
        self._iter = iter(self._seq)
        return self

    def __next__(self) -> str:
        if(self._count >= self._rep):
            self._count = 0
            self._iter = iter(self._seq)

        try:
            val = next(self._iter)
        except StopIteration:
            self._iter = iter(self._seq)
            val = next(self._iter)

        self._count += 1
        return shape_str(val)


class CV2DotShapeDrawer(DotShapeDrawer):
    def __init__(
        self,
        img: np.ndarray,
        color: Tuple[int, int, int, int],
        line_thickness: int = 1,
        line_type: int = cv2.LINE_8
    ):
        self._img = img
        self._color = color
        self._line_thickness = line_thickness
        self._line_type = line_type

    def _draw_circle(self, x: float, y: float, r: float):
        cv2.circle(self._img, (int(x), int(y)), int(r), self._color, self._line_thickness, self._line_type)

    def _draw_square(self, x: float, y: float, r: float):
        r = r * self._INSIDE_SQUARE_RADIUS_RATIO
        cv2.rectangle(self._img, (int(x - r), int(y - r)), (int(x + r), int(y + r)), self._color, self._line_thickness, self._line_type)

    def _draw_triangle(self, x: float, y: float, r: float):
        points = (self._TRIANGLE_POLY * r + np.array([x, y])).astype(int)

        if(self._line_thickness <= 0):
            cv2.fillPoly(self._img, [points], self._color, self._line_type)
        else:
            cv2.polylines(self._img, [points], True, self._color, self._line_thickness, self._line_type)


    def _draw_star(self, x: float, y: float, r: float):
        points = (self._STAR_POLY * r + np.array([x, y])).astype(int)

        if (self._line_thickness <= 0):
            cv2.fillPoly(self._img, [points], self._color, self._line_type)
        else:
            cv2.polylines(self._img, [points], True, self._color, self._line_thickness, self._line_type)

