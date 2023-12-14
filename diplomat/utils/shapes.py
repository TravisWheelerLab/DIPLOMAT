"""
Provides utilities and interfaces for drawing shape markers, and iterating lists of shapes and converting lists of shapes names to shapes.
"""
from typing import Callable, Iterable, Tuple, Iterator, Optional
import numpy as np
import cv2
from abc import ABC, abstractmethod


class DotShapeDrawer(ABC):
    """
    Abstract class defining an interface for drawing various markers, or dots, based on shape.
    """

    # Unit polygons for certain types of items that for which built-in drawing functions don't exist in most
    # programming languages...
    _TRIANGLE_POLY = np.array([[0, -1], [-0.8660254037844386, 0.5], [0.8660254037844386, 0.5]])
    _STAR_POLY = np.array([
        [0, -1.0],
        [-0.22451398828979272, -0.3090169943749475],
        [-0.9510565162951535, -0.3090169943749475],
        [-0.36327126400268056, 0.11803398874989483],
        [-0.5877852522924732, 0.8090169943749473],
        [-7.01660179590785e-17, 0.38196601125010526],
        [0.5877852522924729, 0.8090169943749476],
        [0.36327126400268056, 0.11803398874989496],
        [0.9510565162951536, -0.3090169943749472],
        [0.22451398828979283, -0.30901699437494745]
    ])
    _INSIDE_SQUARE_RADIUS_RATIO = 0.9

    SHAPE_TYPES = tuple()

    def __getitem__(self, shape: str) -> Callable[[float, float, float], None]:
        """
        Get a drawer for the provided shape type.

        :param shape: The shape to get a drawing function for be default, all drawers must support "circle", "square",
                      "triangle", and "star".

        :return: A function or callable which accepts 3 floats (x coordinate, y coordinate, shape radius), that draws a
                 shape marker to the specified location when called.
        """
        return getattr(self, "_draw_" + shape)

    def __contains__(self, shape: str) -> bool:
        """
        Check if this shape drawer supports this shape.

        :param shape: A string representing a shape type ("square", "circle", etc.)

        :return: True if this shape drawer supports that shape, otherwise False.
        """
        return hasattr(self, "_draw_" + shape)

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


DotShapeDrawer.SHAPE_TYPES = tuple(
    ["_".join(val.split("_")[2:]) for val in dir(DotShapeDrawer) if(val.startswith("_draw_"))]
)


def shape_str(shape: str) -> str:
    shape = str(shape)
    if(shape not in DotShapeDrawer.SHAPE_TYPES):
        raise ValueError(
            f"Shape name '{shape}' not valid, supported shape names are: {list(DotShapeDrawer.SHAPE_TYPES)}"
        )
    return shape


class shape_iterator:
    """
    Allows one to iterate over a list of shape strings indefinitely, and in groups. Used to iterate over shapes on a
    per individual basis.
    """
    def __init__(self, sequence: Optional[Iterable[str]] = None, rep_count: int = None):
        """
        Get a new shape iterator.

        :param sequence: The sequence of shapes to iterate over. If this is None, uses the default shape list.
        :param rep_count: The number of values to iterate through before restarting at the beginning of the sequence.
                          If larger than the sequence length, the iteration will wrap around the sequence, and
                          continue until this value is reached and then reset.
        """
        if(isinstance(sequence, type(self))):
            self._seq = sequence._seq
            self._rep = sequence._rep
            return

        self._seq = sequence if(sequence is not None) else ("circle", "triangle", "square", "star")
        self._rep = 1 if(rep_count is None) else rep_count

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

    def __tojson__(self):
        return {
            "sequence": self._seq,
            "rep_count": self._rep
        }

    @classmethod
    def __fromjson__(cls, data):
        return cls(data["sequence"], data["rep_count"])


class CV2DotShapeDrawer(DotShapeDrawer):
    """
    A shape dot or marker implementation that utilizes opencv2 for drawing. It can draw to images stored as 2D
    numpy arrays.
    """
    def __init__(
        self,
        img: np.ndarray,
        color: Tuple[int, int, int, int],
        line_thickness: int = 1,
        line_type: int = cv2.LINE_8
    ):
        """
        Create a cv2 marker, or shape drawer.

        :param img: The image to draw results onto, a 2D numpy array, indexed by y coordinate first.
        :param color: The color of the dots to be drawn. Should be a tuple of 4 integers between 0 and 255 being the
                      rgba color.
        :param line_thickness: The thickness of the border of the dots.
        :param line_type: The type of line to ask cv2 to draw.
        """
        self._img = img
        self._color = color
        self._line_thickness = line_thickness
        self._line_type = line_type

    def _draw_circle(self, x: float, y: float, r: float):
        cv2.circle(self._img, (int(x), int(y)), int(r), self._color, self._line_thickness, self._line_type)

    def _draw_square(self, x: float, y: float, r: float):
        r = r * self._INSIDE_SQUARE_RADIUS_RATIO
        cv2.rectangle(
            self._img,
            (int(x - r), int(y - r)),
            (int(x + r), int(y + r)),
            self._color,
            self._line_thickness,
            self._line_type
        )

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

