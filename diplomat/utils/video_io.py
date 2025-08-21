"""
Contains utilities for reading and writing video files using OpenCV.
"""

import functools
import cv2
from typing import TypeVar, Type, ContextManager, Any, Tuple

import numpy as np

T = TypeVar("T")


@functools.lru_cache(None)
def _create_cv2_manager(clazz: Type[T], **extra_args: Any) -> Type[ContextManager[T]]:
    """
    Create a context manager for a CV2 io writing class. Requires the class implements release for closing a
    file resource.

    :param clazz: The cv2 class to subclass, adding support for python context managers to the class.

    :return: A new class, with support for with statements.
    """

    class cv2_context_manager:
        """
        A wrapper class around {clazz} that supports opening the file with the python
        'with' statement, and automatically closes the file if used in a with statement.
        """
        def __init__(self, *args, throw_on_unopened: bool = True, **kwargs):
            """
            Create a new {clazz}.

            :param args: Any positional arguments to be passed to the {clazz} constructor, see cv2 documentation for details.
            :param throw_on_unopened: If True, throw an IOError if file fails to open when initialized. Otherwise, fail silently like default cv2 behavior.
            """
            self._inst = clazz(*args, **kwargs, **extra_args)
            self._throw_on_unopened = throw_on_unopened

        def __enter__(self):
            """
            Open the video file, and confirm that the file is open.
            """
            if self._throw_on_unopened and not self.isOpened():
                self.release()
                raise IOError("Unable to open video capture...")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Closes the video file when exiting with block.
            """
            self.release()

        def read(self) -> Tuple[bool, np.ndarray]:
            """
            Read a frame from the video file. Only works for the cv2.VideoCapture class.

            :returns: A tuple, the first element indicating if a frame was successfully read from the video, the second
                      being the video frame data as a numpy array of Height by Width by BGR (blue, green, red). Second
                      argument will be an empty array if cv2 failed to read a video frame.
            """
            if not self.isOpened():
                raise IOError("Video capture is not open.")
            return self._inst.read()

        def write(self, frame):
            """
            Write a frame from the video file. Only works for the cv2.VideoWriter class.

            :param frame: A numpy array of Height by Width by BGR (blue, green, red), to write to the end of the
                          video file.
            :returns: A boolean, if the frame was successfully written.
            """
            if not self.isOpened():
                raise IOError("Video writer is not open.")
            return self._inst.write(frame)

        def __getattr__(self, item: str):
            return getattr(self._inst, item)

    cls_name = type(clazz).__name__
    cv2_context_manager.__name__ = f"Context{cls_name}"
    cv2_context_manager.__doc__.format(clazz=cls_name)
    cv2_context_manager.__init__.__doc__.format(clazz=cls_name)

    return cv2_context_manager


ContextVideoWriter = _create_cv2_manager(cv2.VideoWriter, apiPreference=cv2.CAP_FFMPEG)
""" An implementation of cv2.VideoWriter that supports opening the file with the python 'with' statement.  """


ContextVideoCapture = _create_cv2_manager(
    cv2.VideoCapture, apiPreference=cv2.CAP_FFMPEG
)
""" An implementation of cv2.VideoCapture that supports opening the file with the python 'with' statement. """
