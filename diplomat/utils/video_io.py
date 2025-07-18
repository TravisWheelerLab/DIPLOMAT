import functools
import cv2
from typing import TypeVar, Type, ContextManager, Any

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
        def __init__(self, *args, throw_on_unopened=True, **kwargs):
            self._inst = clazz(*args, **kwargs, **extra_args)
            self._throw_on_unopened = throw_on_unopened

        def __enter__(self):
            if self._throw_on_unopened and not self.isOpened():
                self.release()
                raise IOError("Unable to open video capture...")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()

        def read(self):
            if not self.isOpened():
                raise IOError("Video capture is not open.")
            return self._inst.read()

        def write(self, frame):
            if not self.isOpened():
                raise IOError("Video writer is not open.")
            return self._inst.write(frame)

        def __getattr__(self, item: str):
            return getattr(self._inst, item)

    return cv2_context_manager


""" An implementation of cv2.VideoWriter with support for context managers. """
ContextVideoWriter = _create_cv2_manager(cv2.VideoWriter, apiPreference=cv2.CAP_FFMPEG)

""" An implementation of cv2.VideoWriter with support for context managers. """
ContextVideoCapture = _create_cv2_manager(
    cv2.VideoCapture, apiPreference=cv2.CAP_FFMPEG
)
