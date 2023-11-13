from abc import ABC, abstractmethod
from typing import Dict, TypeVar, Tuple, Type, Any, Sequence, Optional, Set, Callable
from .sparse_storage import ForwardBackwardData, ForwardBackwardFrame, AttributeDict
from diplomat.processing import *

oint = Optional[int]


class PassOrderError(ValueError):
    pass


class RangeSlicer:
    """
    A RangeSlicer! Allows for one to convert a slice into a range.
    """
    def __init__(self, array: Sequence):
        self._wrap_len = len(array)

    def __getitem__(self, item):
        if(isinstance(item, slice)):
            return range(*item.indices(self._wrap_len))


class FramePass(ABC):
    UTILIZE_GLOBAL_POOL = False
    GLOBAL_POOL = None

    def __init__(
        self,
        width: int,
        height: int,
        multi_threading_allowed: bool,
        config: Dict[str, Any]
    ):
        # Set defaults to forward iteration...
        self._step = 1
        self._start = None
        self._stop = None
        self._prior_off = -1

        self.__width = width
        self.__height = height
        self.__multi_threading_allowed = multi_threading_allowed

        self._config = Config(config, self.get_config_options())
        self._frame_data = None

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height

    @property
    def multi_threading_allowed(self) -> bool:
        return self.__multi_threading_allowed

    def _get_step_controls(self) -> Tuple[oint, oint, oint, oint]:
        return self._start, self._stop, self._step, self._prior_off

    def _set_step_controls(self, start: oint, stop: oint, step: oint, prior_offset: oint):
        self._start, self._stop, self._step, self._prior_off = start, stop, step, prior_offset

    def run_pass(
        self,
        fb_data: ForwardBackwardData,
        prog_bar: Optional[ProgressBar] = None,
        in_place: bool = True,
        reset_bar: bool = True
    ) -> ForwardBackwardData:
        """
        Optional Override: Runs a pass of this FramePass.
        """
        self._frame_data = fb_data

        arr = fb_data.frames
        dest = fb_data if(in_place) else ForwardBackwardData(fb_data.num_frames, fb_data.num_bodyparts)
        dest_arr = dest.frames

        if((prog_bar is not None) and reset_bar):
            prog_bar.reset(len(RangeSlicer(arr)[self._start:self._stop:self._step]))

        for frame_idx in RangeSlicer(arr)[self._start:self._stop:self._step]:
            for bp_idx in range(fb_data.num_bodyparts):
                prior = (dest_arr[frame_idx + self._prior_off][bp_idx]
                         if(0 <= (frame_idx + self._prior_off) < len(dest_arr)) else None)
                current = arr[frame_idx][bp_idx] if(in_place) else arr[frame_idx][bp_idx].copy()
                result = self.run_step(prior, current, frame_idx, bp_idx, fb_data.metadata)
                if(result is not None):
                    dest_arr[frame_idx][bp_idx] = result
                else:
                    dest_arr[frame_idx][bp_idx] = arr[frame_idx][bp_idx]

            if(prog_bar is not None):
                prog_bar.update()

        dest.metadata = fb_data.metadata
        self._frame_data = None

        return dest

    def run_step(
        self,
        prior: Optional[ForwardBackwardFrame],
        current: ForwardBackwardFrame,
        frame_index: int,
        bodypart_index: int,
        metadata: AttributeDict
    ) -> Optional[ForwardBackwardFrame]:
        raise NotImplementedError()

    @property
    def fb_data(self) -> Optional[ForwardBackwardData]:
        return self._frame_data

    @property
    def config(self) -> Config:
        return self._config

    T = TypeVar("T")

    @classmethod
    @abstractmethod
    def get_config_options(cls) -> Optional[Dict[str, Tuple[T, Callable[[Any], T], str]]]:
        raise NotImplementedError()

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__.split(".")[-1]

    @classmethod
    def get_subclasses(cls) -> Set[Type["FramePass"]]:
        from diplomat.utils.pluginloader import load_plugin_classes
        from . import frame_passes
        return load_plugin_classes(frame_passes, cls)


class ConfigError(ValueError):
    pass

