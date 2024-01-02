"""
Includes history-based event for swapping part tracks. Used in stripped down (aka. cli:`diplomat tweak`) UI.
"""

from typing import List, Tuple, Any, Callable, Iterable
from diplomat.predictors.supervised_fpe.labelers import EditableFramePassEngine


def _invert(order: List[int]) -> List[int]:
    lst = [None] * len(order)

    for i in range(len(order)):
        lst[order[i]] = i

    return lst


class IdentitySwapper:
    """
    Swaps tracks to match a new ordering. Can be added to UI history and undone/redone.
    """
    def __init__(self, frame_engine: EditableFramePassEngine):
        self._frame_engine = frame_engine
        self._extra_hook = None
        self._progress_handler = None

    def set_extra_hook(self, hook: Callable[[int, List[int]], None]):
        self._extra_hook = hook

    def set_progress_handler(self, handler: Callable[[str, Iterable], Iterable]):
        self._progress_handler = handler

    def do(self, frame_idx: int, order: List[int]) -> Tuple[int, List[int]]:
        progress_hdlr = self._progress_handler

        if(progress_hdlr is None):
            def progress_hdlr(msg, gen):
                yield from gen

        for f_i in progress_hdlr("Updating Track Identities", range(frame_idx, self._frame_engine.frame_data.num_frames)):
            frame = self._frame_engine.frame_data.frames[f_i]

            for idx, val in enumerate([frame[idx] for idx in order]):
                frame[idx] = val

        swap_keys = [(key, val) for key, val in self._frame_engine.changed_frames.items() if(key[0] >= frame_idx)]

        for (f_i, bp_i), val in swap_keys:
            del self._frame_engine.changed_frames[(f_i, bp_i)]
            self._frame_engine.changed_frames[(f_i, order[bp_i])] = val

        if(self._extra_hook is not None):
            self._extra_hook(frame_idx, order)

        return (frame_idx, _invert(order))

    def undo(self, data: Any) -> Any:
        return self.do(*data)

    def redo(self, data: Any) -> Any:
        return self.do(*data)
