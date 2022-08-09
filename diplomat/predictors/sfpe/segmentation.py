from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from diplomat.predictors.fpe.sparse_storage import ForwardBackwardData
from diplomat.predictors.sfpe.growable_numpy_array import GrowableNumpyArray
from diplomat.processing import ProgressBar


class Segmentor(ABC):
    @abstractmethod
    def __init__(self, size: int):
        pass

    @abstractmethod
    def segment(self, scores: np.ndarray, progress_bar: Optional[ProgressBar]) -> np.ndarray:
        pass


class MidpointSegmentor(Segmentor):
    def __init__(self, size: int):
        self._size = size

    def segment(self, scores: np.ndarray, progress_bar: Optional[ProgressBar]) -> np.ndarray:
        visited = np.zeros(len(scores), bool)
        ordered_scores = np.argsort(scores, kind="stable")[::-1]

        segments = GrowableNumpyArray(3, np.int64)

        # We now iterate through the scores in sorted order, marking off segments...
        for frame_idx in ordered_scores:
            if(visited[frame_idx]):
                continue

            search_start = max(0, int(frame_idx - self._size / 2))
            search_end = min(len(ordered_scores), int(frame_idx + self._size / 2))
            section = slice(search_start, search_end)
            rev_section = slice(search_end - 1, search_start - 1 if(search_start > 0) else None, -1)

            start_idx = search_start + np.argmin(visited[section])
            end_idx = search_end - np.argmin(visited[rev_section])
            visited[section] = True

            if(np.isneginf(scores[frame_idx])):
                # Bad segment, we resolve these later, setting the fix frame to -1 tells the segmented FPE to use fallback to a method that is
                # basically sequential...
                if(len(segments) == 0):
                    raise ValueError("No fix frame found over the entire video!")
                frame_idx = -1

            # Start of the segment, end of the segment, the fix frame index...
            segments.add([start_idx, end_idx, frame_idx])

            if(progress_bar is not None):
                progress_bar.update()

        return segments.finalize()
