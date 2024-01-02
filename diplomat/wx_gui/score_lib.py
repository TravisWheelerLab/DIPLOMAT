"""
Provides an abstract class for implementing scores, or metrics that can be displayed/monitored in DIPLOMAT's UI.
These are the scores displayed at the bottom of the UI..
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Iterable
from diplomat.wx_gui.labeler_lib import SettingCollection
from diplomat.wx_gui.probability_displayer import ProbabilityDisplayer
import wx
from diplomat.processing import *
import numpy as np


class ScoreEngine(ABC):
    """
    Represents a "ScoreEngine" which assigns scores to every frame that can be
    displayed in the UI.
    """

    @abstractmethod
    def compute_scores(self, poses: Pose, prog_bar: ProgressBar, sub_section: Optional[slice] = None) -> np.ndarray:
        """
        Compute the scores for the given set of poses.

        :param poses: The predicted locations/probabilities for given body parts.
        :param prog_bar: A progress bar for keeping track of current progress
                         on computing the scores.
        :param sub_section: Optional slice, passed when only some scores need to be recomputed.

        :returns: A numpy array of floats between 1 and 0, being the length of
                  the video...
        """
        pass

    @abstractmethod
    def compute_bad_indexes(self, scores: np.ndarray) -> np.ndarray:
        """
        Given all the scores, return a numpy array of integers specifying the indexes of frames which score "bad" or
        poorly for this metric. Typically done via a configurable threshold.
        """
        pass

    @abstractmethod
    def get_settings(self) -> SettingCollection:
        """
        Return a single setting, being the config for this single widget...
        """

    def get_name(self) -> str:
        return "".join([
            f" {c}" if(65 <= ord(c) <= 90) else c for c in type(self).__name__
        ]).strip()


class ScoreEngineDisplayer(wx.Control):
    def __init__(
        self,
        score_engine: ScoreEngine,
        poses: Pose,
        progress_bar: ProgressBar,
        parent=None,
        *args,
        **kwargs
    ):
        super().__init__(parent, *args, **kwargs)

        self._score_engine = score_engine
        self._main_layout = wx.BoxSizer(wx.HORIZONTAL)

        settings = self._score_engine.get_settings()

        for w in settings.widgets.values():
            w.set_hook(self._on_setting_change)
            self._setting_section = w.get_new_widget(self)
            self._main_layout.Add(self._setting_section, 0, wx.ALIGN_CENTER)
            break

        progress_bar.reset(poses.get_frame_count())
        progress_bar.message(f"Updating {score_engine.get_name()} Scores.")
        scores = score_engine.compute_scores(poses, progress_bar)
        bad_labels = score_engine.compute_bad_indexes(scores)

        self._prob_displayer = ProbabilityDisplayer(
            self, scores, bad_labels, score_engine.get_name()
        )
        self._main_layout.Add(self._prob_displayer, 1, wx.EXPAND)

        self.SetSizerAndFit(self._main_layout)

    def _on_setting_change(self, val: Any):
        new_bad_labels = self._score_engine.compute_bad_indexes(
            self._prob_displayer.get_data()
        )
        self._prob_displayer.set_bad_locations(new_bad_labels)
        self._prob_displayer.Refresh()

    def update_all(self, poses: Pose, progress_bar: ProgressBar):
        progress_bar.reset(poses.get_frame_count())
        progress_bar.message(f"Updating {self._score_engine.get_name()} Scores.")
        new_scores = self._score_engine.compute_scores(poses, progress_bar)
        new_bad_labels = self._score_engine.compute_bad_indexes(new_scores)
        self._prob_displayer.set_data(new_scores)
        self._prob_displayer.set_bad_locations(new_bad_labels)

    def update_partial(self, poses: Pose, progress_bar: ProgressBar, slices: Iterable[slice]):
        progress_bar.reset(sum(len(range(*s.indices(poses.get_frame_count()))) for s in slices))
        progress_bar.message(f"Updating {self._score_engine.get_name()} Scores.")

        data = np.copy(self._prob_displayer.get_data())

        for s in slices:
            new_scores = self._score_engine.compute_scores(poses, progress_bar, s)
            data[s] = new_scores

        bad_labels = self._score_engine.compute_bad_indexes(data)
        self._prob_displayer.set_data(data)
        self._prob_displayer.set_bad_locations(bad_labels)

    def update_at(self, frame: int, value: float):
        self._prob_displayer.set_data_at(frame, value)
        # Update errors in display...
        idx = self._score_engine.compute_bad_indexes(np.array([value]))
        locs = self._prob_displayer.get_bad_locations()
        self._prob_displayer.set_bad_locations(np.unique(np.append(locs, idx + frame)))

    def get_data_at(self, frame: int) -> float:
        return self._prob_displayer.get_data_at(frame)

    def get_data(self) -> np.ndarray:
        return self._prob_displayer.get_data()

    def get_location(self) -> int:
        return self._prob_displayer.get_location()

    def set_location(self, frame: int):
        self._prob_displayer.set_location(frame)

    def get_next_bad_location(self) -> int:
        return self._prob_displayer.get_next_bad_location()

    def get_prev_bad_location(self) -> int:
        return self._prob_displayer.get_prev_bad_location()

    def get_user_modified_locations(self) -> np.ndarray:
        return self._prob_displayer.get_user_modified_locations()

    def set_prior_modified_user_locations(self, value: np.ndarray):
        self._prob_displayer.set_prior_modified_user_locations(value)

    def get_prior_modified_user_locations(self) -> np.ndarray:
        return self._prob_displayer.get_prior_modified_user_locations()

    def set_segment_starts(self, value: np.ndarray):
        self._prob_displayer.set_segment_starts(value)

    def set_segment_fix_frames(self, value: np.ndarray):
        self._prob_displayer.set_segment_fix_frames(value)


