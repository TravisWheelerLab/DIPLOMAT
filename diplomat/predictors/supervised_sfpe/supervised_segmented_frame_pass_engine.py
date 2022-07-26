import os
import traceback

# We first check if this is a headless environment, and if so don't even allow this module to be imported...
from collections import UserList
from pathlib import Path

from diplomat.predictors.supervised_fpe.labelers import Approximate, Point
from diplomat.predictors.supervised_fpe.scorers import EntropyOfTransitions, MaximumJumpInStandardDeviations

if os.environ.get('DLClight', default=False) == 'True':
    raise ImportError("Can't use this module in DLClight mode!")

from typing import Optional, Dict, Tuple, List
from diplomat.predictors.sfpe.segmented_frame_pass_engine import SegmentedFramePassEngine
from diplomat.predictors.supervised_fpe.guilib.fpe_editor import FPEEditor
from diplomat.predictors.fpe.sparse_storage import ForwardBackwardFrame, ForwardBackwardData, SparseTrackingData
from diplomat.processing import *

import cv2
import wx
import numpy as np


class SegmentedSubList(UserList):
    def __init__(self, l: List, segment: np.ndarray, segment_alignment: np.ndarray):
        super().__init__()
        self.data = l
        self._segment = segment
        self._segment_alignment = segment_alignment

    def __getitem__(self, item: int):
        return self.data[self._segment_alignment[item]]

    def __setitem__(self, key: int, value):
        self.data[self._segment_alignment[key]] = value

class SegmentedList(UserList):
    def __init__(self, l: List, segments: np.ndarray, segment_alignments: np.ndarray):
        super().__init__()
        self.data = l
        self._segments = segments
        self._segment_alignments = segment_alignments

    def _segment_find(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        si = np.searchsorted(self._segments[:, 1], index, "right")
        return self._segments[si], self._segment_alignments[si]

    def __getitem__(self, item: int):
        return SegmentedSubList(self.data[item], *self._segment_find(item))

    def __setitem__(self, key: int, value):
        self.data[key] = list(SegmentedSubList(value, *self._segment_find(key)))

class SegmentedFramePassData(ForwardBackwardData):
    def __init__(self, data: ForwardBackwardData, segments: np.ndarray, segment_alignments: np.ndarray):
        super().__init__(0, 0)
        self._frames = data.frames
        self._num_bps = data.num_bodyparts
        self._metadata = data.metadata
        self.allow_pickle = data.allow_pickle
        self._segments = segments
        self._segment_alignments = segment_alignments

    @property
    def frames(self):
        return SegmentedList(self._frames, self._segments, self._segment_alignments)

    @frames.setter
    def frames(self, frames):
        raise NotImplementedError("Not allowed to modify the frames directly through this view!")


class SupervisedSegmentedFramePassEngine(SegmentedFramePassEngine):

    RERUN_HIST_EVT = "engine_rerun"

    def __init__(
        self,
        bodyparts: List[str],
        num_outputs: int,
        num_frames: int,
        settings: Config,
        video_metadata: Config
    ):
        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata)

        if(video_metadata["orig-video-path"] is None):
            raise ValueError("Unable to find the original video file, which is required by this plugin!")

        self._video_path = video_metadata["orig-video-path"]
        self._video_hdl: Optional[cv2.VideoCapture] = None
        self._final_probabilities = None
        self._fb_editor: Optional[FPEEditor] = None

        self._changed_frames: Dict[Tuple[int, int], ForwardBackwardFrame] = {}

    def _get_names(self):
        """
        PRIVATE: Returns a list of strings being the expanded list of body part names (to fill in for when
        num_outputs > 1).
        """
        return [
            self.bodyparts[bp_idx // self.num_outputs] + str((bp_idx % self.num_outputs) + 1)
            for bp_idx in range(self._num_total_bp)
        ]

    def _get_crop_box(self) -> Optional[Tuple[int, int, int, int]]:
        """
        PRIVATE: Get the cropping box of the passed video, uses internally stored _vid_meta dictionary.
        """
        offset = self.video_metadata["cropping-offset"]
        down_scaling = self._frame_holder.metadata.down_scaling

        if(offset is not None):
            y, x = offset
            h, w = self._frame_holder.metadata.width, self._frame_holder.metadata.height
            w, h = w * down_scaling, h * down_scaling
            return (int(x), int(y), int(w), int(h))

        return None

    def _confirm_action(self, is_undo: bool) -> bool:
        """
        PRIVATE: Asks the user if they are sure they want to rerun Forward Backward, returning a boolean based on the
        user's response.

        :param is_undo: True if this is an undo event, False if it is a redo event. Changes the message presented to
                        the user in the confirmation dialog.

        :returns: A boolean, True if the user confirmed they wanted the action done, otherwise false.
        """
        message = (f"Are you sure you want to {'undo' if is_undo else 'redo'} the Passes? "
                   f"Undoing this step might take a while.")
        caption = f"Confirm {'Undo' if is_undo else 'Redo'}"
        style = wx.YES_NO | wx.CANCEL | wx.CANCEL_DEFAULT | wx.CENTRE | wx.ICON_WARNING

        with wx.MessageDialog(self._fb_editor, message, caption, style) as msg:
            result = msg.ShowModal()

        return result == wx.ID_YES


    @property
    def frame_data(self) -> ForwardBackwardData:
        return SegmentedFramePassData(self._frame_holder, self._segments, self._segment_bp_order)

    @property
    def changed_frames(self) -> Dict[Tuple[int, int], ForwardBackwardFrame]:
        return self._changed_frames

    def get_maximum_with_defaults(self, frame) -> Tuple[int, int, float, float, float]:
        return self.get_maximum(frame, self.settings.relaxed_maximum_radius)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @classmethod
    def scmap_to_video_coord(
        cls,
        x_scmap: float,
        y_scmap: float,
        prob: float,
        x_off: float,
        y_off: float,
        down_scaling: int
    ) -> Tuple[float, float, float]:
        x_video = (x_scmap + 0.5) * down_scaling + x_off
        y_video = (y_scmap + 0.5) * down_scaling + y_off
        return (x_video, y_video, prob)

    def video_to_scmap_coord(self, coord: Tuple[float, float, float]) -> Tuple[int, int, float, float, float]:
        """
        PRIVATE: Convert a coordinate in video space to a coordinate in source map space.

        :param coord: A tuple of (x, y, probability), the x and y being represented as floating point numbers in video
                      pixel space.
        :returns: A tuple of (x index, y index, x offset, y offset, probability) being the coordinate stored in source
                  map space.
        """
        down_scaling = self._frame_holder.metadata.down_scaling

        vid_x, vid_y, prob = coord
        x, off_x = divmod(vid_x, down_scaling)
        y, off_y = divmod(vid_y, down_scaling)
        # Correct offsets to be relative to the center of the stride block...
        off_x = off_x - (down_scaling * 0.5)
        off_y = off_y - (down_scaling * 0.5)

        return (int(x), int(y), off_x, off_y, prob)

    def _make_plots(self, evt = None):
        # TODO...
        raise NotImplementedError

    def _on_frame_export(self, export_type: int, file_format: str, file_path: Path) -> Tuple[bool, str]:
        # TODO...
        raise NotImplementedError

    def _on_hist_fb(self, old_data: Tuple[np.ndarray, Dict[Tuple[int, int], ForwardBackwardFrame]]):
        """
        PRIVATE: Used for handling undo/redo Forward Backward events in history. Takes a older/newer edited point state
        and restores it, returning the current state as a new history event to be added to the history

        :param old_data: A tuple containing the following items:
                            - A numpy array storing user modified locations before FB was run...
                            - A dictionary of frame and body part index tuples to original sparse source map frames.
                              Represents original point state since last FB was run on the data.

        :returns: The current point edit state in the same format as old_dict. This gets added to history.
        """
        # TODO...
        raise NotImplementedError

    def _on_run_fb(self, submit_evt = True) -> bool:
        # TODO...
        raise NotImplementedError

    def on_end(self, progress_bar: ProgressBar) -> Optional[Pose]:
        self._run_frame_passes(progress_bar)

        progress_bar.message("Selecting Maximums")
        poses = self.get_maximums(
            self._frame_holder,
            self._segments,
            self._segment_bp_order,
            progress_bar,
            relaxed_radius=self.settings.relaxed_maximum_radius
        )

        probs = np.transpose(poses.get_all_prob())

        self._video_hdl = cv2.VideoCapture(self._video_path)

        app = wx.App()

        self._fb_editor = FPEEditor(
            None,
            self._video_hdl,
            probs,
            poses,
            self._get_names(),
            self.video_metadata,
            self._get_crop_box(),
            [Approximate(self), Point(self)],
            [EntropyOfTransitions(self), MaximumJumpInStandardDeviations(self)]
        )

        self._fb_editor.plot_button.Bind(wx.EVT_BUTTON, self._make_plots)
        self._fb_editor.set_frame_exporter(self._on_frame_export)
        self._fb_editor.history.register_undoer(self.RERUN_HIST_EVT, self._on_hist_fb)
        self._fb_editor.history.register_redoer(self.RERUN_HIST_EVT, self._on_hist_fb)
        self._fb_editor.history.register_confirmer(self.RERUN_HIST_EVT, self._confirm_action)
        self._fb_editor.set_fb_runner(self._on_run_fb)

        self._fb_editor.Show()

        app.MainLoop()

        return self._fb_editor.video_player.video_viewer.get_all_poses()

    @classmethod
    def get_tests(cls) -> Optional[List[TestFunction]]:
        return None

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True

