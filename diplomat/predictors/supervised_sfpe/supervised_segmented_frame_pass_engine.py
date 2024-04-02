import shutil
import traceback

from collections import UserList
from pathlib import Path

from diplomat.predictors.sfpe.disk_sparse_storage import DiskBackedForwardBackwardData
from diplomat.wx_gui.progress_dialog import FBProgressDialog
from diplomat.predictors.supervised_fpe.labelers import Approximate, Point, NearestPeakInSource, ApproximateSourceOnly
from diplomat.predictors.supervised_fpe.scorers import EntropyOfTransitions, MaximumJumpInStandardDeviations
from typing import Optional, Dict, Tuple, List, MutableMapping, Iterator, Iterable
from diplomat.predictors.sfpe.segmented_frame_pass_engine import SegmentedFramePassEngine, AntiCloseObject
from diplomat.wx_gui.fpe_editor import FPEEditor
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

    def __len__(self):
        return len(self.data)


class SegmentedList(UserList):
    def __init__(self, data: List, segments: np.ndarray, segment_alignments: np.ndarray):
        super().__init__()
        self.data = data
        self._segments = segments
        self._segment_alignments = segment_alignments

    def _segment_find(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        si = np.searchsorted(self._segments[:, 1], index, "right")
        return self._segments[si], self._segment_alignments[si]

    def __getitem__(self, item: int):
        return SegmentedSubList(self.data[item], *self._segment_find(item))

    def __setitem__(self, key: int, value):
        self.data[key] = list(SegmentedSubList(value, *self._segment_find(key)))

    def __delitem__(self, key):
        raise NotImplementedError


class SegmentedDict(MutableMapping):
    def __init__(
        self,
        wrapper_dict: dict,
        segments: np.ndarray,
        segment_alignments: np.ndarray,
        rev_segment_alignments: np.ndarray
    ):
        super().__init__()
        self.data = wrapper_dict
        self._segments = segments
        self._segment_alignments = segment_alignments
        self._rev_segment_alignments = rev_segment_alignments

    def _index_resolve(self, frame: int, bp: int) -> Tuple[int, int]:
        si = np.searchsorted(self._segments[:, 1], frame, "right")
        return frame, self._segment_alignments[si, bp]

    def _rev_index_resolve(self, frame: int, bp: int) -> Tuple[int, int]:
        si = np.searchsorted(self._segments[:, 1], frame, "right")
        return frame, self._rev_segment_alignments[si, bp]

    def __getitem__(self, item):
        return self.data[self._index_resolve(*item)]

    def __setitem__(self, key, value):
        self.data[self._index_resolve(*key)] = value

    def __delitem__(self, key):
        del self.data[self._index_resolve(*key)]

    def __iter__(self) -> Iterator:
        return (self._rev_index_resolve(*k) for k in self.data)

    def __len__(self) -> int:
        return len(self.data)


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
    """
    The supervised (aka interactive) version of the :plugin:`~diplomat.predictors.SegmentedFramePassEngine` predictor.
    Provides a GUI for modifying results at the end of the tracking process.
    """

    RERUN_HIST_EVT = "engine_rerun"

    def __init__(
        self,
        bodyparts: List[str],
        num_outputs: int,
        num_frames: int,
        settings: Config,
        video_metadata: Config,
        restore_path: Optional[str] = None
    ):
        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata, restore_path)

        if(video_metadata["orig-video-path"] is None):
            raise ValueError("Unable to find the original video file, which is required by this plugin!")

        self._video_path = video_metadata["orig-video-path"] if(restore_path is None) else restore_path
        self._video_hdl: Optional[cv2.VideoCapture] = None
        self._final_probabilities = None
        self._fb_editor: Optional[FPEEditor] = None
        self._reverse_segment_bp_order = None

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
            w, h = self._frame_holder.metadata.width, self._frame_holder.metadata.height
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
        return SegmentedFramePassData(
            self._frame_holder,
            self._segments,
            self._reverse_segment_bp_order
        )

    @property
    def changed_frames(self) -> MutableMapping[Tuple[int, int], ForwardBackwardFrame]:
        return SegmentedDict(
            self._changed_frames,
            self._segments,
            self._reverse_segment_bp_order,
            self._segment_bp_order
        )

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
        down_scaling: float
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

    def _make_plot_of(
        self,
        figsize: Tuple[float, float],
        dpi: int, title: str,
        track_data: SparseTrackingData
    ) -> wx.Bitmap:
        # Get the frame...
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=figsize, dpi=dpi)
        axes = figure.gca()
        axes.set_title(title)

        h, w = self.frame_data.metadata.height, self.frame_data.metadata.width
        down_scaling = self.frame_data.metadata.down_scaling
        track_data = track_data.desparsify(w, h, down_scaling)

        axes.imshow(track_data.get_prob_table(0, 0))
        figure.tight_layout()
        figure.canvas.draw()

        w, h = figure.canvas.get_width_height()
        bitmap = wx.Bitmap.FromBufferRGBA(w, h, figure.canvas.buffer_rgba())

        axes.cla()
        figure.clf()
        plt.close(figure)

        return bitmap

    def _custom_multicluster_plot(
        self,
        figsize: Tuple[float, float],
        dpi: int,
        title: str,
        track_datas: List[SparseTrackingData]
    ) -> wx.Bitmap:
        # Get the frame...
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        cmaps = ['Blues', 'Reds', 'Greys', 'Oranges', 'Purples', 'Greens']
        overlap_color = [0.224, 1, 0.078, 1]  # None of the cmaps use neon green, for good reason...

        figure = plt.figure(figsize=figsize, dpi=dpi)
        axes = figure.gca()
        axes.set_title(title)

        h, w = self.frame_data.metadata.height, self.frame_data.metadata.width
        down_scaling = self.frame_data.metadata.down_scaling

        counts = 0
        img = 0

        for track_data, cmap in zip(track_datas, cmaps * int(np.ceil(len(track_datas) / len(cmaps)))):
            cmap = plt.get_cmap(cmap).copy()
            cmap.set_extremes(bad=(0, 0, 0, 0), under=(0, 0, 0, 0), over=(0, 0, 0, 0))

            track_data = track_data.desparsify(w, h, down_scaling).get_prob_table(0, 0)

            track_data /= np.nanmax(track_data)
            track_data *= 0.75

            counts += (track_data != 0)
            track_data[track_data == 0] = -np.inf

            img += cmap(track_data)

        img[counts > 1] = overlap_color
        axes.imshow(img)
        if(np.any(counts > 1)):
            axes.set_title(title + "\n(OVERLAP!)")

        figure.tight_layout()
        figure.canvas.draw()

        w, h = figure.canvas.get_width_height()
        bitmap = wx.Bitmap.FromBufferRGBA(w, h, figure.canvas.buffer_rgba())

        axes.cla()
        figure.clf()
        plt.close(figure)

        return bitmap

    def _make_plots(self, evt = None):
        """
        PRIVATE: Creates plots of data for current frame in UI and puts them in the side panel.
        """
        frame_idx = self._fb_editor.video_player.video_viewer.get_offset_count()

        new_bitmap_list = []
        figsize = (3.6, 2.8)
        dpi = 200

        is_fix_frame = np.any(frame_idx == self._segments[:, 2])
        fix_frame_data = []

        # For every body part...
        for bp_idx in range(self._num_total_bp):
            bp_name = self.bodyparts[bp_idx // self.num_outputs] + str((bp_idx % self.num_outputs) + 1)
            all_data = self.frame_data.frames[frame_idx][bp_idx]

            if ((frame_idx, bp_idx) in self.changed_frames):
                f = self.changed_frames[frame_idx, bp_idx]
                frames, occluded, orig_data = f.frame_probs, f.occluded_probs, f.orig_data
            else:
                frames, occluded, orig_data = all_data.frame_probs, all_data.occluded_probs, all_data.orig_data

            if(frames is not None):
                # Plot post MIT-Viterbi frame data if it exists...
                data = orig_data.unpack()
                track_data = SparseTrackingData()
                track_data.pack(*data[:2], frames, *data[3:])

                new_bitmap_list.append(self._make_plot_of(figsize, dpi, bp_name + " Post Passes", track_data))

            # Plot Pre-MIT-Viterbi frame data, or the original suggested probability frame...
            new_bitmap_list.append(self._make_plot_of(figsize, dpi, bp_name + " Original Source Frame", orig_data))
            if(is_fix_frame):
                fix_frame_data.append(orig_data)

            # If user edited, show user edited frame...
            if ((frame_idx, bp_idx) in self.changed_frames):
                track_data = all_data.orig_data
                new_bitmap_list.append(self._make_plot_of(figsize, dpi, bp_name + " Modified Source Frame", track_data))

            if(len(fix_frame_data) >= self.num_outputs):
                new_bitmap_list.append(
                    self._custom_multicluster_plot(figsize, dpi, bp_name + " Fix Frame Clustering", fix_frame_data)
                )
                fix_frame_data.clear()

        # Now that we have appended all the above bitmaps to a list, update the ScrollImageList widget of the editor
        # with the new images.
        self._fb_editor.plot_list.set_bitmaps(new_bitmap_list)

    def _on_frame_export(self, export_type: int, file_format: str, file_path: Path) -> Tuple[bool, str]:
        # TODO...
        self._fb_editor.Enable(False)

        changed_frames = {}

        if (export_type >= 1):
            # Option is exporting data after latest fpe run, remove the latest user edits...
            for (fi, bpi), frame in self._changed_frames.items():
                changed_frames[(fi, bpi)] = self._frame_holder[fi][bpi]
                self._frame_holder[fi][bpi] = frame

        try:
            with FBProgressDialog(self._fb_editor, title="Export Progress", inner_msg="Exporting Frames...") as dialog:
                dialog.Show()
                self._export_frames(
                    self._frame_holder,
                    self._segments,
                    self._segment_bp_order,
                    self.video_metadata,
                    file_path,
                    file_format,
                    dialog.progress_bar,
                    export_type == 1,
                    export_type == 2
                )
        except (IOError, OSError, ValueError) as e:
            traceback.print_exc()
            return (False, f"An error occurred while saving the file: {str(e)}")
        finally:
            # If we overwrote the latest user edits, put them back in now...
            for (fi, bpi), frame in changed_frames.items():
                self._frame_holder[fi][bpi] = frame

            self._fb_editor.Enable(True)

        return (True, "")

    def _resolve_frame_orderings(
        self,
        progress_bar: ProgressBar,
        reset_bar: bool = True,
        reverse_arr: Optional[np.ndarray] = None
    ):
        # Ignore the last argument, we need to be able to reverse segment ordering...
        self._reverse_segment_bp_order = np.zeros((len(self._segments), self._frame_holder.num_bodyparts), np.uint16)
        return super()._resolve_frame_orderings(progress_bar, reset_bar, self._reverse_segment_bp_order)

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
        old_user_mods, old_dict = old_data

        # Restore the original frames, and temporarily store the user modified frames...
        current_data = {}
        frames = self._frame_holder.frames

        for loc, sparse_frame in old_dict.items():
            frm, bp = loc
            current_data[loc] = frames[frm][bp]
            frames[frm][bp] = sparse_frame

        # Get current user modified frames...
        new_user_mods = np.array([], dtype=np.uint64)
        for score in self._fb_editor.score_displays:
            new_user_mods = score.get_prior_modified_user_locations()
            break

        # Perform a run with the old data put back in place...
        current_dict = self._changed_frames
        # If we are doing an undo, use old_dict, otherwise use the data already in current dict...
        # the fb run doesn't use the data in _current_frames, but uses it to determine which segments
        # need to be rerun...
        self._changed_frames = self.changed_frames if(len(self._changed_frames) > 0) else old_dict
        self._on_run_fb(False)
        self._changed_frames = old_dict

        # Vital: If _frame_holder got updated when FB was run.
        frames = self._frame_holder.frames
        d_scale = self._frame_holder.metadata.down_scaling

        # Restore old user edits...
        for score in self._fb_editor.score_displays:
            score.set_prior_modified_user_locations(old_user_mods)

        for (frm, bp), sparse_frame in current_data.items():
            frames[frm][bp] = sparse_frame

        for (frm, bp), sparse_frame in SegmentedDict(
            current_data,
            self._segments,
            self._reverse_segment_bp_order,
            self._segment_bp_order
        ).items():
            x, y, prob = self.scmap_to_video_coord(
                *self.get_maximum_with_defaults(
                    sparse_frame
                ),
                d_scale
            )

            for score in self._fb_editor.score_displays:
                score.update_at(frm, np.nan)
            self._fb_editor.video_player.video_viewer.set_pose(
                frm, bp, (x, y, prob)
            )

        return (new_user_mods, current_dict)

    def _partial_rerun(
        self,
        changed_frames: Dict[Tuple[int, int], ForwardBackwardFrame],
        old_poses: Pose,
        progress_bar: ProgressBar
    ) -> Tuple[Pose, Iterable[int]]:
        
        #TODO : delete below lines, not doing as expected
        
        # Determine what segments have been manipulated...
        segment_indexes = sorted({np.searchsorted(self._segments[:, 1], f_i, "right") for f_i, b_i in changed_frames})

        poses = old_poses.get_all().reshape((old_poses.get_frame_count(), old_poses.get_bodypart_count(), 3))
        # Restore poses to there original order....
        for (s_i, e_i, f_i), seg_rev in zip(self._segments, self._reverse_segment_bp_order):
            poses[s_i:e_i, :] = poses[s_i:e_i, seg_rev]

        self._run_segmented_passes(progress_bar, segment_indexes)
        self._resolve_frame_orderings(progress_bar)

        # Now compute new order of poses...
        for (s_i, e_i, f_i), seg_ord in zip(self._segments, self._segment_bp_order):
            poses[s_i:e_i, :] = poses[s_i:e_i, seg_ord]
        old_poses.get_all()[:] = poses.reshape(old_poses.get_frame_count(), old_poses.get_bodypart_count() * 3)
        
        
        return (
            self.get_maximums(
                self._frame_holder,
                self._segments,
                self._segment_bp_order,
                progress_bar,
                relaxed_radius=self.settings.relaxed_maximum_radius,
                old_poses=(old_poses, segment_indexes)
            ),
            segment_indexes
        )

    def _on_run_fb(self, submit_evt: bool = True) -> bool:
        """
        PRIVATE: Method is run whenever the Frame Pass Engine is rerun on the data. Runs the Frame Passes only in
        chunks the user has modified and then updates the UI to display the changed data.

        :param submit_evt: A boolean, determines if this should submit a new history event. Undo/Redo actions call
                           this method with this parameter set to false, otherwise is defaults to true.

        :returns: False. Tells the FBEditor that it should never clear the history when this method is run.
        """
        if (submit_evt and len(self._changed_frames) == 0):
            return False

        with FBProgressDialog(self._fb_editor, title="Rerunning Passes...") as dialog:
            dialog.Show()
            self._fb_editor.Enable(False)

            user_modified_frames = np.ndarray([], dtype=np.uint64)
            new_user_modified_frames = np.ndarray([], dtype=np.uint64)
            for score in self._fb_editor.score_displays:
                user_modified_frames = score.get_prior_modified_user_locations()
                new_user_modified_frames = score.get_user_modified_locations()
                break

            if(submit_evt):
                self._fb_editor.history.do(self.RERUN_HIST_EVT, (user_modified_frames, self._changed_frames))

            poses, segments = self._partial_rerun(
                self._changed_frames,
                self._fb_editor.video_player.video_viewer.get_all_poses(),
                AntiCloseObject(dialog.progress_bar)
            )
            segments = [slice(*self._segments[i, :2]) for i in segments]

            self._changed_frames = {}

            self._fb_editor.video_player.video_viewer.set_all_poses(poses)
            dialog.set_inner_message("Updating Scores...")
            for score in self._fb_editor.score_displays:
                score.update_partial(poses, dialog.progress_bar, segments)
                score.set_prior_modified_user_locations(new_user_modified_frames)

            self._fb_editor.Enable(True)

        # Return false to not clear the history....
        return False

    def _copy_to_disk(self, progress_bar: ProgressBar, new_frame_holder: ForwardBackwardData):
        progress_bar.message("Saving to Disk")
        progress_bar.reset(self._frame_holder.num_frames * self._frame_holder.num_bodyparts)

        new_frame_holder.metadata = self._frame_holder.metadata
        for frame_idx in range(len(self._frame_holder.frames)):
            for bodypart_idx in range(len(self._frame_holder.frames[frame_idx])):
                new_frame_holder.frames[frame_idx][bodypart_idx] = self._frame_holder.frames[frame_idx][
                    bodypart_idx]
                progress_bar.update()

    def _on_manual_save(self):
        output_path = Path(self.video_metadata["output-file-path"]).resolve()
        video_path = Path(self.video_metadata["orig-video-path"]).resolve()
        disk_path = output_path.parent / (output_path.stem + ".dipui")

        with disk_path.open("w+b") as disk_ui_file:
            with video_path.open("rb") as f:
                shutil.copyfileobj(f, disk_ui_file)

            with DiskBackedForwardBackwardData(
                self.num_frames,
                self._num_total_bp,
                disk_ui_file,
                self.settings.memory_cache_size
            ) as disk_frame_holder:
                with FBProgressDialog(self._fb_editor, title="Save to Disk") as dialog:
                    dialog.Show()
                    self._fb_editor.Enable(False)
                    self._copy_to_disk(dialog.progress_bar, disk_frame_holder)
                    self._fb_editor.Enable(True)

    def _on_visual_settings_change(self, data):
        old_data = self._frame_holder.metadata["video_metadata"]
        old_data.update(data)
        self._frame_holder.metadata["video_metadata"] = old_data

    def _on_end(self, progress_bar: ProgressBar) -> Optional[Pose]:
        if(self._restore_path is None):
            self._run_frame_passes(progress_bar)
            self._frame_holder.metadata["segments"] = self._segments.tolist()
            self._frame_holder.metadata["segment_scores"] = self._segment_scores.tolist()
        else:
            progress_bar.reset(self._frame_holder.num_frames * self._frame_holder.num_bodyparts)
            progress_bar.message("Restoring Partial Frames")
            for frame_list in self._frame_holder.frames:
                for frame in frame_list:
                    if(frame.frame_probs is None):
                        frame.frame_probs = frame.src_data.probs[:]
                    progress_bar.update()

            self._width = self._frame_holder.metadata.width
            self._height = self._frame_holder.metadata.height
            self._resolve_frame_orderings(progress_bar)

        progress_bar.message("Selecting Maximums")
        poses = self.get_maximums(
            self._frame_holder,
            self._segments,
            self._segment_bp_order,
            progress_bar,
            relaxed_radius=self.settings.relaxed_maximum_radius
        )

        if(self._restore_path is None and self.settings.storage_mode == "hybrid"):
            new_frame_holder = self.get_frame_holder()
            self._copy_to_disk(progress_bar, new_frame_holder)
            self._frame_holder = new_frame_holder
            self._frame_holder._frames.flush()

        self._video_hdl = cv2.VideoCapture(self._video_path)

        app = wx.App()

        self._fb_editor = FPEEditor(
            None,
            self._video_hdl,
            poses,
            self._get_names(),
            self.video_metadata,
            self._get_crop_box(),
            [Approximate(self), Point(self), NearestPeakInSource(self), ApproximateSourceOnly(self)],
            [EntropyOfTransitions(self), MaximumJumpInStandardDeviations(self)],
            None,
            list(range(1, self.num_outputs + 1)) * (self._num_total_bp // self.num_outputs),
            self._on_manual_save if(self.settings.storage_mode == "memory") else None
        )

        for s in self._fb_editor.score_displays:
            s.set_segment_starts(self._segments[:, 0])
            s.set_segment_fix_frames(self._segments[:, 2])

        self._fb_editor.plot_button.Bind(wx.EVT_BUTTON, self._make_plots)
        self._fb_editor.set_frame_exporter(self._on_frame_export)
        self._fb_editor.history.register_undoer(self.RERUN_HIST_EVT, self._on_hist_fb)
        self._fb_editor.history.register_redoer(self.RERUN_HIST_EVT, self._on_hist_fb)
        self._fb_editor.history.register_confirmer(self.RERUN_HIST_EVT, self._confirm_action)
        self._fb_editor.set_fb_runner(self._on_run_fb)
        self._fb_editor.set_plot_settings_changer(self._on_visual_settings_change)

        self._fb_editor.Show()

        app.MainLoop()
        self._video_hdl.release()

        return self._fb_editor.video_player.video_viewer.get_all_poses()

    @classmethod
    def get_tests(cls) -> Optional[List[TestFunction]]:
        return None

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True

