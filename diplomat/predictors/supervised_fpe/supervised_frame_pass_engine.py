import traceback

from typing import Union, List, Tuple, Dict, Optional
from diplomat.wx_gui.fpe_editor import FPEEditor
from diplomat.wx_gui.progress_dialog import FBProgressDialog
from ..fpe.frame_pass_engine import FramePassEngine, SparseTrackingData
from ..fpe.sparse_storage import ForwardBackwardFrame, ForwardBackwardData
from .labelers import Approximate, Point, NearestPeakInSource, ApproximateSourceOnly
from .scorers import EntropyOfTransitions, MaximumJumpInStandardDeviations

import wx
import cv2
import numpy as np
from diplomat.processing import *
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


class SupervisedFramePassEngine(FramePassEngine):
    """
    A predictor that applies the frame pass engine to frames in order to predict poses.
    This version includes an additional GUI for previewing and modifying results before saving them.
    """

    RERUN_HIST_EVT = "rerun_fb"

    def __init__(
        self,
        bodyparts: List[str],
        num_outputs: int,
        num_frames: int,
        settings: Config,
        video_metadata: Config,
    ):
        super().__init__(
            bodyparts, num_outputs, num_frames, settings, video_metadata
        )

        if(video_metadata["orig-video-path"] is None):
            raise ValueError("Unable to find the original video file, which is required by this plugin!")

        self._video_path = video_metadata["orig-video-path"]
        self._video_hdl: Optional[cv2.VideoCapture] = None
        self._final_probabilities = None
        self._fb_editor: Optional[FPEEditor] = None

        self._changed_frames: Dict[Tuple[int, int], ForwardBackwardFrame] = {}

    def on_end(self, progress_bar: ProgressBar) -> Union[None, Pose]:
        self._run_frame_passes(progress_bar)

        poses = self.get_maximums(
            self._frame_holder, progress_bar,
            relaxed_radius=self.settings.relaxed_maximum_radius
        )

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
            list(range(1, self.num_outputs + 1)) * (self._num_total_bp // self.num_outputs)
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

    @property
    def frame_data(self) -> ForwardBackwardData:
        return self._frame_holder

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

    def _on_frame_export(self, export_type: int, file_format: str, file_path: Path) -> Tuple[bool, str]:
        """
        PRIVATE: Handles frame export. Excepts a format string "DLFS" or "HDF5" and a file path, and dumps the frames
                 to that location...
        """
        down_scaling = self._frame_holder.metadata.down_scaling
        self._fb_editor.Enable(False)

        try:
            with FBProgressDialog(self._fb_editor, inner_msg="Exporting Frames...") as d:
                d.Show()
                with file_path.open("wb") as f:
                    with self._get_frame_writer(
                        self.num_frames, self._frame_holder.metadata, self.video_metadata, file_format, f
                    ) as exporter:
                        header = exporter.get_header()

                        for f_idx in d.progress_bar(range(self.num_frames)):

                            frame_data = TrackingData.empty_tracking_data(1, self._num_total_bp, header.frame_width,
                                                                          header.frame_height, down_scaling, True)

                            for bp_idx in range(self._num_total_bp):
                                if(export_type == 0):
                                    res = self._frame_holder.frames[f_idx][bp_idx].orig_data
                                elif(export_type == 1):
                                    if ((f_idx, bp_idx) in self._changed_frames):
                                        data = self._changed_frames[(f_idx, bp_idx)].orig_data.unpack()
                                    else:
                                        data = self._frame_holder.frames[f_idx][bp_idx].orig_data.unpack()

                                    probs = self._frame_holder.frames[f_idx][bp_idx].frame_probs
                                    if(probs is None):
                                        probs = np.zeros(len(data[2]), np.float32)

                                    res = SparseTrackingData()
                                    res.pack(*data[:2], probs, *data[3:])
                                else:
                                    return (False, "Unsupported Export Type!")
                                res = res.desparsify(header.frame_width, header.frame_height, header.stride)

                                frame_data.get_prob_table(0, bp_idx)[:] = res.get_prob_table(0, 0)
                                frame_data.get_offset_map()[0, :, :, bp_idx, :] = res.get_offset_map()[:, :, 0, :]

                            exporter.write_data(frame_data)
        except (IOError, OSError, ValueError) as e:
            traceback.print_exc()
            return (False, f"An error occurred while saving the file: {str(e)}")
        finally:
            self._fb_editor.Enable(True)

        return (True, "")

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

    def _on_run_fb(self, submit_evt = True) -> bool:
        """
        PRIVATE: Method is run whenever the Forward Backward Algorithm is rerun on the data. Runs the FB Algorithm and
        then updates the UI to display the changed data.

        :param submit_evt: A boolean, determines if this should submit a new history event. Undo/Redo actions call
                           this method with this parameter set to false, otherwise is defaults to true.

        :returns: False. Tells the FBEditor that it should never clear the history when this method is run.
        """
        if(submit_evt and len(self._changed_frames) == 0):
            return False
        # Build the progress dialog....
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

            self._changed_frames = {}

            self._run_frame_passes(dialog.progress_bar, fresh_run=True)
            poses = self.get_maximums(
                self._frame_holder, dialog.progress_bar,
                relaxed_radius=self.settings.relaxed_maximum_radius
            )

            # We now need to update the UI...
            self._fb_editor.video_player.video_viewer.set_all_poses(poses)

            dialog.set_inner_message("Updating Scores...")
            for score in self._fb_editor.score_displays:
                score.update_all(poses, dialog.progress_bar)
                score.set_prior_modified_user_locations(new_user_modified_frames)

            self._fb_editor.Enable(True)

        # Return false to not clear the history....
        return False

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

    def _on_hist_fb(self, old_data: Tuple[np.ndarray, Dict[Tuple[int, int], SparseTrackingData]]):
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

        current_data = {}
        frames = self._frame_holder.frames

        for loc, sparse_frame in old_dict.items():
            frm, bp = loc
            current_data[loc] = frames[frm][bp]
            frames[frm][bp] = sparse_frame

        new_user_mods = np.array([], dtype=np.uint64)
        for score in self._fb_editor.score_displays:
            new_user_mods = score.get_prior_modified_user_locations()
            break

        current_dict = self._changed_frames
        self._on_run_fb(False)
        self._changed_frames = old_dict

        # Vital: If _frame_holder got updated when FB was run.
        frames = self._frame_holder.frames
        d_scale = self._frame_holder.metadata.down_scaling

        for score in self._fb_editor.score_displays:
            score.set_prior_modified_user_locations(old_user_mods)

        for (frm, bp), sparse_frame in current_data.items():
            frames[frm][bp] = sparse_frame

            x, y, prob = self.scmap_to_video_coord(
                *self.get_maximum(
                    sparse_frame,
                    relaxed_radius=self.settings.relaxed_maximum_radius
                ),
                d_scale
            )

            for score in self._fb_editor.score_displays:
                score.update_at(frm, np.nan)
            self._fb_editor.video_player.video_viewer.set_pose(
                frm, bp, (x, y, prob)
            )

        return (new_user_mods, current_dict)

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

    def _get_names(self):
        """
        PRIVATE: Returns a list of strings being the expanded list of body part names (to fill in for when
        num_outputs > 1).
        """
        return [self.bodyparts[bp_idx // self.num_outputs] + str((bp_idx % self.num_outputs) + 1)
                for bp_idx in range(self._num_total_bp)]

    def _make_plots(self, evt = None):
        """
        PRIVATE: Produces all plots for a given frame, showing body part probability maps, both after and before FB
        algorithm.

        :param evt: A button event, note this is just so it can be hooked up to a wx Event hookup without a lambda, not
                    actually used and defaults to None.
        """
        # Get the frame...
        frame_idx = self._fb_editor.video_player.video_viewer.get_offset_count()

        new_bitmap_list = []

        # For every body part...
        for bp_idx in range(self._num_total_bp):
            bp_name = self.bodyparts[bp_idx // self.num_outputs] + str((bp_idx % self.num_outputs) + 1)

            all_data = self._frame_holder.frames[frame_idx][bp_idx]
            if((frame_idx, bp_idx) in self._changed_frames):
                f = self._changed_frames[(frame_idx, bp_idx)]
                frames, occluded = f.frame_probs, f.occluded_probs
            else:
                frames, occluded = all_data.frame_probs, all_data.occluded_probs

            if(frames is not None):
                # PHASE 1: Generate post forward backward probability map data, as a colormap.
                figure = plt.figure(figsize=(3.6, 2.8), dpi=100)
                axes = figure.gca()
                axes.set_title(bp_name + " Post Passes")

                if((frame_idx, bp_idx) in self._changed_frames):
                    data = self._changed_frames[(frame_idx, bp_idx)].orig_data.unpack()
                else:
                    data = all_data.orig_data.unpack()

                track_data = SparseTrackingData()
                track_data.pack(*data[:2], frames, *data[3:])
                h, w = self._frame_holder.metadata.height, self._frame_holder.metadata.width
                down_scaling = self._frame_holder.metadata.down_scaling
                track_data = track_data.desparsify(w, h, down_scaling)
                axes.imshow(track_data.get_prob_table(0, 0))
                plt.tight_layout()
                figure.canvas.draw()

                w, h = figure.canvas.get_width_height()
                new_bitmap_list.append(wx.Bitmap.FromBufferRGBA(w, h, figure.canvas.buffer_rgba()))
                axes.cla()
                figure.clf()
                plt.close(figure)

            # PHASE 2: Generate original source probability map data, as a colormap.
            figure = plt.figure(figsize=(3.6, 2.8), dpi=100)
            axes = figure.gca()
            axes.set_title(bp_name + " Original Source Frame")
            h, w = self._frame_holder.metadata.height, self._frame_holder.metadata.width
            down_scaling = self._frame_holder.metadata.down_scaling
            if((frame_idx, bp_idx) in self._changed_frames):
                track_data = self._changed_frames[(frame_idx, bp_idx)].orig_data.desparsify(w, h, down_scaling)
            else:
                track_data = all_data.orig_data.desparsify(w, h, down_scaling)
            axes.imshow(track_data.get_prob_table(0, 0))
            plt.tight_layout()
            figure.canvas.draw()

            w, h = figure.canvas.get_width_height()
            new_bitmap_list.append(wx.Bitmap.FromBufferRGBA(w, h, figure.canvas.buffer_rgba()))
            axes.cla()
            figure.clf()
            plt.close(figure)

            # If user edited, show user edited frame...
            if((frame_idx, bp_idx) in self._changed_frames):
                figure = plt.figure(figsize=(3.6, 2.8), dpi=100)
                axes = figure.gca()
                axes.set_title(bp_name + " Modified Source Frame")
                h, w = self._frame_holder.metadata.height, self._frame_holder.metadata.width
                track_data = all_data.orig_data.desparsify(w, h, down_scaling)
                axes.imshow(track_data.get_prob_table(0, 0))
                plt.tight_layout()
                figure.canvas.draw()

                w, h = figure.canvas.get_width_height()
                new_bitmap_list.append(wx.Bitmap.FromBufferRGBA(w, h, figure.canvas.buffer_rgba()))
                axes.cla()
                figure.clf()
                plt.close(figure)

        # Now that we have appended all the above bitmaps to a list, update the ScrollImageList widget of the editor
        # with the new images.
        self._fb_editor.plot_list.set_bitmaps(new_bitmap_list)

    @classmethod
    def get_tests(cls) -> Optional[List[TestFunction]]:
        return None

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True
