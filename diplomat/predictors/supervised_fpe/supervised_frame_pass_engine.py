import os
import traceback

# We first check if this is a headless environment, and if so don't even allow this module to be imported...
if os.environ.get('DLClight', default=False) == 'True':
    raise ImportError("Can't use this module in DLClight mode!")

from typing import Union, List, Tuple, Dict, Optional, Any

try:
    from .guilib.fpe_editor import FPEEditor
    from .guilib.progress_dialog import FBProgressDialog
    from .guilib.score_lib import ScoreEngine
    from ..fpe.frame_pass_engine import FramePassEngine, SparseTrackingData
    from ..fpe.sparse_storage import ForwardBackwardFrame
    from ..fpe import fpe_math
    from .guilib import labeler_lib
except ImportError:
    __package__ = "diplomat.predictors.supervised_fpe"
    from .guilib.fpe_editor import FPEEditor
    from .guilib.progress_dialog import FBProgressDialog
    from .guilib.score_lib import ScoreEngine
    from ..fpe.frame_pass_engine import FramePassEngine, SparseTrackingData
    from ..fpe.sparse_storage import ForwardBackwardFrame
    from ..fpe import fpe_math
    from .guilib import labeler_lib

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
            h, w = self._frame_holder.metadata.width, self._frame_holder.metadata.height
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
        down_scaling: int
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


class Point(labeler_lib.PoseLabeler):
    """
    The manual labeling mode, sets probability map to exact location of the
    user click always.
    """
    def __init__(self, frame_engine: SupervisedFramePassEngine):
        super().__init__()
        self._frame_engine = frame_engine

    def predict_location(
        self,
        frame_idx: int,
        bp_idx: int,
        x: float,
        y: float,
        probability: float
    ) -> Tuple[Any, Tuple[float, float, float]]:
        meta = self._frame_engine._frame_holder.metadata
        frame = self._frame_engine._frame_holder.frames[frame_idx][bp_idx]
        s = self._frame_engine.settings

        if(x is None):
            x, y, prob = self._frame_engine.scmap_to_video_coord(
                *self._frame_engine.get_maximum(frame, s.relaxed_maximum_radius),
                meta.down_scaling
            )
            return ((frame_idx, bp_idx, x, y, 0), (x, y, 0))

        return ((frame_idx, bp_idx, x, y, probability), (x, y, probability))

    def pose_change(self, new_state: Any) -> Any:
        frm, bp, x, y, p = new_state
        changed_frames = self._frame_engine._changed_frames
        frames = self._frame_engine._frame_holder.frames

        x, y, off_x, off_y, prob = self._frame_engine.video_to_scmap_coord((x, y, p))
        old_frame_data = frames[frm][bp]
        is_orig = False

        idx = (frm, bp)
        if (idx not in changed_frames):
            changed_frames[idx] = old_frame_data
            is_orig = True

        new_data = SparseTrackingData()
        if (prob > 0):
            new_data.pack(*[np.array([item]) for item in [y, x, prob, off_x, off_y]])

        new_frame = ForwardBackwardFrame()
        new_frame.orig_data = new_data
        new_frame.disable_occluded = True
        new_frame.ignore_clustering = True

        frames[frm][bp] = new_frame

        return (frm, bp, is_orig, old_frame_data)

    def undo(self, data: Any) -> Any:
        frames = self._frame_engine._frame_holder.frames
        changed_frames = self._frame_engine._changed_frames
        frm, bp, is_orig, frame_data = data

        idx = (frm, bp)
        new_is_orig = False
        new_old_frame_data = frames[frm][bp]

        if (idx not in changed_frames):
            changed_frames[idx] = new_old_frame_data
            new_is_orig = True
        elif (is_orig):
            del changed_frames[idx]

        frames[frm][bp] = frame_data

        return (frm, bp, new_is_orig, new_old_frame_data)

    def redo(self, data: Any) -> Any:
        return self.undo(data)


class Approximate(labeler_lib.PoseLabeler):
    """
    Approximate labeling mode, adds a Gaussian centered around the user
    predicted location to generate a new frame. This makes results 'snap' to
    already existing DLC probs when the user input is close enough the
    DLC predictions.
    """
    def __init__(self, frame_engine: SupervisedFramePassEngine):
        super().__init__()
        self._frame_engine = frame_engine
        self._settings = labeler_lib.SettingCollection(
            user_input_strength = labeler_lib.Slider(500, 1000, 667),
            user_input_spread = labeler_lib.FloatSpin(0.5, None, 20, 1, 4)
        )
        self._cached_gaussian_std = None
        self._cached_gaussian = None

    def _make_gaussian(self, new_std: float):
        self._cached_gaussian_std = new_std
        meta = self._frame_engine._frame_holder.metadata

        d_scale = meta.down_scaling
        std = self._cached_gaussian_std / d_scale
        two_std = min(
            np.ceil(self._cached_gaussian_std * 2),
            max(meta.width, meta.height)
        )

        eval_vals = np.arange(-two_std, two_std + 1)
        x, y = np.meshgrid(eval_vals, eval_vals)
        g = fpe_math.gaussian_formula(0, x, 0, y, std, 1, 0)

        # Filter to improve memory usage, and performance....
        good_loc = g > meta.threshold
        g = g[good_loc]
        x = x[good_loc]
        y = y[good_loc]

        self._cached_gaussian = (
            g.reshape(-1),
            np.asarray([x.reshape(-1), y.reshape(-1)], dtype=int)
        )

    @staticmethod
    def _absorb_frame_data(p1, c1, off1, p2, c2, off2):
        comb_c = np.concatenate([c1.T, c2.T])
        comb_p = np.concatenate([p1, p2])
        comb_off = np.concatenate([off1.T, off2.T])
        from_dlc = np.repeat([True, False], [len(p1), len(p2)])

        sort_idx = np.lexsort([comb_c[:, 1], comb_c[:, 0]])
        comb_c = comb_c[sort_idx]
        comb_p = comb_p[sort_idx]
        comb_off = comb_off[sort_idx]
        from_dlc = from_dlc[sort_idx]

        match_idx, = np.nonzero(np.all(comb_c[1:] == comb_c[:-1], axis=1))
        match_idx_after = match_idx + 1

        comb_p[match_idx_after] = comb_p[match_idx] + comb_p[match_idx_after]
        comb_off[match_idx_after] = comb_off[match_idx]

        return (
            np.delete(comb_p, from_dlc, axis=0),
            np.delete(comb_c, from_dlc, axis=0).T,
            np.delete(comb_off, from_dlc, axis=0).T
        )

    def predict_location(
        self,
        frame_idx: int,
        bp_idx: int,
        x: float,
        y: float,
        probability: float
    ) -> Tuple[Any, Tuple[float, float, float]]:
        info = self._settings.get_values()
        user_amp = info.user_input_strength / 1000
        if(info.user_input_spread != self._cached_gaussian_std):
            self._make_gaussian(info.user_input_spread)

        meta = self._frame_engine._frame_holder.metadata
        modified_frames = self._frame_engine._changed_frames
        if((frame_idx, bp_idx) in modified_frames):
            frame = modified_frames[(frame_idx, bp_idx)]
        else:
            frame = self._frame_engine._frame_holder.frames[frame_idx][bp_idx]

        s = self._frame_engine.settings

        if(x is None):
            x, y, prob = self._frame_engine.scmap_to_video_coord(
                *self._frame_engine.get_maximum(frame, s.relaxed_maximum_radius),
                meta.down_scaling
            )
            return ((frame_idx, bp_idx, None, (x, y)), (x, y, 0))

        xvid, yvid = x, y
        x, y, x_off, y_off, prob = self._frame_engine.video_to_scmap_coord((x, y, probability))
        gp, gc = self._cached_gaussian
        gc = gc + np.array([[x], [y]], dtype=int)

        good_locs = ((0 <= gc[0]) & (gc[0] < meta.width)) & ((0 <= gc[1]) & (gc[1] < meta.height))
        gc = gc[:, good_locs]
        gp = gp[good_locs]

        fy, fx, fp, foffx, foffy = [a if(a is not None) else np.array([]) for a in frame.orig_data.unpack()]
        final_p, (final_x, final_y), (final_off_x, final_off_y) = self._absorb_frame_data(
            fp * ((1 - user_amp) / np.max(fp)),
            np.asarray([fx, fy]),
            np.asarray([foffx, foffy]),
            gp * user_amp,
            gc,
            np.asarray([
                xvid - (gc[0] * meta.down_scaling + meta.down_scaling * 0.5),
                yvid - (gc[1] * meta.down_scaling + meta.down_scaling * 0.5)
            ])
        )

        final_x = final_x.astype(np.int32)
        final_y = final_y.astype(np.int32)
        final_p /= np.max(final_p)

        sp = SparseTrackingData()
        sp.pack(final_y, final_x, final_p, final_off_x, final_off_y)
        temp_f = ForwardBackwardFrame(src_data=sp, frame_probs=final_p)

        x, y, prob = self._frame_engine.scmap_to_video_coord(
            *self._frame_engine.get_maximum(temp_f, s.relaxed_maximum_radius),
            meta.down_scaling
        )

        return ((frame_idx, bp_idx, temp_f, (x, y)), (x, y, prob))

    def pose_change(self, new_state: Any) -> Any:
        frm, bp, suggested_frame, coord = new_state
        changed_frames = self._frame_engine._changed_frames
        frames = self._frame_engine._frame_holder.frames

        old_frame_data = frames[frm][bp]
        is_orig = False

        idx = (frm, bp)
        if (idx not in changed_frames):
            changed_frames[idx] = old_frame_data
            is_orig = True

        if(suggested_frame is None):
            new_data = SparseTrackingData()
            x, y, off_x, off_y, prob = self._frame_engine.video_to_scmap_coord(
                coord + (0,)
            )
            new_data.pack(*[np.array([item]) for item in [y, x, prob, off_x, off_y]])
        else:
            new_data = suggested_frame.src_data

        new_frame = ForwardBackwardFrame()
        new_frame.orig_data = new_data
        new_frame.disable_occluded = True
        new_frame.ignore_clustering = True

        frames[frm][bp] = new_frame

        return (frm, bp, is_orig, old_frame_data)

    def undo(self, data: Any) -> Any:
        frames = self._frame_engine._frame_holder.frames
        changed_frames = self._frame_engine._changed_frames
        frm, bp, is_orig, frame_data = data

        idx = (frm, bp)
        new_is_orig = False
        new_old_frame_data = frames[frm][bp]

        if (idx not in changed_frames):
            changed_frames[idx] = new_old_frame_data
            new_is_orig = True
        elif (is_orig):
            del changed_frames[idx]

        frames[frm][bp] = frame_data

        return (frm, bp, new_is_orig, new_old_frame_data)

    def redo(self, data: Any) -> Any:
        return self.undo(data)

    def get_settings(self) -> Optional[labeler_lib.SettingCollection]:
        return self._settings


def normalized_shanon_entropy(dists: np.ndarray):
    # Entropy in bits / Entropy in bits of uniform distribution of size n...
    dists = dists / np.sum(dists, -1)
    dists[dists == 0] = 1
    return (-np.sum(dists * np.log2(dists), -1)) / np.log2(dists.shape[-1])

class EntropyOfTransitions(ScoreEngine):
    def __init__(self, frame_engine: SupervisedFramePassEngine):
        super().__init__()
        self._frame_engine = frame_engine

        self._gaussian_table = None
        self._std = self._get_std(self._frame_engine._frame_holder.metadata)
        self._init_gaussian_table(frame_engine._width, frame_engine._height)

        self._settings = labeler_lib.SettingCollection(
            threshold=labeler_lib.FloatSpin(0, 1, 0.8, 0.05, 4)
        )

    def _get_std(self, metadata):
        if("optimal_std" in metadata):
            return metadata.optimal_std[2]
        else:
            return 1 / metadata.down_scaling

    def _init_gaussian_table(self, width, height):
        if(self._gaussian_table is None):
            self._gaussian_table = fpe_math.gaussian_table(
                height, width, self._std, 1, 0
            )

    def compute_scores(self, poses: Pose, prog_bar: ProgressBar) -> np.ndarray:
        frames = self._frame_engine._frame_holder

        scores = np.zeros(poses.get_frame_count() + 1, dtype=np.float32)
        num_in_group = frames.metadata.num_outputs
        num_groups = poses.get_bodypart_count() // num_in_group

        for f_i in range(1, frames.num_frames):
            f_list_p = frames.frames[f_i - 1]
            f_list_c = frames.frames[f_i]

            for b_g_i in range(num_groups):
                matrix = np.zeros((num_in_group, num_in_group), dtype=np.float32)

                for b_off_i in range(num_in_group):
                    bp_i = b_g_i * num_in_group + b_off_i
                    for b_off_j in range(num_in_group):
                        bp_j = b_g_i * num_in_group + b_off_j
                        matrix[b_off_i, b_off_j] = self._compute_transition_score(
                            f_list_p[bp_j],
                            f_list_c[bp_i],
                            self._gaussian_table
                        )

                k = np.nanmax(normalized_shanon_entropy(matrix))
                scores[f_i] = max(scores[f_i], k)

            prog_bar.update()

        scores[0] = scores[1]
        scores[-1] = scores[-2]
        scores = (scores[:-1] + scores[1:]) / 2

        return scores

    @staticmethod
    def _compute_transition_score(
        prior_frame: ForwardBackwardFrame,
        current_frame: ForwardBackwardFrame,
        trans_matrix: np.ndarray
    ) -> float:
        py, px, __, __, __ = prior_frame.src_data.unpack()
        cy, cx, __, __, __ = current_frame.src_data.unpack()

        if(py is None or cy is None):
            return 0

        return float(np.sum(
            np.expand_dims(current_frame.frame_probs, 1)
            * fpe_math.table_transition((px, py), (cx, cy), trans_matrix)
            * np.expand_dims(prior_frame.frame_probs, 0)
        ))

    def compute_bad_indexes(self, scores: np.ndarray) -> np.ndarray:
        threshold = self._settings.get_values().threshold
        return np.flatnonzero(scores > threshold)

    def get_settings(self) -> labeler_lib.SettingCollection:
        return self._settings

    def get_name(self) -> str:
        return "Maximum Entropy of Transitions"

class MaximumJumpInStandardDeviations(ScoreEngine):
    def __init__(self, frame_engine: SupervisedFramePassEngine):
        self._frame_engine = frame_engine
        self._std = self._get_std(self._frame_engine._frame_holder.metadata)
        self._pcutoff = self._frame_engine.video_metadata["pcutoff"]

        self._settings = labeler_lib.SettingCollection(
            threshold = labeler_lib.FloatSpin(0.25, 1000, 4, 0.25, 4)
        )

    def _get_std(self, metadata):
        if ("optimal_std" in metadata):
            return metadata.optimal_std[2] * metadata.down_scaling
        else:
            return 1

    def compute_scores(self, poses: Pose, prog_bar: ProgressBar) -> np.ndarray:
        scores = np.zeros(poses.get_frame_count(), dtype=np.float32)

        for f_i in range(1, poses.get_frame_count()):
            prior_x = poses.get_x_at(f_i - 1, slice(None))
            prior_y = poses.get_y_at(f_i - 1, slice(None))
            prior_p = poses.get_prob_at(f_i - 1, slice(None))
            c_x = poses.get_x_at(f_i, slice(None))
            c_y = poses.get_y_at(f_i, slice(None))
            c_p = poses.get_prob_at(f_i, slice(None))


            dists = np.sqrt((c_x - prior_x) ** 2 + (c_y - prior_y) ** 2)
            dists /= self._std
            dists[(prior_p < self._pcutoff) | (c_p < self._pcutoff)] = 0

            scores[f_i] = np.max(dists)

            prog_bar.update()

        return scores

    def compute_bad_indexes(self, scores: np.ndarray) -> np.ndarray:
        threshold = self._settings.get_values().threshold
        return np.flatnonzero(scores > threshold)

    def get_settings(self) -> labeler_lib.SettingCollection:
        return self._settings

    def get_name(self) -> str:
        return "Maximum Jump in Standard Deviations"