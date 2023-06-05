"""
Contains utilities for loading user tracks into a lighter version of the supervised UI to allow for minor modifications
to user saved tracking data.
"""

import os
from typing import List, Any, Dict, Optional, Tuple, MutableMapping, Sequence, Union, Callable
import cv2
import numpy as np
from diplomat.predictors.sfpe.segmented_frame_pass_engine import SegmentedFramePassEngine
from diplomat.predictors.fpe.sparse_storage import ForwardBackwardData, ForwardBackwardFrame, SparseTrackingData
from diplomat.processing import Pose, Config, ProgressBar


class UIImportError(ImportError):
    """
    This error is thrown when TweakUI is unable to import the required UI toolkit packages, it typically indicates
    the user has installed DIPLOMAT without GUI support and so UI packages are missing.
    """
    pass


class _DummySubPoseList(Sequence[ForwardBackwardFrame]):
    def __init__(self, sub_index: Union[int, slice], poses: Pose):
        self._sub_index = sub_index
        self._poses = poses

    def __getitem__(self, index: Union[int, slice]) -> Union[ForwardBackwardFrame, List[ForwardBackwardFrame], List[List[ForwardBackwardFrame]]]:
        x = self._poses.get_x_at(self._sub_index, index)
        y = self._poses.get_y_at(self._sub_index, index)
        prob = self._poses.get_prob_at(self._sub_index, index)
        dims = np.ndim(x)

        if(dims == 0):
            return self._get_item_single(x, y, prob)
        elif(dims == 1):
            return [self._get_item_single(subx, suby, subp) for subx, suby, subp in zip(x, y, prob)]
        else:
            return [[self._get_item_single(ssx, ssy, ssp) for ssx, ssy, ssp in zip(sx, sy, sp)] for sx, sy, sp in zip(x, y, prob)]

    def __setitem__(self, key: Union[int, slice], value: List[ForwardBackwardFrame]):
        pass

    @staticmethod
    def _get_item_single(x: float, y: float, p: float) -> ForwardBackwardFrame:
        if(np.isnan(x) or np.isnan(y)):
            x, y, p = 0, 0, 0

        sx, sy, sp, sox, soy = _DummyFramePassEngine.video_to_scmap_coord((x, y, p))

        res = SparseTrackingData().pack([sy], [sx], [sp], [sox], [soy])
        return ForwardBackwardFrame(
            orig_data=res,
            src_data=res,
            frame_probs=np.array([sp])
        )

    def __len__(self) -> int:
        return self._poses.get_bodypart_count()


class _DummyPoseList(Sequence[_DummySubPoseList]):
    def __init__(self, poses: Pose):
        self._poses = poses

    def __getitem__(self, index: Union[int, slice]) -> _DummySubPoseList:
        return _DummySubPoseList(index, self._poses)

    def __setitem__(self, key, value):
        pass

    def __len__(self) -> int:
        return self._poses.get_frame_count()


class _DummyForwardBackwardData(ForwardBackwardData):
    def __init__(self, poses: Pose, num_outputs: int):
        super().__init__(0, 0)
        self._frames = _DummyPoseList(poses)
        self._num_bps = poses.get_bodypart_count()
        self.metadata.down_scaling = 1
        self.metadata.num_outputs = num_outputs

    @property
    def frames(self) -> Sequence[Sequence[ForwardBackwardFrame]]:
        return self._frames

    @frames.setter
    def frames(self, val: Sequence[Sequence[ForwardBackwardFrame]]):
        raise NotImplementedError("Direct setting not supported by this dummy data structure...")


class _DummyFramePassEngine:
    def __init__(
        self,
        poses: Pose,
        crop_box: Tuple[int, int, int, int],
        video_meta: Dict[str, Any],
        num_outputs: int, prog_bar: ProgressBar
    ):
        self._poses = poses
        self._size = (int(crop_box[2]), int(crop_box[3]))
        self._changed_frames = {}
        self._video_meta = Config(video_meta)
        self._num_outputs = num_outputs

        self._fake_fb_data = _DummyForwardBackwardData(self._poses, self._num_outputs)

        from diplomat.predictors.fpe.frame_passes.optimize_std import OptimizeStandardDeviation
        optimizer = OptimizeStandardDeviation(self.width, self.height, True, Config({}, OptimizeStandardDeviation.get_config_options()))
        optimizer.run_pass(self.frame_data, prog_bar, True, True)

    @property
    def frame_data(self) -> ForwardBackwardData:
        return self._fake_fb_data

    @property
    def video_metadata(self) -> Config:
        return self._video_meta

    @property
    def width(self) -> int:
        return self._size[0]

    @property
    def height(self) -> int:
        return self._size[1]

    @property
    def changed_frames(self) -> MutableMapping[Tuple[int, int], ForwardBackwardFrame]:
        return self._changed_frames

    @staticmethod
    def video_to_scmap_coord(coord: Tuple[float, float, float]) -> Tuple[int, int, float, float, float]:
        vid_x, vid_y, prob = coord
        x, off_x = divmod(vid_x, 1)
        y, off_y = divmod(vid_y, 1)
        # Correct offsets to be relative to the center of the stride block...
        off_x = off_x - 0.5
        off_y = off_y - 0.5

        return (int(x), int(y), off_x, off_y, prob)

    @staticmethod
    def scmap_to_video_coord(
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

    @staticmethod
    def get_maximum_with_defaults(frame: ForwardBackwardFrame) -> Tuple[int, int, float, float, float]:
        return SegmentedFramePassEngine.get_maximum(frame, 0)


def _simplify_editor_class(wx, editor_class):
    class SimplifiedEditor(editor_class):
        def __init__(self, do_save, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._toolbar.RemoveTool(self._run.GetId())
            self._toolbar.RemoveTool(self._export_btn.GetId())

            idx = self._tools.index(self._run)
            del self._tools[idx]
            del self._bitmaps[idx]
            idx2 = self._tools.index(self._export_btn)
            del self._tools[idx2]
            del self._bitmaps[idx2]

            self._video_splitter.Unsplit(self._plot_panel)
            self.Bind(wx.EVT_CLOSE, self.on_close)
            self._was_save = False
            self._do_save = do_save

        def on_close(self, evt):
            if (not self._was_save and evt.CanVeto()):
                res = wx.MessageBox(
                    "Are you sure you want to close without saving your results?",
                    "Confirmation",
                    wx.ICON_QUESTION | wx.YES_NO,
                    self
                )

                if(res != wx.YES):
                    evt.Veto()
                else:
                    evt.Skip(True)
                return

            try:
                self._do_save(self._was_save, self.video_player.video_viewer.get_all_poses())
            finally:
                evt.Skip(True)

        def on_tool(self, evt):
            if (evt.GetId() == self._save.GetId()):
                self._was_save = True
                self.Close()
                return

            super().on_tool(evt)

    return SimplifiedEditor


class TweakUI:
    """
    The tweak UI manager. Provides a functionality for creating a UI for modifying user tracks. Should be used by frontends to implementing
    DIPLOMAT's tweak command functionality.
    """
    def __init__(self):
        """
        Create a tweak UI manager, which can be used to make modifications to user tracks passed to it.

        :raises UIImportError: If the UI manager it is unable to import needed packages for creating a UI.
        """
        try:
            import wx
            self._wx = wx
            from diplomat.predictors.supervised_fpe.guilib.fpe_editor import FPEEditor
            from diplomat.predictors.supervised_fpe.guilib.progress_dialog import FBProgressDialog
            self._editor_class = _simplify_editor_class(wx, FPEEditor)
            self._progress_dialog_cls = FBProgressDialog

            from diplomat.predictors.supervised_fpe.labelers import Point
            from diplomat.predictors.supervised_fpe.scorers import MaximumJumpInStandardDeviations, EntropyOfTransitions
            from diplomat.predictors.supervised_fpe.guilib.identity_swapper import IdentitySwapper
            self._labeler_class = Point
            self._scorer_classes = [MaximumJumpInStandardDeviations, EntropyOfTransitions]
            self._id_class = IdentitySwapper
        except ImportError:
            raise UIImportError(
                "Unable to load wx UI, make sure wxPython is installed,"
                " or diplomat was installed with optional dependencies enabled."
            )

    def tweak(
        self,
        parent,
        video_path: Union[os.PathLike, str],
        poses: Pose,
        bodypart_names: List[str],
        video_metadata: Dict[str, Any],
        num_outputs: int,
        crop_box: Optional[Tuple[int, int, int, int]],
        on_end: Callable[[bool, Pose], Any],
        make_app: bool = True
    ):
        """
        Load a lighter version of the supervised UI to allow for minor modifications to user saved tracking data.

        :param parent: The parent wx widget of the UI. Can be None, indicating no parent widget, or an independent window.
        :param video_path: The path to the video to display in the editor.
        :param poses: The poses for the given video, contains x and y locations and likelihood values. Must be converted to a Pose object.
        :param bodypart_names: A list of strings, the names for each body part.
        :param video_metadata: Various required video info needed to set up the UI to handle the video and specify appearance settings. See
                               the video_metadata attribute of Predictors to get more information about the required attributes for this dictionary.
        :param num_outputs: An integer, the number of each individual in the tracking data.
        :param crop_box: A tuple of 4 integers (x, y, width, height), specifying the box to crop results to within the video.
        :param on_end: A callable that is executed when the user attempts to save there results or close the window. Two arguments are passed, a
                       boolean specifying if the user wants to save the modified results (True if they do), and a Pose object containing the user
                       modified poses.
        :param make_app: A boolean. If true, this function will create a wx app and run its main loop, meaning execution will pause until the user
                         saves their results or closes the window. If false, no app is made. Defaults to true.
        :return:
        """
        app = self._wx.App() if(make_app) else None

        with self._progress_dialog_cls(parent, title="Progress", inner_msg="Computing Average Standard Deviation") as dialog:
            dialog.Show()
            fake_fpe = _DummyFramePassEngine(
                poses,
                crop_box if(crop_box is not None) else [0, 0, video_metadata["size"][1], video_metadata["size"][0]],
                video_metadata,
                num_outputs,
                dialog.progress_bar
            )

        editor = self._editor_class(
            on_end,
            parent,
            cv2.VideoCapture(str(video_path)),
            poses,
            bodypart_names,
            video_metadata,
            crop_box,
            [self._labeler_class(fake_fpe)],
            [sc(fake_fpe) for sc in self._scorer_classes],
            self._id_class(fake_fpe),
            list(range(1, num_outputs + 1)) * (len(bodypart_names) // num_outputs),
            title="Tweak Tracks"
        )

        editor.Show()

        if(make_app):
            app.MainLoop()

