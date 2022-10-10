import os
from typing import List, Any, Dict, Optional, Tuple, MutableMapping, Sequence, Union
import cv2
from diplomat.predictors.sfpe.segmented_frame_pass_engine import SegmentedFramePassEngine
from diplomat.predictors.fpe.sparse_storage import ForwardBackwardData, ForwardBackwardFrame, SparseTrackingData
from diplomat.processing import Pose, Config


class UIImportError(ImportError):
    pass

class _DummySubPoseList(Sequence[ForwardBackwardData]):
    def __init__(self, sub_index: Union[int, slice], poses: Pose):
        self._sub_index = sub_index
        self._poses = poses

    def __getitem__(self, index: Union[int, slice]) -> ForwardBackwardData:
        x = self._poses.get_x_at(self._sub_index, index)
        y = self._poses.get_y_at(self._sub_index, index)
        prob = self._poses.get_prob_at(self._sub_index, index)

        res = SparseTrackingData.pack()



    def __len__(self) -> int:
        return self._poses.get_bodypart_count()


class _DummyPoseList(Sequence[_DummySubPoseList]):
    def __init__(self, poses: Pose):
        self._poses = poses

    def __getitem__(self, index: Union[int, slice]) -> _DummySubPoseList:
        return _DummySubPoseList(index, self._poses)

    def __len__(self) -> int:
        return self._poses.get_frame_count()


class _DummyForwardBackwardData(ForwardBackwardData):
    def __init__(self, poses: Pose, crop_box: Tuple[int, int, int, int]):
        super().__init__(0, 0)
        self._frames = _DummyPoseList(poses)

        self._num_bps



class _DummyFramePassEngine:
    def __init__(self, poses: Pose, crop_box: Tuple[int, int, int, int], video_meta: Dict[str, Any]):
        self._poses = poses
        self._crop_box = crop_box
        self._changed_frames = {}
        self._video_meta = Config(video_meta)

    @property
    def frame_data(self) -> ForwardBackwardData:
        pass

    @property
    def video_metadata(self) -> Config:
        return self._video_meta

    @property
    def width(self) -> int:
        return self._crop_box[2]

    @property
    def height(self) -> int:
        return self._crop_box[3]

    @property
    def changed_frames(self) -> MutableMapping[Tuple[int, int], ForwardBackwardFrame]:
        return self._changed_frames

    def video_to_scmap_coord(self, coord: Tuple[float, float, float]) -> Tuple[int, int, float, float, float]:
        vid_x, vid_y, prob = coord
        x, off_x = divmod(vid_x, 1)
        y, off_y = divmod(vid_y, 1)
        # Correct offsets to be relative to the center of the stride block...
        off_x = off_x - 0.5
        off_y = off_y - 0.5

        return (int(x), int(y), off_x, off_y, prob)

    def scmap_to_video_coord(
        self,
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

    def get_maximum_with_defaults(self, frame: ForwardBackwardFrame) -> Tuple[int, int, float, float, float]:
        return SegmentedFramePassEngine.get_maximum(frame, 0)


class TweakUI:
    def __init__(self):
        try:
            import wx
            self._wx = wx
            from diplomat.predictors.supervised_fpe.guilib.fpe_editor import FPEEditor
            self._editor_class = FPEEditor
            from diplomat.predictors.supervised_fpe.labelers import Point
            from diplomat.predictors.supervised_fpe.scorers import MaximumJumpInStandardDeviations, EntropyOfTransitions
            self._labeler_class = Point
            self._scorer_classes = [MaximumJumpInStandardDeviations, EntropyOfTransitions]
        except ImportError:
            raise UIImportError(
                "Unable to load wx UI, make sure wxPython is installed,"
                " or diplomat was installed with optional dependencies enabled."
            )


    def tweak(
        self,
        video_path: os.PathLike,
        poses: Pose,
        bodypart_names: List[str],
        plot_settings: Dict[str, Any],
        crop_box: Optional[Tuple[int, int, int, int]]
    ) -> (bool, Pose):
        fake_fpe = _DummyFramePassEngine(poses, crop_box)

        editor = self._editor_class(
            None,
            cv2.VideoCapture(str(video_path)),
            poses,
            bodypart_names,
            plot_settings,
            crop_box,
            [self._labeler_class(fake_fpe)],
            [sc(fake_fpe) for sc in self._scorer_classes],
            title="Tweak User Interface"
        )