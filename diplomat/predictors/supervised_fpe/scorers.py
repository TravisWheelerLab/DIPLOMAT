from typing import Optional
import numpy as np
from typing_extensions import Protocol
from diplomat.predictors.fpe import fpe_math
from diplomat.predictors.fpe.sparse_storage import ForwardBackwardFrame, ForwardBackwardData
from diplomat.wx_gui import labeler_lib
from diplomat.wx_gui.score_lib import ScoreEngine
from diplomat.processing import *
import warnings


class ScoreAbleFramePassEngine(Protocol):
    @property
    def frame_data(self) -> ForwardBackwardData:
        raise NotImplementedError

    @property
    def video_metadata(self) -> Config:
        raise NotImplementedError

    @property
    def width(self) -> int:
        raise NotImplementedError

    @property
    def height(self) -> int:
        raise NotImplementedError


def normalized_shanon_entropy(dists: np.ndarray):
    # Entropy in bits / Entropy in bits of uniform distribution of size n...
    dists = dists / np.sum(dists, -1)
    dists[dists == 0] = 1
    return (-np.sum(dists * np.log2(dists), -1)) / np.log2(dists.shape[-1])


class EntropyOfTransitions(ScoreEngine):
    def __init__(self, frame_engine: ScoreAbleFramePassEngine):
        super().__init__()
        self._frame_engine = frame_engine

        self._gaussian_table = None
        self._std = self._get_std(self._frame_engine.frame_data.metadata)
        self._init_gaussian_table(
            int(frame_engine.width) + 1,
            int(frame_engine.height) + 1
        )

        self._settings = labeler_lib.SettingCollection(
            threshold=labeler_lib.FloatSpin(0, 1, 0.8, 0.05, 4)
        )

    def _get_std(self, metadata):
        if("optimal_std" in metadata):
            return metadata.optimal_std[2]
        else:
            return 1

    def _init_gaussian_table(self, width, height):
        if(self._gaussian_table is None):
            self._gaussian_table = fpe_math.gaussian_table(
                height, width, self._std, 1, 0
            )

    def compute_scores(self, poses: Pose, prog_bar: ProgressBar, sub_section: Optional[slice] = None) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            frames = self._frame_engine.frame_data

            if(sub_section is None):
                sub_section = slice(None)

            s, e, j = sub_section.indices(frames.num_frames)
            sub_section = range(s + 1, e, j)

            scores = np.zeros(len(sub_section) + 2, dtype=np.float32)
            num_in_group = frames.metadata.num_outputs
            num_groups = poses.get_bodypart_count() // num_in_group

            for f_i in sub_section:
                for b_g_i in range(num_groups):
                    cxs = poses.get_x_at(f_i, slice(b_g_i * num_in_group, (b_g_i + 1) * num_in_group))
                    cys = poses.get_y_at(f_i, slice(b_g_i * num_in_group, (b_g_i + 1) * num_in_group))
                    cprobs = poses.get_prob_at(f_i, slice(b_g_i * num_in_group, (b_g_i + 1) * num_in_group))
                    pxs = poses.get_x_at(f_i, slice(b_g_i * num_in_group, (b_g_i + 1) * num_in_group))
                    pys = poses.get_y_at(f_i, slice(b_g_i * num_in_group, (b_g_i + 1) * num_in_group))
                    pprobs = poses.get_prob_at(f_i, slice(b_g_i * num_in_group, (b_g_i + 1) * num_in_group))

                    matrix = np.expand_dims(cprobs, 1) * fpe_math.table_transition_interpolate(
                        (pxs, pys), 1, (cxs, cys), 1, self._gaussian_table, 1
                    ) * np.expand_dims(pprobs, 0)

                    k = np.nanmax(normalized_shanon_entropy(matrix))
                    scores[f_i - sub_section.start + 1] = max(scores[f_i - sub_section.start + 1], k)

                prog_bar.update()

            scores[0] = scores[1]
            scores[-1] = scores[-2]
            scores = (scores[:-1] + scores[1:]) / 2

            return scores

    def compute_bad_indexes(self, scores: np.ndarray) -> np.ndarray:
        threshold = self._settings.get_values().threshold
        return np.flatnonzero(scores > threshold)

    def get_settings(self) -> labeler_lib.SettingCollection:
        return self._settings

    def get_name(self) -> str:
        return "Maximum Entropy of Transitions"


class MaximumJumpInStandardDeviations(ScoreEngine):
    def __init__(self, frame_engine: ScoreAbleFramePassEngine):
        self._frame_engine = frame_engine
        self._std = self._get_std(self._frame_engine.frame_data.metadata)
        self._pcutoff = self._frame_engine.video_metadata["pcutoff"]

        self._settings = labeler_lib.SettingCollection(
            threshold = labeler_lib.FloatSpin(0.25, 1000, 4, 0.25, 4)
        )

    def _get_std(self, metadata):
        if ("optimal_std" in metadata):
            return metadata.optimal_std[2]
        else:
            return 1

    def compute_scores(self, poses: Pose, prog_bar: ProgressBar, sub_section: Optional[slice] = None) -> np.ndarray:
        if(sub_section is None):
            sub_section = slice(None)

        s, e, j = sub_section.indices(poses.get_frame_count())
        sub_section = range(s + 1, e, j)

        scores = np.zeros(len(sub_section) + 1, dtype=np.float32)

        for f_i in sub_section:
            prior_x = poses.get_x_at(f_i - 1, slice(None))
            prior_y = poses.get_y_at(f_i - 1, slice(None))
            prior_p = poses.get_prob_at(f_i - 1, slice(None))
            c_x = poses.get_x_at(f_i, slice(None))
            c_y = poses.get_y_at(f_i, slice(None))
            c_p = poses.get_prob_at(f_i, slice(None))

            dists = np.sqrt((c_x - prior_x) ** 2 + (c_y - prior_y) ** 2)
            dists /= self._std
            dists[(prior_p < self._pcutoff) | (c_p < self._pcutoff)] = 0

            scores[f_i - sub_section.start + 1] = np.max(dists)

            prog_bar.update()

        return scores

    def compute_bad_indexes(self, scores: np.ndarray) -> np.ndarray:
        threshold = self._settings.get_values().threshold
        return np.flatnonzero(scores > threshold)

    def get_settings(self) -> labeler_lib.SettingCollection:
        return self._settings

    def get_name(self) -> str:
        return "Maximum Jump in Standard Deviations"
