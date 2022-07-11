from typing import List, Optional, Tuple

from diplomat.predictors.fpe.frame_pass import FramePass, PassOrderError
from diplomat.predictors.fpe.skeleton_structures import StorageGraph
from diplomat.predictors.fpe.sparse_storage import SparseTrackingData, ForwardBackwardFrame, ForwardBackwardData
import numpy as np
from diplomat.processing import ProgressBar, ConfigSpec
import diplomat.processing.type_casters as tc


class FixFrame(FramePass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scores = None
        self._max_frame_idx = None
        self._fixed_frame = None

    @classmethod
    def merge_tracks(cls, track_data: List[SparseTrackingData]) -> SparseTrackingData:
        if(len(track_data) == 0):
            raise ValueError("No arguments passed!")

        new_merged_data = SparseTrackingData()

        if(all([data.probs is None for data in track_data])):
            return new_merged_data

        new_merged_data.pack(
            *(
                np.concatenate([data.unpack()[i] for data in track_data if(data.probs is not None)])
                for i in range(len(track_data[0].unpack()))
            )
        )

        return new_merged_data

    @classmethod
    def get_max_location(
        cls,
        frame: ForwardBackwardFrame,
        down_scaling: int
    ) -> Tuple[Optional[float], Optional[float]]:
        y, x, prob, x_off, y_off = frame.src_data.unpack()

        if(prob is None):
            return (None, None)

        max_idx = np.argmax(prob)

        return (
            x[max_idx] + 0.5 + (x_off[max_idx] / down_scaling),
            y[max_idx] + 0.5 + (y_off[max_idx] / down_scaling)
        )

    @classmethod
    def dist(cls, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    @classmethod
    def create_fix_frame(
        cls,
        fb_data: ForwardBackwardData,
        frame_idx: int,
        skeleton: Optional[StorageGraph] = None
    ) -> List[ForwardBackwardFrame]:
        fixed_frame = [None] * fb_data.num_bodyparts
        num_outputs = fb_data.metadata.num_outputs
        down_scaling = fb_data.metadata.down_scaling

        # Copy over data to start, ignoring skeleton...
        for bp_i in range(fb_data.num_bodyparts):
            fixed_frame[bp_i] = fb_data.frames[frame_idx][bp_i].copy()
            fixed_frame[bp_i].disable_occluded = True

        if(skeleton is not None):
            # For skeletal info, we need to swap order all clusters to get the minimum score with the skeleton...

            # Returns the skeleton score between two body parts, lower is better. (Gets absolute distance from average)
            def score(avg, frame1, frame2):
                return np.abs(avg - cls.dist(
                    cls.get_max_location(frame1, down_scaling),
                    cls.get_max_location(frame2, down_scaling)
                ))

            # Our traversal function, run for each edge in the skeleton graph.
            def on_traversal(dfs_edge, value):
                prior_n, current_n = dfs_edge
                __, __, avg = value

                for offset in range(num_outputs):
                    prior_n_exact = prior_n * num_outputs + offset
                    current_n_exact = current_n * num_outputs + offset

                    current_n_best = current_n * num_outputs + np.argmin([
                            score(
                                avg,
                                fixed_frame[prior_n_exact],
                                fb_data.frames[frame_idx][current_n * num_outputs + i]
                            ) for i in range(num_outputs)
                    ])

                    fixed_frame[current_n_exact] = fb_data.frames[frame_idx][current_n_best].copy()
                    fixed_frame[current_n_exact].disable_occluded = True

            # Run the dfs to find the best indexes for each cluster and rearrange them...
            skeleton.dfs(on_traversal)

        return fixed_frame

    @classmethod
    def compute_scores(
        cls,
        fb_data: ForwardBackwardData,
        prog_bar: ProgressBar,
        reset_bar: bool = False
    ) -> np.ndarray:
        if(not "is_clustered" in fb_data.metadata):
            raise PassOrderError(
                "Clustering must be done before frame fixing!"
            )

        scores = np.zeros(fb_data.num_frames)

        num_outputs = fb_data.metadata.num_outputs
        num_frames = fb_data.num_frames
        down_scaling = fb_data.metadata.down_scaling
        num_bp = fb_data.num_bodyparts // num_outputs

        if(reset_bar and prog_bar is not None):
            prog_bar.reset(fb_data.num_frames)

        for f_idx in range(num_frames):
            score = 0

            for bp_group_off in range(num_bp):

                min_dist = np.inf
                # For body part groupings...
                for i in range(num_outputs - 1):
                    f1_loc = cls.get_max_location(
                        fb_data.frames[f_idx][bp_group_off * num_outputs + i],
                        down_scaling
                    )

                    if (f1_loc[0] is None):
                        min_dist = -np.inf
                        continue

                    for j in range(i + 1, num_outputs):
                        f2_loc = cls.get_max_location(
                            fb_data.frames[f_idx][bp_group_off * num_outputs + j], down_scaling
                        )

                        if (f2_loc[0] is None):
                            min_dist = -np.inf
                            continue

                        min_dist = min(cls.dist(f1_loc, f2_loc), min_dist)

                score += min_dist

            # If skeleton is implemented...
            if ("skeleton" in fb_data.metadata):
                skel = fb_data.metadata.skeleton

                for bp in range(fb_data.num_bodyparts):
                    bp_group_off, bp_off = divmod(bp, num_outputs)

                    num_pairs = num_outputs * len(skel[bp_group_off])
                    f1_loc = cls.get_max_location(
                        fb_data.frames[f_idx][
                            bp_group_off * num_outputs + bp_off], down_scaling
                    )

                    if (f1_loc[0] is None):
                        score -= np.inf
                        continue

                    for (bp2_group_off, (__, __, avg)) in skel[bp_group_off]:
                        min_score = np.inf

                        for bp2_off in range(num_outputs):
                            f2_loc = cls.get_max_location(
                                fb_data.frames[f_idx][
                                    bp2_group_off * num_outputs + bp2_off],
                                down_scaling
                            )

                            if (f2_loc[0] is None):
                                score -= np.inf
                                continue

                            result = np.abs(cls.dist(f1_loc, f2_loc) - avg)
                            min_score = min(result, min_score)

                        score -= (min_score / num_pairs)

            scores[f_idx] = score
            if (prog_bar is not None):
                prog_bar.update(1)

        return scores

    @classmethod
    def restore_all_except_fix_frame(
        cls,
        fb_data: ForwardBackwardData,
        frame_idx: int,
        fix_frame_data: List[ForwardBackwardFrame],
        prog_bar: ProgressBar,
        reset_bar: bool = False
    ) -> ForwardBackwardData:
        # For passes to use....
        fb_data.metadata.fixed_frame_index = int(frame_idx)

        if(reset_bar and prog_bar is not None):
            prog_bar.reset(fb_data.num_frames)

        for f_i, frm in enumerate(fb_data.frames):
            for b_i, data in enumerate(frm):
                # If the fixed frame, return the fixed frame...
                if(f_i == frame_idx):
                    fb_data.frames[f_i][b_i] = fix_frame_data[b_i]
                # If any other frame, return the frame as the merged clusters...
                else:
                    fb_data.frames[f_i][b_i].src_data = fb_data.frames[f_i][b_i].orig_data

        return fb_data

    def run_pass(
        self,
        fb_data: ForwardBackwardData,
        prog_bar: Optional[ProgressBar] = None,
        in_place: bool = True,
        reset_bar: bool = True,
        run_main_pass: bool = True
    ) -> ForwardBackwardData:
        if(reset_bar and prog_bar is not None):
            prog_bar.reset(fb_data.num_frames * 2)

        self._scores = self.compute_scores(fb_data, prog_bar, False)

        self._max_frame_idx = int(np.argmax(self._scores))

        if(self.config.fix_frame_override is not None):
            if(not (0 <= self.config.fix_frame_override < len(self._scores))):
                raise ValueError("Override Fix Frame Value is not valid!")
            self._max_frame_idx = self.config.fix_frame_override

        if(self.config.DEBUG):
            print(f"Max Scoring Frame: {self._max_frame_idx}")

        self._fixed_frame = self.create_fix_frame(
            fb_data,
            self._max_frame_idx,
            fb_data.metadata.skeleton if("skeleton" in fb_data.metadata) else None
        )

        # Now the pass...
        if(run_main_pass):
            return self.restore_all_except_fix_frame(
                fb_data,
                self._max_frame_idx,
                self._fixed_frame,
                prog_bar,
                False
            )
        else:
            return fb_data

    @classmethod
    def get_config_options(cls) -> ConfigSpec:
        return {
            "DEBUG": (False, bool, "Set to True to dump additional information while the pass is running."),
            "fix_frame_override": (
                None,
                tc.Union(tc.Literal(None), tc.RangedInteger(0, np.inf)),
                "Specify the fixed frame manually by setting to an integer index."
            )
        }