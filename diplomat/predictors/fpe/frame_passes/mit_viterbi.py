from typing import Optional, Tuple, Callable, Iterable, Sequence, List, Union, TypeVar
import numpy as np
from diplomat.predictors.fpe.frame_pass import FramePass, PassOrderError, RangeSlicer, ConfigSpec
from diplomat.predictors.fpe.sparse_storage import ForwardBackwardFrame, AttributeDict, ForwardBackwardData, SparseTrackingData
from diplomat.predictors.fpe.fpe_math import TransitionFunction, Probs, Coords
from diplomat.processing import ProgressBar
from diplomat.predictors.fpe import fpe_math
from diplomat.predictors.fpe.skeleton_structures import StorageGraph
from diplomat.predictors.fpe.frame_pass import type_casters as tc
from diplomat.predictors.fpe import arr_utils
import warnings


# Used when the body part count is <= 2, or multi-threading is disabled...
class NotAPool:
    T = TypeVar("T")
    E = TypeVar("E")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def starmap(self, func: Callable[[T], E], iterable: Iterable[T]) -> Iterable[E]:
        return (func(*item) for item in iterable)

to_log_space = np.log2
from_log_space = np.exp2
norm = lambda arr: arr - np.max(arr)

NumericArray = Union[int, float, np.ndarray]

def norm_together(arrs):
    max_val = max(np.max(arr) for arr in arrs)
    return [arr - max_val for arr in arrs]

def select(*info):
    for val, cond in zip(info[::2, 1::2]):
        if(cond):
            return val

    return info[::2][-1]


class ViterbiTransitionTable:
    def __init__(
        self,
        table: np.ndarray,
        enter_exit_prob: float,
        enter_stay_prob: float
    ):
        self._table = table
        self._enter_exit_prob = to_log_space(enter_exit_prob)
        self._enter_stay_prob = to_log_space(enter_stay_prob)

    @staticmethod
    def _is_enter_state(coords: Coords) -> bool:
        return len(coords[0]) == 1 and np.isneginf(coords[0][0])

    def __call__(
        self,
        prior_probs: Probs,
        prior_coords: Coords,
        current_probs: Probs,
        current_coords: Coords
    ) -> np.ndarray:
        if(self._is_enter_state(prior_coords)):
            return np.full(
                (len(current_probs), len(prior_probs)),
                self._enter_stay_prob if(self._is_enter_state(current_coords)) else self._enter_exit_prob,
                np.float32
            )
        elif(self._is_enter_state(current_coords)):
            return np.full((len(current_probs), len(prior_probs)), -np.inf, np.float32)

        return fpe_math.table_transition(
            prior_coords, current_coords, self._table
        )


class MITViterbi(FramePass):
    """
    An implementation of the Multi-Individual Tracking Viterbi algorithm. Runs a viterbi-like algorithm across the frames to determine the
    maximum scoring paths per individual, assuming an individuals can't take a paths that would have been more likely for other individuals
    to have taken.
    """

    ND_UNIT_PER_SIDE_COUNT = 10
    # Hidden attribute used for checking if this plugin class uses a pool...
    UTILIZE_GLOBAL_POOL = True

    def __init__(self, width, height, *args, **kwargs):
        super().__init__(width, height, *args, **kwargs)
        self._scaled_std = None
        self._flatten_std = None
        self._gaussian_table = None
        self._skeleton_tables = None

    def _init_gaussian_table(self, metadata: AttributeDict):
        """
        Initialize the standard deviation and gaussian table.

        :param metadata: The metadata from the ForwardBackwardData object.
                         Provides the optimized standard deviation if the
                         standard_deviation is set to 'auto'.
        """
        conf = self.config
        std = conf.standard_deviation

        if (std == "auto" and ("optimal_std" in metadata)):
            self._scaled_std = metadata.optimal_std[2]
        else:
            self._scaled_std = (std if (std != "auto") else 1) / metadata.down_scaling
        self._flatten_std = None if (conf.gaussian_plateau is None) else self._scaled_std * conf.gaussian_plateau
        self._gaussian_table = norm(to_log_space(fpe_math.gaussian_table(
            self.height, self.width, self._scaled_std, conf.amplitude,
            conf.lowest_value, self._flatten_std, conf.square_distances
        )))

    def _init_skeleton(self, data: ForwardBackwardData):
        if("skeleton" in data.metadata):
            meta = data.metadata
            self._skeleton_tables = StorageGraph(meta.skeleton.node_names())

            for ((n1, n2), (bin_val, freq, avg)) in meta.skeleton.items():
                fill_func = lambda x, y: fpe_math.skeleton_formula(
                    x, y, avg, **meta.skeleton_config
                )

                self._skeleton_tables[n1, n2] = norm(
                    to_log_space(
                        fpe_math.get_func_table(
                            self.height, self.width, fill_func
                        )
                    )
                )
        else:
            self.config.skeleton_weight = 0

    def _init_obscured_state(self, metadata: AttributeDict):
        metadata.obscured_prob = self.config.obscured_probability
        metadata.obscured_survival_max = self.config.obscured_survival_max

    def _init_edge_state(self, metadata: AttributeDict):
        metadata.enter_prob = self.config.enter_state_probability
        metadata.enter_trans_prob = self.config.enter_state_exit_probability

    def run_pass(
        self,
        fb_data: ForwardBackwardData,
        prog_bar: Optional[ProgressBar] = None,
        in_place: bool = True,
        reset_bar: bool = True
    ) -> ForwardBackwardData:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if("fixed_frame_index" not in fb_data.metadata):
                raise PassOrderError(f"Must run FixFrame before this pass!")

            self._init_gaussian_table(fb_data.metadata)
            self._init_skeleton(fb_data)
            self._init_obscured_state(fb_data.metadata)
            self._init_edge_state(fb_data.metadata)

            fix_frame = fb_data.metadata.fixed_frame_index

            if(reset_bar and prog_bar is not None):
                prog_bar.reset(fb_data.num_frames * 3)

            # Initialize fixed frame...
            if(not fb_data.metadata.get("is_pre_initialized", False)):
                for bp_i in range(fb_data.num_bodyparts):
                    self._compute_init_frame(
                        fb_data.frames[fb_data.metadata.fixed_frame_index][bp_i],
                        fb_data.metadata
                    )
            else:
                for bp_i in range(fb_data.num_bodyparts):
                    frame = fb_data.frames[fb_data.metadata.fixed_frame_index][bp_i]
                    frame.frame_probs = to_log_space(frame.frame_probs)
                    frame.occluded_probs = to_log_space(frame.occluded_probs)
                    frame.enter_state = to_log_space(frame.enter_state)

            fb_data = fb_data if(in_place) else fb_data.copy()

            # Viterbi
            super()._set_step_controls(fix_frame + 1, None, 1, -1)
            self._run_forward(fb_data, prog_bar, True, False)
            super()._set_step_controls(None, fix_frame, -1, 1)
            self._run_backtrace(fb_data, prog_bar)

            super()._set_step_controls(fix_frame - 1, None, -1, 1)
            self._run_forward(fb_data, prog_bar, True, False)
            super()._set_step_controls(None, fix_frame, 1, -1)
            self._run_backtrace(fb_data, prog_bar)

            for f_i in range(fb_data.num_frames):
                for bp_i in range(fb_data.num_bodyparts):
                    fix_frame = fb_data.frames[f_i][bp_i]
                    fix_frame.frame_probs = from_log_space(fix_frame.frame_probs)
                    fix_frame.occluded_probs = from_log_space(fix_frame.occluded_probs)
                    fix_frame.enter_state = from_log_space(fix_frame.enter_state)
                if(prog_bar is not None):
                    prog_bar.update()

            return fb_data

    def _run_backtrace(
        self,
        fb_data: ForwardBackwardData,
        prog_bar: Optional[ProgressBar] = None,
    ) -> ForwardBackwardData:
        pool_cls = self._get_pool if(self.multi_threading_allowed and (fb_data.num_bodyparts // fb_data.metadata.num_outputs) > 2) else NotAPool

        backtrace_priors = [None for __ in range(fb_data.num_bodyparts)]
        backtrace_current = [None for __ in range(fb_data.num_bodyparts)]

        with pool_cls() as pool:
            for frame_idx in RangeSlicer(fb_data.frames)[self._start:self._stop:self._step]:
                if(not (0 <= (frame_idx + self._prior_off) < len(fb_data.frames))):
                    continue

                # Compute the prior maximum locations for all body parts in
                # the prior frame...
                for bp_idx in range(fb_data.num_bodyparts):
                    prior = fb_data.frames[frame_idx + self._prior_off][bp_idx]
                    current = fb_data.frames[frame_idx][bp_idx]

                    py, px, __, __, __ = prior.src_data.unpack()
                    cy, cx, __, __, __ = current.src_data.unpack()

                    if (py is None):
                        px = py = np.array([0])
                    if (cy is None):
                        cx = cy = np.array([0])

                    # Compute the max of all the priors...
                    combined, combined_coords, source_idxs = arr_utils.pad_coordinates_and_probs(
                        [prior.frame_probs, prior.occluded_probs],
                        [np.array([px, py]).T, prior.occluded_coords],
                        -np.inf
                    )
                    combined = np.asarray(combined)

                    prior_max_idxs = np.argmax(combined, axis=1)
                    max_of_maxes = np.argmax(
                        np.max(combined, axis=1))  # Is it in occluded or frame?

                    # If edge state is higher, select it over the actual frame coordinates...
                    if(combined[max_of_maxes][prior_max_idxs[max_of_maxes]] < prior.enter_state):
                        prob, coords = prior.enter_state, [-np.inf, -np.inf]
                    else:
                        prob = combined[max_of_maxes][prior_max_idxs[max_of_maxes]]
                        coords = combined_coords[max_of_maxes][prior_max_idxs[max_of_maxes]]

                    prior_data = [(
                        np.asarray([prob]),
                        np.asarray([coords]).T
                    )]

                    current_data = [
                        (current.frame_probs, (cx, cy)),
                        (current.occluded_probs, current.occluded_coords.T),
                        (np.array([current.enter_state]), np.array([[-np.inf, -np.inf]]).T)
                    ]

                    backtrace_priors[bp_idx] = prior_data
                    backtrace_current[bp_idx] = current_data

                exit_prob = fb_data.metadata.enter_trans_prob
                # Now run the actual backtrace, computing transitions for
                # prior maximums -> current frames...
                results = pool.starmap(
                    type(self)._compute_backtrace_step,
                    [(
                        backtrace_priors,
                        backtrace_current[bp_i],
                        bp_i,
                        fb_data.metadata,
                        ViterbiTransitionTable(self._gaussian_table, exit_prob, 1 - exit_prob),
                        self._skeleton_tables if (self.config.include_skeleton) else None,
                        self.config.skeleton_weight
                    ) for bp_i in range(fb_data.num_bodyparts)]
                )

                # We stash the entire frame, the maximums are computed on
                # the next step, and by the FramePassEngine at the end...
                for bp_i, (frm_prob, occ_prob, enter_state) in enumerate(results):
                    fb_data.frames[frame_idx][bp_i].frame_probs = frm_prob
                    fb_data.frames[frame_idx][bp_i].occluded_probs = occ_prob
                    fb_data.frames[frame_idx][bp_i].enter_state = enter_state[0]

                if(prog_bar is not None):
                    prog_bar.update()

        return fb_data

    @staticmethod
    def _get_pool():
        # Check globals for a pool...
        if(FramePass.GLOBAL_POOL is not None):
            return FramePass.GLOBAL_POOL

        from multiprocessing import get_context
        for ctx, args in [("forkserver", {}), ("spawn", {}), ("fork", {"maxtasksperchild": 1})]:
            try:
                ctx = get_context(ctx)
                return ctx.Pool(**args)
            except ValueError:
                continue

        return get_context().Pool()

    def _run_forward(
        self,
        fb_data: ForwardBackwardData,
        prog_bar: Optional[ProgressBar] = None,
        in_place: bool = True,
        reset_bar: bool = True
    ) -> ForwardBackwardData:
        frame_iter = RangeSlicer(fb_data.frames)[self._start:self._stop:self._step]
        meta = fb_data.metadata

        fb_data = fb_data if(in_place) else fb_data.copy()

        if(prog_bar is not None and reset_bar):
            prog_bar.reset(total=fb_data.num_frames)

        # We only use a pool if the body part group is high enough...
        pool_cls = self._get_pool if(self.multi_threading_allowed and (fb_data.num_bodyparts // meta.num_outputs) > 2) else NotAPool

        with pool_cls() as pool:
            for i in frame_iter:
                prior_idx = i + self._prior_off

                if (not (0 <= prior_idx < fb_data.num_frames)):
                    continue
                prior = fb_data.frames[prior_idx]
                current = fb_data.frames[i]
                current = current if (in_place) else [c.copy() for c in current]

                exit_prob = fb_data.metadata.enter_trans_prob

                results = pool.starmap(
                    MITViterbi._compute_normal_frame,
                    [(
                        prior,
                        current,
                        bp_grp_i,
                        meta,
                        ViterbiTransitionTable(self._gaussian_table, exit_prob, 1 - exit_prob),
                        self._skeleton_tables if (self.config.include_skeleton) else None,
                        self.config.skeleton_weight
                    ) for bp_grp_i in range(fb_data.num_bodyparts // meta.num_outputs)]
                )

                for (bp_grp_i, res) in enumerate(results):
                    section = slice(bp_grp_i * meta.num_outputs, (bp_grp_i + 1) * meta.num_outputs)
                    current[section] = res[section]

                fb_data.frames[i] = current

                if(prog_bar is not None):
                    prog_bar.update()

        return fb_data

    @classmethod
    def _compute_backtrace_step(
        cls,
        prior: List[List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]],
        current: List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
        bp_idx: int,
        metadata: AttributeDict,
        transition_function: TransitionFunction,
        skeleton_table: Optional[StorageGraph] = None,
        skeleton_weight: float = 0
    ) -> List[np.ndarray]:
        skel_res = cls._compute_from_skeleton(
            prior,
            current,
            bp_idx,
            metadata,
            skeleton_table
        )
        trans_res = cls.log_viterbi_between(
            current,
            prior[bp_idx],
            transition_function
        )

        return norm_together([
            t + s * skeleton_weight for t, s in zip(trans_res, skel_res)
        ])

    @classmethod
    def _compute_init_frame(
        cls,
        frame: ForwardBackwardFrame,
        metadata: AttributeDict
    ) -> ForwardBackwardFrame:
        y, x, probs, x_off, y_off = frame.src_data.unpack()

        if(y is None):
            print("Invalid frame to start on! Using enter state...")
            frame.enter_state = to_log_space(1)
            frame.occluded_probs = to_log_space(np.array([0]))
            frame.occluded_coords = np.array([[0, 0]])

        # Occluded state for first frame...
        occ_coord = np.array([x, y]).T
        occ_probs = np.full(
            occ_coord.shape[0], to_log_space(metadata.obscured_prob)
        )

        # Filter probabilities to limit occluded state...
        occ_coord, occ_probs = cls.filter_occluded_probabilities(
            occ_coord, occ_probs, metadata.obscured_survival_max
        )

        # Store results in current frame (normalized and in log-space)
        frame.occluded_coords = occ_coord
        frame.frame_probs, frame.occluded_probs = norm_together(
            [to_log_space(probs), occ_probs]
        )
        frame.enter_state = -np.inf  # Make enter state 0 in log space...

        return frame

    @classmethod
    def filter_occluded_probabilities(
        cls,
        occluded_coords: np.ndarray,
        occluded_probs: np.ndarray,
        max_count: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter occluded coordinates and probabilities such that there is only max_count of them left, those with the
        highest scoring probabilities.

        :param occluded_coords: The occluded coordinates, Nx2 numpy array.
        :param occluded_probs: The occluded probabilities, N numpy array.
        :param max_count: The max number of occluded locations to keep.

        :returns: A tuple containing the updated coordinates and probabilities.
        """
        if (len(occluded_probs) <= max_count):
            return (occluded_coords, occluded_probs)

        indexes = np.argpartition(occluded_probs, -max_count)[-max_count:]

        return (occluded_coords[indexes], occluded_probs[indexes])

    @classmethod
    def _compute_from_skeleton(
        cls,
        prior: Union[List[ForwardBackwardFrame], List[List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]]],
        current_data: List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
        bp_idx: int,
        metadata: AttributeDict,
        skeleton_table: Optional[StorageGraph] = None,
        merge_arrays: Callable[[Iterable[np.ndarray]], np.ndarray] = np.maximum.reduce,
        merge_internal: Callable[[np.ndarray, int], np.ndarray] = np.max,
        merge_results: bool = True
    ) -> Union[List[Tuple[int, List[NumericArray]]], List[NumericArray]]:
        if(skeleton_table is None):
            return [0] * len(current_data) if(merge_results) else []

        bp_group_idx = bp_idx // metadata.num_outputs
        bp_off = bp_idx % metadata.num_outputs

        results = []
        final_result = [0] * len(current_data)

        for other_bp_group_idx, trans_table in skeleton_table[bp_group_idx]:
            # Grab the prior frame...
            prior_frame = prior[other_bp_group_idx * metadata.num_outputs + bp_off]
            if(isinstance(prior_frame, ForwardBackwardFrame)):
                py, px, __, __, __ = prior_frame.src_data.unpack()

                # If wasn't found, don't include in the result.
                if(prior_frame.frame_probs is None):
                    continue

                z_a = lambda: np.array([0])
                nf_a = lambda: np.array([-np.inf])

                prior_data = [
                    (prior_frame.frame_probs, (px, py) if(py is not None) else (z_a(), z_a())),
                    (prior_frame.occluded_probs, prior_frame.occluded_coords.T),
                    (np.array([prior_frame.enter_state]), (nf_a(), nf_a()))
                ]
            else:
                prior_data = prior_frame

            # No skeleton penalty for transitioning from
            transition_func = ViterbiTransitionTable(trans_table, 0.5, 0.5)

            results.append((
                other_bp_group_idx * metadata.num_outputs + bp_off,
                cls.log_viterbi_between(
                    current_data,
                    prior_data,
                    transition_func,
                    merge_arrays,
                    merge_internal
                )
            ))

        if(not merge_results):
            return results

        for __, bp_res in results:
            for i, (current_total, bp_sub_res) in enumerate(zip(final_result, bp_res)):
                merged_result = current_total + bp_sub_res
                if(np.all(np.isneginf(merged_result))):
                    continue
                final_result[i] = merged_result

            final_result = norm_together(final_result)

        return final_result

    @classmethod
    def _compute_normal_frame(
        cls,
        prior: List[ForwardBackwardFrame],
        current: List[ForwardBackwardFrame],
        bp_group: int,
        metadata: AttributeDict,
        transition_function: TransitionFunction,
        skeleton_table: Optional[StorageGraph] = None,
        skeleton_weight: float = 0
    ) -> List[ForwardBackwardFrame]:
        group_range = range(
            bp_group * metadata.num_outputs,
            (bp_group + 1) * metadata.num_outputs
        )

        results = []
        result_coords = []

        # Looks like normal viterbi until domination step...
        for bp_i in group_range:
            py, px, pprob, p_occx, p_occy = prior[bp_i].src_data.unpack()
            cy, cx, cprob, c_occx, c_occy = current[bp_i].src_data.unpack()

            if((cprob is not None) and np.all(cprob <= 0)):
                current[bp_i].src_data = SparseTrackingData()
                cy = cx = cprob = None

            z_arr = lambda: np.array([0])
            neg_inf_arr = lambda: np.array([-np.inf])

            prior_data = [
                (
                    prior[bp_i].frame_probs if(py is not None) else neg_inf_arr(),
                    (px, py) if(py is not None) else (z_arr(), z_arr())
                ),
                (
                    prior[bp_i].occluded_probs,
                    prior[bp_i].occluded_coords.T
                ),
                (
                    np.array([prior[bp_i].enter_state]),
                    (neg_inf_arr(), neg_inf_arr())
                )
            ]

            c_frame_data = (
                (norm(to_log_space(cprob)), (cx, cy))
                if(cy is not None) else (to_log_space(z_arr()), (z_arr(), z_arr()))
            )

            c_occ_coords, c_occ_probs = cls.generate_occluded(
                np.asarray(c_frame_data[1]).T,
                prior[bp_i].occluded_coords,
                metadata.obscured_prob,
                current[bp_i].disable_occluded and (cy is not None)
            )

            current[bp_i].occluded_coords = c_occ_coords

            current_data = [
                c_frame_data,
                (c_occ_probs, c_occ_coords.T),
                (np.array([to_log_space(metadata.enter_prob)]), (neg_inf_arr(), neg_inf_arr()))
            ]

            from_skel = cls._compute_from_skeleton(
                prior,
                current_data,
                bp_i,
                metadata,
                skeleton_table
            )

            from_transition = cls.log_viterbi_between(
                current_data,
                prior_data,
                transition_function
            )

            results.append([
                t + s * skeleton_weight
                for t, s in zip(from_transition, from_skel)
            ])

            # Result coordinates, exclude the enter state...
            result_coords.append([c[1] for c in current_data[:-1]])

        # Pad all arrays so we can do element-wise comparisons between them...
        frm_probs, frm_coords, frm_idxs = arr_utils.pad_coordinates_and_probs(
            [r[0] for r in results],
            [np.asarray(r[0]).T for r in result_coords],
            -np.inf
        )
        occ_probs, occ_coords, occ_idxs = arr_utils.pad_coordinates_and_probs(
            [r[1] for r in results],
            [np.asarray(r[1]).T for r in result_coords],
            -np.inf
        )

        enter_probs = [r[2][0] for r in results]

        # Compute the dominators (max transitions across individuals)...
        frame_dominators = np.maximum.reduce(frm_probs)
        occ_dominators = np.maximum.reduce(occ_probs)

        # Domination & store step...
        for bp_i, frm_prob, occ_prob, frm_idx, occ_idx, enter_prob in zip(
            group_range, frm_probs, occ_probs, frm_idxs, occ_idxs, enter_probs
        ):
            # Set locations which are not dominators for this identity to 0 in log space (not valid transitions)...
            frm_prob[frm_prob < frame_dominators] = -np.inf

            if(not np.all(occ_prob < occ_dominators)):
                occ_prob[occ_prob < occ_dominators] = -np.inf
            else:
                # Bad domination step, lost all occluded and in-frame probabilities, so keep the best location...
                best_occ = np.argmax(occ_prob)
                occ_prob[occ_prob < occ_dominators] = -np.inf
                occ_prob[best_occ] = 0  # 1 in log space...
                occ_dominators[best_occ] = 0  # Don't allow anyone else to take this spot.

            norm_val = np.nanmax([np.nanmax(frm_prob), np.nanmax(occ_prob), enter_prob])

            # Store the results...
            current[bp_i].frame_probs = frm_prob[frm_idx] - norm_val
            # Filter occluded probabilities...
            c, p = cls.filter_occluded_probabilities(
                current[bp_i].occluded_coords,
                occ_prob[occ_idx],
                metadata.obscured_survival_max
            )
            current[bp_i].occluded_coords = c
            current[bp_i].occluded_probs = p - norm_val
            current[bp_i].enter_state = enter_prob - norm_val

        return current

    @classmethod
    def log_viterbi_between(
        cls,
        current_data: Sequence[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
        prior_data: Sequence[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
        transition_function: TransitionFunction,
        merge_arrays: Callable[[Iterable[np.ndarray]], np.ndarray] = np.maximum.reduce,
        merge_internal: Callable[[np.ndarray, int], np.ndarray] = np.nanmax
    ) -> List[np.ndarray]:
        return [
            merge_arrays([
                merge_internal(
                    np.expand_dims(cprob, 1)
                    + transition_function(pprob, pcoord, cprob, ccoord)
                    + np.expand_dims(pprob, 0)
                    , 1
                )
                for pprob, pcoord in prior_data
            ])
            for (cprob, ccoord) in current_data
        ]

    @classmethod
    def generate_occluded(
        cls,
        current_frame_coords: np.ndarray,
        prior_occluded_coords: np.ndarray,
        occluded_prob: float,
        disable_occluded: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        new_coords = np.unique(
            np.concatenate((current_frame_coords, prior_occluded_coords)),
            axis=0
        )

        return (
            new_coords,
            np.full(
                new_coords.shape[0],
                to_log_space(0 if(disable_occluded) else occluded_prob)
            )
        )

    @classmethod
    def get_config_options(cls) -> ConfigSpec:
        # Class to enforce that probabilities are between 0 and 1....
        return {
            "standard_deviation": (
                "auto", tc.Union(float, tc.Literal("auto")),
                "The standard deviation of the 2D Gaussian curve used for "
                "transition probabilities. Defaults to 'auto', which attempts"
                " to use an optimized value if one has been computed by a "
                "prior pass, and otherwise uses the default value of 1..."
            ),
            "skeleton_weight": (
                1, float,
                "A positive float, determines how much impact probabilities "
                "from skeletal transitions should have in each "
                "forward/backward step if a skeleton was created and enabled "
                "by prior passes... This is not a probability, but rather a "
                "ratio."
            ),
            "amplitude": (
                1, float,
                "The max amplitude of the 2D Gaussian curve used for "
                "transition probabilities."
            ),
            "lowest_value": (
                0, float,
                "The lowest value the 2D Gaussian curve used for transition "
                "probabilities can reach."
            ),
            "obscured_probability": (
                0.000001, tc.RangedFloat(0, 1),
                "A constant float between 0 and 1 that determines the "
                "prior probability of being in any hidden state cell."
            ),
            "enter_state_probability": (
                1e-9, tc.RangedFloat(0, 1),
                "A constant, the probability of being in the enter state."
            ),
            "enter_state_exit_probability": (
                0.99, tc.RangedFloat(0, 1),
                "A constant, the probability of exiting the enter state. Probability of staying in the "
                "enter state is this value subtracted from 1."
            ),
            "obscured_survival_max": (
                50, int,
                "An integer, the max number of points to allow to survive for each frame, "
                "if there is more than this value, the top ones are kept."
            ),
            "gaussian_plateau": (
                None, tc.Union(float, tc.Literal(None)),
                "A float specifying the area over which to flatten the "
                "gaussian curve should be less than the norm_dist value. If "
                "none, set to the norm_dist."
            ),
            "include_skeleton": (
                True, bool,
                "A boolean. If True, include skeleton information in the "
                "forward backward pass, otherwise don't. If no skeleton has "
                "been built in a prior pass, does nothing."
            ),
            "square_distances": (
                False, bool,
                "A boolean. If True, square distances between points before "
                "putting them through the gaussian, otherwise don't."
            )
        }