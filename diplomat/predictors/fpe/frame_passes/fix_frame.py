import heapq
from typing import List, Optional, Tuple, Dict
from diplomat.predictors.fpe.frame_pass import FramePass, PassOrderError
from diplomat.predictors.fpe.skeleton_structures import StorageGraph
from diplomat.predictors.fpe.sparse_storage import SparseTrackingData, ForwardBackwardFrame, ForwardBackwardData
from diplomat.utils.graph_ops import min_cost_matching
import numpy as np
from diplomat.processing import ProgressBar, ConfigSpec
import diplomat.processing.type_casters as tc


class FixFrame(FramePass):
    """
    Scores frames by peak separation, and then selects a single frame to remain clustered (with peaks separated). The
    rest of the frames are restored, and :py:plugin:`~diplomat.predictors.frame_passes.MITViterbi` uses the fixed
    frame as it's ground truth frame.
    """
    SCORES_PER_CHUNK = 20

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
        down_scaling: float
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Determines the maximum location within a frame after downscaling.

        This method identifies the maximum probability location within a given frame and adjusts its coordinates
        based on the provided downscaling factor. It returns the adjusted x and y coordinates along with the
        maximum probability value.

        Parameters:
        - frame: ForwardBackwardFrame, the frame to analyze.
        - down_scaling: float, the factor by which the frame's coordinates are downscaled.

        Returns:
        - Tuple[Optional[float], Optional[float], Optional[float]]: The adjusted x and y coordinates of the maximum
          location and its probability. Returns None for each value if the probability data is not available.
        """
        y, x, prob, x_off, y_off = frame.src_data.unpack()

        if(prob is None):
            return (None, None, None)

        max_idx = np.argmax(prob)

        return (
            x[max_idx] + 0.5 + (x_off[max_idx] / down_scaling),
            y[max_idx] + 0.5 + (y_off[max_idx] / down_scaling),
            prob[max_idx]
        )

    @classmethod
    def dist(cls, a, b):
        return np.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))

    @classmethod
    def skel_score(
        cls,
        avg: float,
        frame1: ForwardBackwardFrame,
        frame2: ForwardBackwardFrame,
        down_scaling: float
    ) -> float:
        return np.abs((avg - cls.dist(
            cls.get_max_location(frame1, down_scaling),
            cls.get_max_location(frame2, down_scaling)
        )) ** 2)

    @classmethod
    def get_bidirectional_score_graphs(
        cls,
        storage_graph: StorageGraph,
        frame_list: List[ForwardBackwardFrame],
        num_outputs: int,
        down_scaling: float,
        fixed_group: int = 0
    ) -> List[List[Dict[int, float]]]:
        # Construct a graph...
        graphs = []

        for i in range(num_outputs):
            graph = [{} for __ in range(len(storage_graph) * num_outputs)]

            # Now traverse the storage graph making connection between things...
            for node_group1 in range(len(storage_graph)):
                for node_group2, (__, __, avg) in storage_graph[node_group1]:
                    for node_off1 in range(num_outputs):
                        if(node_group1 == fixed_group and node_off1 != i):
                            continue

                        idx1 = node_group1 * num_outputs + node_off1

                        for node_off2 in range(num_outputs):
                            if(node_group2 == fixed_group and node_off2 != i):
                                continue

                            idx2 = node_group2 * num_outputs + node_off2

                            graph[idx1][idx2] = cls.skel_score(
                                avg, frame_list[idx1], frame_list[idx2], down_scaling
                            )

            graphs.append(graph)

        return graphs

    @classmethod
    def get_fixed_group(
        cls,
        storage_graph: StorageGraph,
        frame_list: List[ForwardBackwardFrame],
        num_outputs: int,
        down_scaling: float
    ) -> int:
        degrees = [len(storage_graph[i]) for i in range(len(storage_graph))]

        num_groups = len(frame_list) // num_outputs
        group_dist_scores = np.full(num_groups, np.inf, np.float32)

        for g_i in range(num_groups):
            for i_out in range(num_outputs):
                for j_out in range(i_out + 1, num_outputs):
                    m1 = cls.get_max_location(frame_list[g_i * num_outputs + i_out], down_scaling)
                    m2 = cls.get_max_location(frame_list[g_i * num_outputs + j_out], down_scaling)
                    if(m1[0] is None or m2[0] is None):
                        group_dist_scores[g_i] = -np.inf
                    else:
                        group_dist_scores[g_i] = min(cls.dist(m1, m2) * m1[2] * m2[2], group_dist_scores[g_i])

        best = (0, 0)
        best_i = 0

        for i, new in enumerate(zip(degrees, group_dist_scores)):
            # We don't want to allow a high degree node to be used if it's distance is 0 for some of the nodes...
            if(new[1] > 0 and new > best):
                best = new
                best_i = i

        return best_i

    @classmethod
    def _shortest_paths(
        cls,
        score_graph: List[Dict[int, float]],
        start_node: int
    ) -> List[float]:
        visited = [False] * len(score_graph)
        node_scores = [np.inf] * len(score_graph)

        next_nodes = [(0.0, start_node)]
        visited[start_node] = True
        node_scores[start_node] = 0

        while(len(next_nodes) > 0):
            __, node = heapq.heappop(next_nodes)

            for sub_node in score_graph[node]:
                proposed_score = node_scores[node] + score_graph[node][sub_node]

                node_scores[sub_node] = min(node_scores[sub_node], proposed_score)

                if(not visited[sub_node]):
                    visited[sub_node] = True
                    heapq.heappush(next_nodes, (node_scores[sub_node], sub_node))

        return node_scores

    @classmethod
    def _best_skeleton_scores(
        cls,
        shortest_paths: List[List[float]],
        score_graphs: List[List[Dict[int, float]]],
        skeleton: StorageGraph,
        num_outputs: int
    ) -> List[List[float]]:
        best_skeleton_scores = []

        # Now compute skeleton scores....
        for score_graph, best_path_scores in zip(score_graphs, shortest_paths):
            best_skel_scores = [0] * len(best_path_scores)

            for i, score in enumerate(best_path_scores):
                if (np.isinf(score)):
                    best_skel_scores[i] = np.inf
                    continue

                for other_group, __ in skeleton[int(i / num_outputs)]:
                    score = np.inf

                    for other_off in range(num_outputs):
                        other_idx = other_group * num_outputs + other_off
                        score = min(score, best_path_scores[other_idx] + score_graph[i].get(other_idx, np.inf))

                    best_skel_scores[i] += score

            best_skeleton_scores.append(best_skel_scores)

        return best_skeleton_scores

    @classmethod
    def _masked_argmin(cls, arr: np.ndarray, mask: np.ndarray) -> tuple:
        shp = arr.shape
        mask = np.broadcast_to(mask, shp)
        return np.unravel_index(np.flatnonzero(mask)[np.argmin(arr[mask])], shp)

    @classmethod
    def create_fix_frame(
        cls,
        fb_data: ForwardBackwardData,
        frame_idx: int,
        skeleton: Optional[StorageGraph],
        algorithm: str = "greedy"
    ) -> List[ForwardBackwardFrame]:
        #print(f"FixFrame.create_fix_frame.algorithm:={algorithm}")
        if(algorithm not in ("greedy", "hungarian")):
            raise ValueError("Algorithm passed not a support algorithm, use greedy or hungarian.")

        fixed_frame = [None] * fb_data.num_bodyparts
        num_outputs = fb_data.metadata.num_outputs
        down_scaling = fb_data.metadata.down_scaling

        # Copy over data to start, ignoring skeleton...
        for bp_i in range(fb_data.num_bodyparts):
            fixed_frame[bp_i] = fb_data.frames[frame_idx][bp_i].copy()

            __, __, prob, __, __ = fixed_frame[bp_i].src_data.unpack()
            if(prob is None):
                # Fallback fix frame: We just create a single cell with 0 probability, forcing viterbi to use entry
                # states...
                src_data = SparseTrackingData().pack([0], [0], [0], [0], [0])
                fixed_frame[bp_i].src_data = src_data
                fb_data.frames[frame_idx][bp_i].src_data = src_data

            fixed_frame[bp_i].disable_occluded = True

        if(skeleton is not None):
            fixed_group = cls.get_fixed_group(skeleton, fb_data.frames[frame_idx], num_outputs, down_scaling)

            score_graphs = cls.get_bidirectional_score_graphs(
                skeleton,
                fb_data.frames[frame_idx],
                num_outputs,
                down_scaling,
                fixed_group
            )

            if(algorithm == "greedy"):
                select_mask = np.zeros((num_outputs, fb_data.num_bodyparts), bool)
                for __ in range(fb_data.num_bodyparts):
                    # Compute the shortest node paths for every skeleton...
                    skel_scores = np.asarray(cls._best_skeleton_scores(
                        [cls._shortest_paths(score_graph, fixed_group * num_outputs + i)
                         for i, score_graph in enumerate(score_graphs)],
                        score_graphs,
                        skeleton,
                        num_outputs
                    ))

                    # Find the best location...
                    best_body, best_part = cls._masked_argmin(skel_scores, ~select_mask)

                    # Copy the select body part to the correct skeleton.
                    group_start = ((best_part // num_outputs) * num_outputs)
                    select_mask[:, best_part] = True
                    select_mask[best_body, group_start:group_start+num_outputs] = True
                    new_i = group_start + best_body
                    fixed_frame[new_i] = fb_data.frames[frame_idx][best_part]
                    fixed_frame[new_i].disable_occluded = True

                    # Modify graphs based on the selected part...
                    # Construct a zero score link between the newly
                    # added part and the original fixed part of the skeleton.
                    hub_part = fixed_group * num_outputs + best_body
                    score_graphs[best_body][hub_part][best_part] = 0
                    score_graphs[best_body][best_part][hub_part] = 0
                    # Delete all of its edges in other graphs....
                    for i, score_graph in enumerate(score_graphs):
                        if(i == best_body):
                            continue
                        for other_part in list(score_graph[best_part]):
                            del score_graph[other_part][best_part]
                            del score_graph[best_part][other_part]
            else:
                select_mask = np.zeros(fb_data.num_bodyparts // num_outputs, dtype=bool)
                select_mask[fixed_group] = True

                for group_iter in range((fb_data.num_bodyparts // num_outputs) - 1):
                    # Compute the shortest node paths for every skeleton...
                    skel_scores = np.asarray(cls._best_skeleton_scores(
                        [cls._shortest_paths(score_graph, fixed_group * num_outputs + i)
                         for i, score_graph in enumerate(score_graphs)],
                        score_graphs,
                        skeleton,
                        num_outputs
                    ))

                    grouped_skel_scores = skel_scores.reshape((num_outputs, -1, num_outputs))
                    net_part_type_error = np.nanmin(grouped_skel_scores, axis=2).sum(axis=0)

                    min_group = cls._masked_argmin(net_part_type_error, ~select_mask)[0]

                    select_mask[min_group] = True
                    opt_rows, opt_cols = min_cost_matching(
                        grouped_skel_scores[:, min_group, :].reshape(num_outputs, num_outputs)
                    )

                    for row_idx, col_idx in zip(opt_rows, opt_cols):
                        new_i = min_group * num_outputs + row_idx
                        best_part = min_group * num_outputs + col_idx
                        fixed_frame[new_i] = fb_data.frames[frame_idx][best_part]
                        fixed_frame[new_i].disable_occluded = True

                        # Modify graphs based on the selected part...
                        # Construct a zero score link between the newly
                        # added part and the original fixed part of the skeleton.
                        hub_part = fixed_group * num_outputs + row_idx
                        score_graphs[row_idx][hub_part][best_part] = 0
                        score_graphs[row_idx][best_part][hub_part] = 0
                        # Delete all of its edges in other graphs....
                        for i, score_graph in enumerate(score_graphs):
                            if (i == row_idx):
                                continue
                            for other_part in list(score_graph[best_part]):
                                del score_graph[other_part][best_part]
                                del score_graph[best_part][other_part]

        return fixed_frame

    @classmethod
    def compute_single_score(
        cls,
        frames: List[ForwardBackwardFrame],
        num_outputs: int,
        down_scaling: float,
        skeleton: Optional[StorageGraph],
        max_dist: float,
        progress_bar: Optional[ProgressBar] = None
    ) -> Tuple[float, float]:
        """
        Computes a single score for a given set of frames, considering the number of outputs, down scaling factor, 
        an optional skeleton, and the maximum distance. This function aggregates the scores across all body part groupings, 
        taking into account the minimum distance between body parts and their confidence levels. It returns a tuple 
        containing two scores, which represent different aspects of the frame's quality or suitability for further processing.

        Parameters:
        - frames: List[ForwardBackwardFrame], a list of frames to be scored.
        - num_outputs: int, the number of outputs or body parts to consider in each frame.
        - down_scaling: float, the factor by which the frame dimensions have been scaled down.
        - skeleton: Optional[StorageGraph], an optional graph representing the skeleton to be considered in scoring.
        - max_dist: float, the maximum distance to consider when scoring body part pairs.
        - progress_bar: Optional[ProgressBar], an optional progress bar to display processing progress.

        Returns:
        - Tuple[float, float]: A tuple containing two scores calculated based on the minimum distances and confidence levels 
          of body part pairs across the given frames.
        """
        num_bp = len(frames) // num_outputs

        score = 0
        score2 = 0

        for bp_group_off in range(num_bp):

            min_dist = np.inf
            total_conf = 0
            count = 0

            # For body part groupings...
            for i in range(num_outputs - 1):
                #get the maximum probability location for the body part
                f1_loc = cls.get_max_location(
                    frames[bp_group_off * num_outputs + i],
                    down_scaling
                )

                if (f1_loc[0] is None):
                    score = -np.inf
                    continue

                for j in range(i + 1, num_outputs):
                    f2_loc = cls.get_max_location(
                        frames[bp_group_off * num_outputs + j], down_scaling
                    )

                    if (f2_loc[0] is None):
                        score = -np.inf
                        continue
                    
                    #mininum distance between the two body parts
                    min_dist = min(cls.dist(f1_loc, f2_loc), min_dist)
                    total_conf += f1_loc[2] * f2_loc[2]
                    count += 1

            if(np.isinf(min_dist)):
                min_dist = 0
            if(min_dist == 0 or count == 0):
                # BAD! We found a frame that failed to cluster properly...

                #looks like this is the difference between score and score2? 
                score = -np.inf

            # Minimum distance, weighted by average skeleton-pair confidence...
            if(count > 0):
                score += min_dist * (total_conf / count)
                score2 += min_dist * (total_conf / count)
        

        # If skeleton is implemented...
        if (skeleton is not None):
            skel = skeleton

            for bp in range(len(frames)):

                #what is this  ?
                bp_group_off, bp_off = divmod(bp, num_outputs)

                num_pairs = num_outputs * len(skel[bp_group_off])
                f1_loc = cls.get_max_location(
                    frames[bp_group_off * num_outputs + bp_off], down_scaling
                )

                if (f1_loc[0] is None):
                    score = -np.inf
                    score2 -= (max_dist / num_pairs)
                    continue

                for (bp2_group_off, (__, __, avg)) in skel[bp_group_off]:
                    min_score = np.inf

                    for bp2_off in range(num_outputs):
                        f2_loc = cls.get_max_location(
                            frames[bp2_group_off * num_outputs + bp2_off],
                            down_scaling
                        )

                        if(f2_loc[0] is None):
                            score = -np.inf
                            result = max_dist
                        else:
                            result = np.abs(cls.dist(f1_loc, f2_loc) - avg)

                        min_score = min(result, min_score)

                    score -= (min_score / num_pairs)
                    score2 -= (min_score / num_pairs)

        return score, score2

    @classmethod
    def compute_list_of_scores(
        cls,
        frames: List[List[ForwardBackwardFrame]],
        num_outputs: int,
        down_scaling: float,
        skeleton: Optional[StorageGraph],
        max_dist: float,
        progress_bar: Optional[ProgressBar] = None
    ) -> np.ndarray:
        final_scores = np.zeros((len(frames), 2))

        if(progress_bar is not None):
            progress_bar.reset(len(frames))

        for i, frame in enumerate(frames):
            #this will be a tuple of scores per frame 
            final_scores[i] = cls.compute_single_score(frame, num_outputs, down_scaling, skeleton, max_dist)

            if(progress_bar is not None):
                progress_bar.update()

        return final_scores

    @classmethod
    def compute_scores(
        cls,
        fb_data: ForwardBackwardData,
        prog_bar: ProgressBar,
        reset_bar: bool = False,
        thread_count: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the scores for each frame in the ForwardBackwardData object.

        This method calculates two scores for each frame based on the frame data provided. The scores are computed
        by evaluating the frame against a set of criteria defined in the compute_single_score method. The scores
        are intended to be used for determining the quality of the frame data and for further processing steps
        such as segmentation.

        Parameters:
        - fb_data: ForwardBackwardData, the data containing frames to be scored.
        - prog_bar: ProgressBar, a progress bar object for visual feedback during the scoring process.
        - reset_bar: bool, a flag indicating whether to reset the progress bar before starting the scoring process.
        - thread_count: int, the number of threads to use for parallel processing of the scoring.

        Returns:
        - Tuple[np.ndarray, np.ndarray], two arrays containing the computed scores for each frame.
        """
        if("is_clustered" not in fb_data.metadata):
            raise PassOrderError(
                "Clustering must be done before frame fixing!"
            )

        scores = np.zeros((fb_data.num_frames, 2))

        num_outputs = fb_data.metadata.num_outputs
        num_frames = fb_data.num_frames
        down_scaling = fb_data.metadata.down_scaling
        skeleton = fb_data.metadata.get("skeleton", None)

        if(reset_bar and prog_bar is not None):
            prog_bar.reset(fb_data.num_frames)

        to_index = lambda i: slice(i * cls.SCORES_PER_CHUNK, (i + 1) * cls.SCORES_PER_CHUNK)
        max_dist = np.sqrt(fb_data.metadata.width ** 2 + fb_data.metadata.height ** 2)

        if(thread_count > 0):
            from ...sfpe.segmented_frame_pass_engine import PoolWithProgress
            with PoolWithProgress(prog_bar, process_count=thread_count, sub_ticks=1) as pool:
                pool.fast_map(
                    cls.compute_list_of_scores,
                    lambda i: ([list(l) for l in fb_data.frames[to_index(i)]], num_outputs, down_scaling, skeleton, max_dist),
                    lambda i, val: scores.__setitem__(to_index(i), val),
                    (fb_data.num_frames + (cls.SCORES_PER_CHUNK - 1)) // cls.SCORES_PER_CHUNK
                )
        else:
            for f_idx in range(num_frames):
                scores[f_idx] = cls.compute_single_score(
                    fb_data.frames[f_idx], num_outputs, down_scaling, skeleton, max_dist
                )
                if (prog_bar is not None):
                    prog_bar.update(1)

        return (scores[:, 0], scores[:, 1])

    @classmethod
    def restore_all_except_fix_frame(
        cls,
        fb_data: ForwardBackwardData,
        frame_idx: int,
        fix_frame_data: List[ForwardBackwardFrame],
        prog_bar: ProgressBar,
        reset_bar: bool = False,
        is_pre_initialized: bool = False
    ) -> ForwardBackwardData:
        # For passes to use....
        fb_data.metadata.fixed_frame_index = int(frame_idx)
        fb_data.metadata.is_pre_initialized = is_pre_initialized

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

            if(prog_bar is not None):
                prog_bar.update()

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

        self._scores, fallback_scores = self.compute_scores(fb_data, prog_bar, False)

        self._max_frame_idx = int(np.argmax(self._scores))

        if(np.isneginf(self._scores[self._max_frame_idx])):
            self._max_frame_idx = int(np.argmax(fallback_scores))
            self._scores = fallback_scores

        if(self.config.fix_frame_override is not None):
            if(not (0 <= self.config.fix_frame_override < len(self._scores))):
                raise ValueError("Override Fix Frame Value is not valid!")
            self._max_frame_idx = self.config.fix_frame_override

        if(self.config.DEBUG):
            print(f"Max Scoring Frame: {self._max_frame_idx}")

        self._fixed_frame = self.create_fix_frame(
            fb_data,
            self._max_frame_idx,
            fb_data.metadata.skeleton if("skeleton" in fb_data.metadata) else None,
            self.config.skeleton_assignment_algorithm
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
            ),
            "skeleton_assignment_algorithm": (
                "hungarian",
                tc.Literal("greedy", "hungarian"),
                "The algorithm to use for assigning body parts to skeletons when creating the fix frame."
            )
        }