from typing import Dict, Tuple, Optional, List, Union

import numpy
import numpy as np
from .sparse_storage import ForwardBackwardData, AttributeDict, ForwardBackwardFrame, SparseTrackingData
from .skeleton_structures import StorageGraph, Histogram
from .frame_pass import FramePass, ProgressBar, PassOrderError, ConfigSpec
from .frame_pass import type_casters as tc
from scipy.sparse import csgraph
from . import fpe_math


class OptimizeStandardDeviation(FramePass):
    def __init__(self, width, height, config):
        super().__init__(width, height, config)

        self._histogram = Histogram(self.config.bin_size, self.config.bin_offset)
        self._current_frame = None
        self._prior_max_locations = None
        self._max_locations = None

    def run_pass(
        self,
        fb_data: ForwardBackwardData,
        prog_bar: Optional[ProgressBar] = None,
        in_place: bool = True,
        reset_bar: bool = True
    ) -> ForwardBackwardData:
        self._current_frame = 0
        self._prior_max_locations = None
        self._max_locations = [None] * fb_data.num_bodyparts

        result = super().run_pass(fb_data, prog_bar, in_place, reset_bar)

        result.metadata.optimal_std = Histogram.to_floats(
            self._histogram.get_quantile(self.config.quantile)
        )

        if(self.config.DEBUG):
            print(f"Optimal STD: {result.metadata.optimal_std}")

        return result

    def run_step(
        self,
        prior: Optional[ForwardBackwardFrame],
        current: ForwardBackwardFrame,
        frame_index: int,
        bodypart_index: int,
        metadata: AttributeDict
    ) -> Optional[ForwardBackwardFrame]:
        if(self._current_frame != frame_index):
            if(self._prior_max_locations is not None):
                for bp_gi in range(len(self._max_locations) // metadata.num_outputs):
                    for ci in range(metadata.num_outputs):
                        cp, cx, cy = self._max_locations[bp_gi * metadata.num_outputs + ci]

                        if(cp is None):
                            continue

                        min_dist = np.inf

                        for pi in range(metadata.num_outputs):
                            pp, px, py = self._prior_max_locations[bp_gi * metadata.num_outputs + pi]

                            if(pp is None):
                                continue

                            min_dist = min(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5, min_dist)

                        if(min_dist != np.inf):
                            self._histogram.add(min_dist)

            self._prior_max_locations = self._max_locations
            self._max_locations = [None] * self.fb_data.num_bodyparts
            self._current_frame = frame_index

        y, x, probs, ox, oy = current.src_data.unpack()

        if(y is None):
            self._max_locations[bodypart_index] = (None, 0, 0)
            return None

        max_loc = np.argmax(probs)

        self._max_locations[bodypart_index] = (
            probs[max_loc],
            x[max_loc] + 0.5 + ox[max_loc] / metadata.down_scaling,
            y[max_loc] + 0.5 + oy[max_loc] / metadata.down_scaling
        )

        return None


    @classmethod
    def get_config_options(cls) -> ConfigSpec:
        return {
            "bin_size": (1 / 8, tc.RoundedDecimal(5), "A decimal, the size of each bin used in the histogram for computing the mode."),
            "bin_offset": (0, tc.RoundedDecimal(5), "A decimal, the offset of the first bin used in the histogram for computing "
                                   "the mode."),
            "quantile": (0.93, tc.RoundedDecimal(5), "The quantile to use as the standard deviation... Defaults to 0.93 or 93%"),
            "DEBUG": (False, bool, "Set to True to print the optimal standard deviation found...")
        }


class CreateSkeleton(FramePass):
    def __init__(self, width, height, config):
        super().__init__(width, height, config)
        self._skeleton = None
        self._max_locations = None
        self._prior_max_locations = None
        self._current_frame = None

    def run_pass(self,
        fb_data: ForwardBackwardData,
        prog_bar: Optional[ProgressBar] = None,
        in_place: bool = True,
        reset_bar: bool = True
    ) -> ForwardBackwardData:
        # Construct a graph to store skeleton values...
        self._skeleton = StorageGraph(fb_data.metadata.bodyparts)
        self._max_locations = [None] * (fb_data.metadata.num_outputs * len(fb_data.metadata.bodyparts))
        self._prior_max_locations = None
        self._current_frame = 0

        # Build a graph...
        has_skel = self._build_skeleton_graph()

        # Build up skeletal data with frequencies.
        new_frame_data = super().run_pass(fb_data, prog_bar, in_place)

        # Grab max frequency skeletal distances and store them for later passes...
        if(has_skel):
            new_skeleton_info = StorageGraph(fb_data.metadata.bodyparts)
            for edge, hist in self._skeleton.items():
                new_skeleton_info[edge] = Histogram.to_floats(hist.get_max())

            new_frame_data.metadata.skeleton = new_skeleton_info
            new_frame_data.metadata.skeleton_config = {
                "peak_amplitude": self.config.max_amplitude,
                "trough_amplitude": self.config.min_amplitude
            }

            if(self.config.DEBUG):
                print("Selected Skeleton Lengths (bin, freq, avg):")
                print(new_skeleton_info)
                print("Skeleton Histograms:")
                print(self._skeleton)

        return new_frame_data

    def _build_skeleton_graph(self) -> bool:
        lnk_parts = self.config.linked_parts

        if((lnk_parts is not None) and (lnk_parts != False)):
            if(lnk_parts == True):
                lnk_parts = list(self._skeleton.node_names())
            if(not isinstance(lnk_parts[0], (tuple, list))):
                lnk_parts = [(a, b) for a in lnk_parts for b in lnk_parts if(a != b)]

            for (bp1, bp2) in lnk_parts:
                self._skeleton[bp1, bp2] = Histogram(self.config.bin_size, self.config.bin_offset)

            return True
        return False

    def run_step(
        self,
        prior: Optional[ForwardBackwardFrame],
        current: ForwardBackwardFrame, frame_index: int,
        bodypart_index: int, metadata: AttributeDict
    ) -> Optional[ForwardBackwardFrame]:
        # If we have moved to the next frame, update histograms using body part maximums of prior frame...
        if(self._current_frame != frame_index):
            if(self._prior_max_locations is None):
                self._current_frame = frame_index
                self._prior_max_locations = [val for val in self._max_locations]
                return None

            total_bp = metadata.num_outputs if("is_clustered" in metadata) else 1

            for edge, hist in self._skeleton.items():
                s1, s2 = edge.node1 * metadata.num_outputs, edge.node2 * metadata.num_outputs

                for bp1 in range(s1, s1 + total_bp):
                    (p, x, y) = self._prior_max_locations[bp1]

                    if(p is None):
                        continue

                    for bp2 in range(s2, s2 + total_bp):
                        (p2, x2, y2) = self._max_locations[bp2]

                        if(p2 is None):
                            continue

                        hist.add(((x - x2) ** 2 + (y - y2) ** 2) ** 0.5)

                for bp1 in range(s2, s2 + total_bp):
                    (p, x, y) = self._prior_max_locations[bp1]

                    if(p is None):
                        continue

                    for bp2 in range(s1, s1 + total_bp):
                        (p2, x2, y2) = self._max_locations[bp2]

                        if(p2 is None):
                            continue

                        hist.add(((x - x2) ** 2 + (y - y2) ** 2) ** 0.5)

            self._current_frame = frame_index
            self._prior_max_locations = [val for val in self._max_locations]

        # Add max location in frame to list of body part maximums for this frame.
        y, x, probs, ox, oy = current.src_data.unpack()

        if(probs is None):
            self._max_locations[bodypart_index] = (None, 0, 0)
            return

        max_loc = np.argmax(probs)

        self._max_locations[bodypart_index] = (
            probs[max_loc],
            x[max_loc] + 0.5 + (ox[max_loc] / metadata.down_scaling),
            y[max_loc] + 0.5 + (oy[max_loc] / metadata.down_scaling)
        )

        return None

    @classmethod
    def cast_skeleton(cls, skel: Optional[Dict[str, str]]) -> Union[None, bool, List[str], List[Tuple[str, str]]]:
        # Validate the skeleton
        if(skel is None or isinstance(skel, bool)):
            return skel
        if(isinstance(skel, (list, tuple))):
            if(len(skel) == 0):
                return None
            elif(not isinstance(skel[0], (list, tuple))):
                return [str(v) for v in skel]
            else:
                # Convert all dictionary arguments to lists...
                return [(str(k), str(v)) for k, v in skel]


    @classmethod
    def get_config_options(cls) -> ConfigSpec:
        return {
            "linked_parts": (True, cls.cast_skeleton, "None, a boolean, a list of strings, or a list of strings "
                                                      "to strings (as tuples). Determines what parts should be linked together. "
                                                      "with a skeleton. If false or none, specifies no skeleton should"
                                                      "be made, basically disabling this pass. If True, connect all"
                                                      "body parts to each other. If a list of strings, connect the "
                                                      "body parts in that list to every other body part in that list. "
                                                      "If a list of strings to strings, specifies exact links "
                                                      "that should be made between body parts. Defaults to True."),
            "bin_size": (1 / 4, tc.RoundedDecimal(5), "A decimal, the size of each bin used in the histogram for computing the mode."),
            "bin_offset": (0, tc.RoundedDecimal(5), "A decimal, the offset of the first bin used in the histogram for computing "
                                   "the mode."),
            "max_amplitude": (1, float, "A float, the max amplitude of the skeletal curves."),
            "min_amplitude": (0.8, float, "A float the min amplitude of the skeletal curves."),
            "DEBUG": (False, bool, "Set to True to print skeleton information to console while this pass is running.")
        }


class ClusterFrames(FramePass):
    def __init__(self, width, height, config):
        super().__init__(width, height, config)

        self._gaussian_table = fpe_math.gaussian_table(
            height, width, self.config.standard_deviation, self.config.amplitude, self.config.lowest_value
        )

        self._cluster_dict = {}

    def run_pass(self,
        fb_data: ForwardBackwardData,
        prog_bar: Optional[ProgressBar] = None,
        in_place: bool = True,
        reset_bar: bool = True
    ) -> ForwardBackwardData:
        self._cluster_dict = {}

        fb_data = super().run_pass(fb_data, prog_bar, in_place)
        fb_data.metadata.is_clustered = True

        return fb_data

    @classmethod
    def cluster_sum(cls, n_comp: int, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
        weighted_sum = np.sum(weights[None] * (labels[None] == np.arange(n_comp)[:, None]), axis=1)
        return weighted_sum / np.mean(weighted_sum)

    @classmethod
    def remove_cluster(
        cls,
        graph: np.ndarray,
        indexes: np.ndarray,
        labels: np.ndarray,
        component_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        component_to_remove = np.argmin(component_scores)
        comp_mask = component_to_remove != labels
        # Zero out the component, this is an adjacency matrix so we just zero out row with given node indexes...
        graph = graph[comp_mask][:, comp_mask]
        indexes = indexes[comp_mask]

        return graph, indexes

    def _compute_cluster(
        self,
        y: np.ndarray,
        x: np.ndarray,
        prob: np.ndarray,
        x_off: np.ndarray,
        y_off: np.ndarray,
        num_clusters: int,
    ) -> List[Tuple[np.ndarray, ...]]:
        # Special case: When cluster size is 1...
        if(num_clusters == 1):
            return [(y, x, prob, x_off, y_off)]

        iteration_count = 0

        trans = fpe_math.table_transition((x, y), (x, y), self._gaussian_table)
        indexes = np.arange(len(prob))
        inv_indexes = np.arange(len(prob))
        graph = (np.expand_dims(prob, 0)) * trans * (np.expand_dims(prob, 1))

        max_spanning_tree = (-csgraph.minimum_spanning_tree(-graph)).toarray()

        out_of_tree = max_spanning_tree == 0
        max_spanning_tree[out_of_tree] = np.inf

        sorted_edges = np.unravel_index(
            np.argsort(max_spanning_tree, axis=None)[:-np.sum(out_of_tree)], max_spanning_tree.shape
        )

        deleted_edges = 0
        max_spanning_tree[out_of_tree] = 0
        # While the number of components doesn't match the desired cluster amount and not every single cluster
        # contains a 'reasonable' chunk of the total probability in the frame, continue throwing out clusters
        # and removing low scoring edges...
        while(True):
            if(iteration_count >= self.config.max_throwaway_count):
                # Failure to find a good collection of clusters: Just copy the entire frame to each body part...
                return [(y, x, prob, x_off, y_off) for i in range(num_clusters)]

            n_comp, labels = csgraph.connected_components(max_spanning_tree, directed=False)
            scores = self.cluster_sum(n_comp, labels, prob[indexes])

            if((n_comp == num_clusters) and np.all(scores >= self.config.minimum_cluster_size)):
                break
            elif(n_comp >= num_clusters):
                # If we are above or equal to the desired amount of clusters, throw out the worst cluster...
                max_spanning_tree, indexes = self.remove_cluster(max_spanning_tree, indexes, labels, scores)
                # Recompute inverse index lookup array
                inv_indexes[:] = -1
                inv_indexes[indexes] = np.arange(len(indexes))
                # Filter sorted list of edges, removing any nodes that just got removed...
                a, b = [arr[deleted_edges:] for arr in sorted_edges]
                valid = (inv_indexes[a] != -1) & (inv_indexes[b] != -1)
                sorted_edges = (a[valid], b[valid])
                deleted_edges = 0
            else:
                if(deleted_edges >= len(sorted_edges[0])):
                    # Failure to find a good collection of clusters: Just copy the entire frame to each body part...
                    return [(y, x, prob, x_off, y_off) for i in range(num_clusters)]

                # Remove low scoring edges until we attain the desired number of clusters
                num_comp_left = num_clusters - n_comp
                a, b = sorted_edges
                max_spanning_tree[
                    inv_indexes[a[deleted_edges:num_comp_left]],
                    inv_indexes[b[deleted_edges:num_comp_left]]
                ] = 0
                deleted_edges += num_comp_left

            iteration_count += 1

        masks = [indexes[labels == i] for i in range(n_comp)]

        return [(y[mask], x[mask], prob[mask], x_off[mask], y_off[mask]) for mask in masks]


    def run_step(self, prior: Optional[ForwardBackwardFrame], current: ForwardBackwardFrame, frame_index: int,
                 bodypart_index: int, metadata: AttributeDict) -> Optional[ForwardBackwardFrame]:
        num_out = metadata.num_outputs
        y, x, prob, x_off, y_off = current.orig_data.unpack()
        if(y is None):
            return None

        bp_group, bp_off = divmod(bodypart_index, num_out)

        if((frame_index - 1, bp_group) in self._cluster_dict):
            del self._cluster_dict[(frame_index - 1, bp_group)]

        if((not current.ignore_clustering) and ((frame_index, bp_group) not in self._cluster_dict)):
            self._cluster_dict[(frame_index, bp_group)] = self._compute_cluster(y, x, prob, x_off, y_off, num_out)

        if(not current.ignore_clustering):
            current.src_data = SparseTrackingData()
            current.src_data.pack(*(self._cluster_dict[(frame_index, bp_group)][bp_off]))

        return current


    @classmethod
    def get_config_options(cls) -> ConfigSpec:
        return  {
            "standard_deviation": (
                1, float, "The standard deviation of the 2D Gaussian curve used for edge weights in the graph."
            ),
            "amplitude": (
                1, float, "The max amplitude of the 2D Gaussian curve used for edge weights in the graph."
            ),
            "lowest_value": (
                0, float, "The lowest value the 2D Gaussian curve used for edge weights can reach."
            ),
            "minimum_cluster_size": (
                0.1, float, "The minimum size a cluster is allowed to be (As compared to average of all clusters)."
                            "If the cluster is smaller, it get thrown out and a forest is resolved using the rest of"
                            "the data."
            ),
            "max_throwaway_count": (
                10, float, "The maximum number of clusters to throw away before giving up on clustering a given frame."
            )
        }


class FixFrame(FramePass):

    def __init__(self, width, height, config):
        super().__init__(width, height, config)
        self._scores = None
        self._fb_data = None
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
                numpy.concatenate([data.unpack()[i] for data in track_data if(data.probs is not None)])
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

    def _fix_frame(self, fb_data: ForwardBackwardData, frame_idx: int, skeleton: Optional[StorageGraph] = None):
        # For other passes to utilize
        fb_data.metadata.fixed_frame_index = frame_idx

        self._fixed_frame = [None] * fb_data.num_bodyparts
        num_outputs = fb_data.metadata.num_outputs
        down_scaling = fb_data.metadata.down_scaling

        # Copy over data to start, ignoring skeleton...
        for bp_i in range(fb_data.num_bodyparts):
            self._fixed_frame[bp_i] = fb_data.frames[frame_idx][bp_i].copy()
            self._fixed_frame[bp_i].disable_occluded = True

        if(skeleton is not None):
            # For skeletal info, we need to swap order all clusters to get the minimum score with the skeleton...

            # Returns the skeleton score between two body parts, lower is better. (Gets absolute distance from average)
            def score(avg, frame1, frame2):
                return np.abs(avg - self.dist(
                    self.get_max_location(frame1, down_scaling),
                    self.get_max_location(frame2, down_scaling)
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
                                self._fixed_frame[prior_n_exact],
                                fb_data.frames[frame_idx][current_n * num_outputs + i]
                            ) for i in range(num_outputs)
                    ])

                    self._fixed_frame[current_n_exact] = fb_data.frames[frame_idx][current_n_best].copy()
                    self._fixed_frame[current_n_exact].disable_occluded = True

            # Run the dfs to find the best indexes for each cluster and rearrange them...
            skeleton.dfs(on_traversal)

    def run_pass(self,
        fb_data: ForwardBackwardData,
        prog_bar: Optional[ProgressBar] = None,
        in_place: bool = True,
        reset_bar: bool = True,
        run_main_pass: bool = True
    ) -> ForwardBackwardData:
        self._scores = np.zeros(fb_data.num_frames)
        self._fb_data = fb_data

        if(not "is_clustered" in fb_data.metadata):
            raise PassOrderError(
                "Clustering must be done before frame fixing!"
            )

        num_outputs = fb_data.metadata.num_outputs
        num_frames = fb_data.num_frames
        down_scaling = fb_data.metadata.down_scaling
        num_bp = fb_data.num_bodyparts // num_outputs

        if(reset_bar and prog_bar is not None):
            prog_bar.reset(fb_data.num_frames * 2)

        for f_idx in range(num_frames):

            score = 0

            for bp_group_off in range(num_bp):

                min_dist = np.inf
                # For body part groupings...
                for i in range(num_outputs - 1):
                    f1_loc = self.get_max_location(
                        fb_data.frames[f_idx][bp_group_off * num_outputs + i], down_scaling
                    )

                    if(f1_loc[0] is None):
                        min_dist = -np.inf
                        continue

                    for j in range(i + 1, num_outputs):
                        f2_loc = self.get_max_location(
                            fb_data.frames[f_idx][bp_group_off * num_outputs + j], down_scaling
                        )

                        if(f2_loc[0] is None):
                            min_dist = -np.inf
                            continue

                        min_dist = min(self.dist(f1_loc, f2_loc), min_dist)

                score += min_dist

            # If skeleton is implemented...
            if("skeleton" in fb_data.metadata):
                skel = fb_data.metadata.skeleton

                for bp in range(fb_data.num_bodyparts):
                    bp_group_off, bp_off = divmod(bp, num_outputs)

                    num_pairs = num_outputs * len(skel[bp_group_off])
                    f1_loc = self.get_max_location(
                        fb_data.frames[f_idx][bp_group_off * num_outputs + bp_off], down_scaling
                    )

                    if(f1_loc[0] is None):
                        score -= np.inf
                        continue

                    for (bp2_group_off, (__, __, avg)) in skel[bp_group_off]:
                        min_score = np.inf

                        for bp2_off in range(num_outputs):
                            f2_loc = self.get_max_location(
                                fb_data.frames[f_idx][bp2_group_off * num_outputs + bp2_off], down_scaling
                            )

                            if(f2_loc[0] is None):
                                score -= np.inf
                                continue

                            result = np.abs(self.dist(f1_loc, f2_loc) - avg)
                            min_score = min(result, min_score)

                        score -= (min_score / num_pairs)

            self._scores[f_idx] = score
            if(prog_bar is not None):
                prog_bar.update(1)

        self._max_frame_idx = int(np.argmax(self._scores))

        if(self.config.DEBUG):
            print(f"Max Scoring Frame: {self._max_frame_idx}")

        self._fix_frame(
            fb_data,
            self._max_frame_idx,
            fb_data.metadata.skeleton if("skeleton" in fb_data.metadata) else None
        )

        # Now the pass...
        if(run_main_pass):
            return super().run_pass(fb_data, prog_bar, in_place, False)
        else:
            return fb_data

    def run_step(
        self,
        prior: Optional[ForwardBackwardFrame],
        current: ForwardBackwardFrame,
        frame_index: int,
        bodypart_index: int,
        metadata: AttributeDict
    ) -> Optional[ForwardBackwardFrame]:
        # If the fixed frame, return the fixed frame...
        if(frame_index == self._max_frame_idx):
            return self._fixed_frame[bodypart_index]
        # If any other frame, return the frame as the merged clusters...
        else:
            current.src_data = current.orig_data
            return current

    @classmethod
    def get_config_options(cls) -> ConfigSpec:
        return {
            "DEBUG": (False, bool, "Set to True to dump additional information while the pass is running.")
        }
