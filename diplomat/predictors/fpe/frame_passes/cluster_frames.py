import os
from typing import Optional, Tuple, List
from scipy.sparse import csgraph
from diplomat.predictors.fpe import fpe_math
from diplomat.predictors.fpe.frame_pass import FramePass, RangeSlicer, PassOrderError
from diplomat.predictors.fpe.sparse_storage import SparseTrackingData, ForwardBackwardData, ForwardBackwardFrame, AttributeDict
from diplomat.processing import ConfigSpec, ProgressBar, Config
import numpy as np


class ClusterFrames(FramePass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._gaussian_table = fpe_math.gaussian_table(
            self.height,
            self.width,
            self.config.standard_deviation,
            self.config.amplitude,
            self.config.lowest_value
        )

        self._cluster_dict = {}

    def _get_frame(self, index: int):
        return (self._frame_data.frames[index], self._frame_data.metadata.num_outputs, self._gaussian_table, self.config)

    def _set_frame(self, index: int, frame: List[ForwardBackwardFrame]):
        self._frame_data.frames[index] = frame

    def run_pass(
        self,
        fb_data: ForwardBackwardData,
        prog_bar: Optional[ProgressBar] = None,
        in_place: bool = True,
        reset_bar: bool = True
    ) -> ForwardBackwardData:
        if("fixed_frame_index" in fb_data.metadata):
            raise PassOrderError("Can't run clustering again once frame fixing is done!")

        if(not in_place):
            raise ValueError("Clustering must be done in place!")

        thread_count = os.cpu_count() if(self.config.thread_count is None) else self.config.thread_count

        if(self.multi_threading_allowed and (thread_count > 0)):
            from diplomat.predictors.sfpe.segmented_frame_pass_engine import PoolWithProgress

            self._frame_data = fb_data
            self._frame_data.allow_pickle = False

            iter_range = RangeSlicer(self._frame_data.frames)[self._start:self._stop:self._step]


            with PoolWithProgress(prog_bar, process_count=thread_count, sub_ticks=1) as pool:
                pool.fast_map(
                    ClusterFrames._cluster_frames,
                    lambda i: self._get_frame(iter_range[i]),
                    lambda i, f: self._set_frame(iter_range[i], f),
                    len(iter_range)
                )
        else:
            self._cluster_dict = {}
            fb_data = super().run_pass(fb_data, prog_bar, True)

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
        # Zero out the component, this is an adjacency matrix, so we just zero out row with given node indexes...
        graph = graph[comp_mask][:, comp_mask]
        indexes = indexes[comp_mask]

        return graph, indexes

    @classmethod
    def _cluster_frames(
        cls,
        frame_data: List[ForwardBackwardFrame],
        num_outputs: int,
        gaussian_table: np.ndarray,
        config: Config,
        progress_bar: Optional[ProgressBar] = None
    ) -> List[ForwardBackwardFrame]:
        num_groups = len(frame_data) // num_outputs
        # PyCharm is dumb...
        clusters: List[Optional[List]] = [None] * num_groups

        for i, frame in enumerate(frame_data):
            group_idx, group_offset = divmod(i, num_outputs)

            y, x, prob, x_off, y_off = frame.orig_data.unpack()
            if (y is None):
                continue

            if(not frame.ignore_clustering):
                if(clusters[group_idx] is None):
                    clusters[group_idx] = cls._compute_cluster(y, x, prob, x_off, y_off, num_outputs, gaussian_table, config)

                frame.src_data = SparseTrackingData()
                frame.src_data.pack(*(clusters[group_idx][group_offset]))

        return frame_data

    @classmethod
    def _compute_cluster(
        cls,
        y: np.ndarray,
        x: np.ndarray,
        prob: np.ndarray,
        x_off: np.ndarray,
        y_off: np.ndarray,
        num_clusters: int,
        gaussian_table: np.ndarray,
        config: Config
    ) -> List[Tuple[np.ndarray, ...]]:
        # Special case: When cluster size is 1...
        if(num_clusters == 1):
            return [(y, x, prob, x_off, y_off)]

        iteration_count = 0

        trans = fpe_math.table_transition((x, y), (x, y), gaussian_table)
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
            if(iteration_count >= config.max_throwaway_count):
                # Failure to find a good collection of clusters: Just copy the entire frame to each body part...
                return [(y, x, prob, x_off, y_off) for i in range(num_clusters)]

            n_comp, labels = csgraph.connected_components(max_spanning_tree, directed=False)
            scores = cls.cluster_sum(n_comp, labels, prob[indexes])

            if((n_comp == num_clusters) and np.all(scores >= config.minimum_cluster_size)):
                break
            elif(n_comp >= num_clusters):
                # If we are above or equal to the desired amount of clusters, throw out the worst cluster...
                max_spanning_tree, indexes = cls.remove_cluster(max_spanning_tree, indexes, labels, scores)
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
            self._cluster_dict[(frame_index, bp_group)] = self._compute_cluster(y, x, prob, x_off, y_off, num_out, self._gaussian_table, self.config)

        if(not current.ignore_clustering):
            current.src_data = SparseTrackingData()
            current.src_data.pack(*(self._cluster_dict[(frame_index, bp_group)][bp_off]))

        return current


    def __reduce__(self, *args, **kwargs):
        raise ValueError("Not allowed to pickle this class!")

    @classmethod
    def get_config_options(cls) -> ConfigSpec:
        import diplomat.processing.type_casters as tc

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
            ),
            "thread_count": (
                None,
                tc.Union(tc.Literal(None), tc.RangedInteger(0, np.inf)),
                "The number of threads to use during processing. If None, uses os.cpu_count(). If 0 disables multithreading."
            )
        }