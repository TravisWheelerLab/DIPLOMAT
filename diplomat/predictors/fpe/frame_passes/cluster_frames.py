import os
from typing import Optional, Tuple, List
from diplomat.utils.graph_ops import connected_components, min_spanning_tree, to_valid_graph
from diplomat.predictors.fpe import fpe_math
from diplomat.predictors.fpe.arr_utils import find_peaks
from diplomat.predictors.fpe.frame_pass import FramePass, RangeSlicer, PassOrderError
from diplomat.predictors.fpe.sparse_storage import SparseTrackingData, ForwardBackwardData, ForwardBackwardFrame, AttributeDict
from diplomat.processing import ConfigSpec, ProgressBar, Config
import numpy as np


class ClusterFrames(FramePass):
    """
    Breaks up each frame and separates it into a fixed number of frames, where each frame contains typically a single
    peak.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cost_table = fpe_math.get_func_table(
            self.height,
            self.width,
            lambda x, y: np.sqrt(x * x + y * y),
            False
        )

        self._cluster_dict = {}

    def _get_frame(self, index: int):
        return (
            list(self._frame_data.frames[index]),
            self._frame_data.metadata.num_outputs,
            self._cost_table,
            self.config,
            self._frame_data.metadata.down_scaling,
            self.width
        )

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
    def _cluster_frames(
        cls,
        frame_data: List[ForwardBackwardFrame],
        num_outputs: int,
        cost_table: np.ndarray,
        config: Config,
        down_scaling: float,
        width: int,
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
                    clusters[group_idx] = cls._compute_cluster(
                        y, x, prob, x_off, y_off,
                        num_outputs, width,
                        cost_table, down_scaling,
                        config.minimum_cluster_size,
                        config.max_throwaway_count
                    )

                frame.src_data = SparseTrackingData()
                frame.src_data.pack(*(clusters[group_idx][group_offset]))

        return frame_data

    @classmethod
    def _minimum_spanning_forest(
        cls,
        graph: np.ndarray,
        num_trees: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        tree = min_spanning_tree(to_valid_graph(graph))
        n_components, labels = connected_components(tree)

        del_edges = num_trees - n_components

        if(del_edges == 0):
            return tree, labels
        elif(del_edges > 0):
            no_inf_tree = tree.copy()
            no_inf_tree[~np.tri(no_inf_tree.shape[0], k=0, dtype=bool) | np.isinf(no_inf_tree)] = -np.inf
            worst_edges = np.unravel_index(
                np.argpartition(no_inf_tree, -del_edges, axis=None)[-del_edges:], no_inf_tree.shape
            )
            tree[worst_edges] = np.inf
            tree = to_valid_graph(tree)
            return tree, connected_components(tree)[1]
        else:
            scores = np.bincount(labels, weights=np.sum(graph, axis=1), minlength=n_components)

            # Get the top k (lowest values)...
            good_indexes = np.argpartition(scores, num_trees)[:num_trees]
            labels[~np.isin(labels, good_indexes)] = num_trees

            return tree, labels

    @classmethod
    def _get_spanning_forest_centers(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        prob: np.ndarray,
        x_off: np.ndarray,
        y_off: np.ndarray,
        num_clusters: int,
        cost_table: np.ndarray,
        down_scaling: float,
        balance: float,
        attempts: int
    ) -> Optional[Tuple[list, list]]:
        if(attempts <= 0):
            return None

        if(len(x) < num_clusters):
            center_x = list(np.append(x, [np.inf] * (num_clusters - len(x))))
            center_y = list(np.append(y, [np.inf] * (num_clusters - len(y))))
            return (center_x, center_y)

        x_true = (x + 0.5 + x_off / down_scaling).astype(int)
        y_true = (y + 0.5 + y_off / down_scaling).astype(int)

        trans = fpe_math.table_transition((x_true, y_true), (x_true, y_true), cost_table)
        graph = (2 + trans) - (np.expand_dims(prob, 1)) - (np.expand_dims(prob, 0))

        __, components = cls._minimum_spanning_forest(graph, num_clusters)

        prob_max = np.array([np.max(prob[components == i]) for i in range(num_clusters)])
        prob_max = prob_max / ((np.sum(prob_max) - prob_max) / (prob_max.size - 1))
        bad_idx = np.flatnonzero(prob_max < balance)

        if(len(bad_idx) > 0):
            keep = ~np.isin(components, bad_idx)
            result = cls._get_spanning_forest_centers(
                x[keep],
                y[keep],
                prob[keep],
                x_off[keep],
                y_off[keep],
                num_clusters,
                cost_table,
                down_scaling,
                balance,
                attempts - 1
            )
            if(result is not None and (not np.any(np.isinf(result)))):
                return result

        center_x = [np.mean(x_true[components == i]) for i in range(num_clusters)]
        center_y = [np.mean(y_true[components == i]) for i in range(num_clusters)]

        return center_x, center_y

    @classmethod
    def _compute_cluster(
        cls,
        y: np.ndarray,
        x: np.ndarray,
        prob: np.ndarray,
        x_off: np.ndarray,
        y_off: np.ndarray,
        num_clusters: int,
        width: int,
        cost_table: np.ndarray,
        down_scaling: float,
        balance: float,
        attempts: int
    ) -> List[Tuple[np.ndarray, ...]]:
        # Special case: When cluster size is 1...
        if(num_clusters == 1):
            return [(y, x, prob, x_off, y_off)]

        # Find peak locations...
        top_indexes = find_peaks(x, y, prob, width)

        coords = cls._get_spanning_forest_centers(
            x[top_indexes],
            y[top_indexes],
            prob[top_indexes],
            x_off[top_indexes],
            y_off[top_indexes],
            num_clusters,
            cost_table,
            down_scaling,
            balance,
            int(attempts)
        )

        if(coords is None):
            return [(y, x, prob, x_off, y_off) for i in range(num_clusters)]

        dists = [(x - tx) ** 2 + (y - ty) ** 2 for tx, ty in zip(*coords)]
        # Construct masks...
        mask_arr = np.argmin(dists, axis=0)
        masks = [mask_arr == i for i in range(num_clusters)]
        
        return [
            (y[mask], x[mask], prob[mask], x_off[mask], y_off[mask])
            if(np.isfinite(tx) and np.isfinite(ty)) else (None, None, None, None, None)
            for mask, tx, ty in zip(masks, *coords)
        ]

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
            self._cluster_dict[(frame_index, bp_group)] = self._compute_cluster(
                y, x, prob, x_off, y_off, num_out, metadata.width,
                self._cost_table, metadata.down_scaling,
                self.config.minimum_cluster_size,
                self.config.max_throwaway_count
            )

        if(not current.ignore_clustering):
            current.src_data = SparseTrackingData()
            current.src_data.pack(*(self._cluster_dict[(frame_index, bp_group)][bp_off]))

        return current

    def __reduce__(self, *args, **kwargs):
        raise ValueError("Not allowed to pickle this class!")

    @classmethod
    def get_config_options(cls) -> ConfigSpec:
        import diplomat.processing.type_casters as tc

        return {
            "minimum_cluster_size": (
                0.10, float, "The minimum size a cluster is allowed to be (As compared to average of all clusters)."
                             "If the cluster is smaller, it get thrown out and a forest is resolved using the rest of"
                             "the data."
            ),
            "max_throwaway_count": (
                10, float, "The maximum number of clusters to throw away before giving up on clustering a given frame."
            ),
            "thread_count": (
                None,
                tc.Union(tc.Literal(None), tc.RangedInteger(0, np.inf)),
                "The number of threads to use during processing. If None, uses os.cpu_count(). "
                "If 0 disables multithreading."
            )
        }