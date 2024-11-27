from typing import List, Dict, TypeVar, Tuple, Type, Any, Sequence, Optional, Set, Callable
from diplomat.processing import *
from diplomat.predictors.fpe import fpe_math
from diplomat.predictors.fpe.skeleton_structures import StorageGraph
from diplomat.predictors.fpe.sparse_storage import ForwardBackwardData, ForwardBackwardFrame, SparseTrackingData, AttributeDict
from diplomat.predictors.fpe.frame_pass import FramePass, PassOrderError
from diplomat.predictors.fpe.frame_passes.fix_frame import FixFrame
from diplomat.predictors.fpe.frame_passes.mit_viterbi import norm, to_log_space, from_log_space
import numpy as np

class RepairClusters(FramePass):
    """
    Scans source data frames for clustering errors that split bodies into 
    positions not viable under skeletal constraints. When a split body is 
    encountered, a new clustering is created by choosing the most probable 
    positions in the original (unclustered) frame data relative to the 
    skeleton's distance frequencies.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_config_options(cls) -> ConfigSpec:
        import diplomat.processing.type_casters as tc

        return {
            "max_difference_factor": (
                5, int,     "Sets the threshold by which a distance between"
                            "body parts can exceed the expected distance from"
                            "the skeleton. Defaults to 2x."
            )
        }

    def run_pass(
        self,
        fb_data: ForwardBackwardData,
        prog_bar: Optional[ProgressBar] = None,
        in_place: bool = True,
        reset_bar: bool = True
    ) -> ForwardBackwardData:
        """
        """

        skeleton = fb_data.metadata.get("skeleton", None)
        
        if("is_clustered" not in fb_data.metadata):
            raise PassOrderError(
                "Clustering must be done before cluster repair!"
            )

        if(skeleton is None):
            raise PassOrderError(
                "Skeleton must be created before cluster repair!"
            )

        splits = RepairClusters._locate_splits(
            fb_data,
        )


        num_frames = len(fb_data.frames)
        num_bodies = fb_data.metadata.num_outputs

        print(f"num splits {len(splits)} / {num_frames * num_bodies}")

        best_comps = RepairClusters._select_best_components(
            splits,
            fb_data,
        )

        fb_data = RepairClusters._recreate_clusters(
            splits,
            best_comps,
            fb_data,
        )

        return fb_data

    def run_step(
        self,
        prior: Optional[ForwardBackwardFrame],
        current: ForwardBackwardFrame,
        frame_index: int,
        bodypart_index: int,
        metadata: AttributeDict
    ) -> Optional[ForwardBackwardFrame]:
        return current

    @classmethod
    def _detect_split_bodies(
        cls,
        frames: List[ForwardBackwardFrame],
        skeleton: StorageGraph,
        num_bodies: int,
        num_parts: int,
        down_scaling: int,
        max_difference_factor: int = 5,
    ) -> List[int]:
        """
        Detects split bodies from the body part frames at a particular timestep 
        by comparing distances between body parts in each body to the distance 
        distribution in the skeleton for the respective pair of body parts.
        Returns a list of body indices for bodies that are split.
        """
        split = [False] * num_bodies

        for body_idx in range(num_bodies):
            for part_idx in range(num_parts):
                if split[body_idx]:
                    # a split has already been identified in this body; 
                    # move on to the next one.
                    break
                # recover location of first part
                idx = (num_bodies * part_idx) + body_idx
                if not ((part_idx, body_idx) == divmod(idx, num_bodies)):
                    print((part_idx, body_idx),divmod(idx, num_bodies))
                    assert False
                part_x, part_y, part_p = FixFrame.get_max_location(frames[idx], down_scaling)
                if (part_x == None or part_y == None):
                    split[body_idx] = True
                    break
                for part2_idx in range(part_idx + 1, num_parts):
                    # recover location of second part
                    idx2 = (num_bodies * part2_idx) + body_idx
                    if not ((part2_idx, body_idx) == divmod(idx2, num_bodies)): 
                        print((part2_idx, body_idx),divmod(idx2, num_bodies))
                        assert False
                    part2_x, part2_y, part2_p = FixFrame.get_max_location(frames[idx2], down_scaling)
                    # measure distance between parts
                    if (part2_x == None or part2_y == None):
                        continue
                    measured_distance = FixFrame.dist((part_x,part_y),(part2_x,part2_y))
                    # compare to skeleton distribution
                    try:
                        expected_distance = skeleton[part_idx,part2_idx][2]
                        if measured_distance > max_difference_factor * expected_distance:
                            # if past threshold, mark split body,
                            # and break both part loops.
                            split[body_idx] = True
                            break
                    except KeyError:
                        pass
        return np.arange(num_bodies)[split]
                        

    @classmethod
    def _locate_splits(
        cls,
        fb_data: ForwardBackwardData,
    ) -> List[Tuple[int,int]]:
        """
        Given ForwardBackwardData `fb_data` with clusters and a skeleton,
        detects bodies that have been split by clustering errors. Returns a
        list of (frame index, body index) tuples which locate clustering 
        errors.
        """
        num_frames = len(fb_data.frames)
        num_bodies = fb_data.metadata.num_outputs
        num_parts = len(fb_data.metadata.bodyparts)
        down_scaling = fb_data.metadata.down_scaling
        skeleton = fb_data.metadata.skeleton

        split_locations = []
        
        for frame_idx in range(num_frames):
            split_bodies = cls._detect_split_bodies(
                fb_data.frames[frame_idx],
                skeleton,
                num_bodies,
                num_parts,
                down_scaling)
            split_locations.extend(zip(
                [frame_idx] * len(split_bodies), 
                split_bodies))
        
        return split_locations

    @classmethod
    def _create_body_graph(
        cls,
        frames: List[ForwardBackwardFrame],
        skeleton: StorageGraph,
        body_idx: int,
        num_bodies: int,
        num_parts: int,
        down_scaling: int,
        max_difference_factor: int = 5,
    ) -> List[StorageGraph]:
        """
        Construct graphs for each body with body parts as nodes and edges
        between body parts whose distance is less than `max_difference_factor`
        times greater than the expected (skeletal) distance.
        """
        body_graph = StorageGraph(skeleton.node_names())
        
        for part_idx in range(num_parts):
            # recover location of first part
            idx = (num_bodies * part_idx) + body_idx
            if not ((part_idx, body_idx) == divmod(idx, num_bodies)):
                print((part_idx, body_idx),divmod(idx, num_bodies))
                assert False
            part_x, part_y, part_p = FixFrame.get_max_location(frames[idx], down_scaling)
            if (part_x == None or part_y == None):
                continue
            for part2_idx in range(part_idx + 1, num_parts):
                # recover location of second part
                idx2 = (num_bodies * part2_idx) + body_idx
                if not ((part2_idx, body_idx) == divmod(idx2, num_bodies)): 
                    print((part2_idx, body_idx),divmod(idx2, num_bodies))
                    assert False
                part2_x, part2_y, part2_p = FixFrame.get_max_location(frames[idx2], down_scaling)
                # measure distance between parts
                if (part2_x == None or part2_y == None):
                    continue
                measured_distance = FixFrame.dist((part_x,part_y),(part2_x,part2_y))
                # compare to skeleton distribution
                try:
                    expected_distance = skeleton[part_idx,part2_idx][2]
                    if measured_distance < max_difference_factor * expected_distance:
                        # if below threshold, create an edge.
                        body_graph[part_idx,part2_idx] = measured_distance
                except KeyError:
                    pass
        
        return body_graph

    @classmethod
    def _score_components(
        cls,
        components: np.array,
        body_graph: StorageGraph,
        frames: List[ForwardBackwardFrame],
        skeleton: StorageGraph,
        body_idx: int,
        num_bodies: int,
        num_parts: int,
        down_scaling: int,
    ) -> List[float]:
        """
        Measures the variance of clusters' edges' distances from the average 
        distance in the skeleton.
        """
        node_names = skeleton.node_names()
        component_ids = np.unique(components)
        component_node_lists = [np.arange(num_parts)[components == cid] for cid in component_ids]
        component_scores = [0] * len(component_ids)
        for (component_idx, component_nodes) in enumerate(component_node_lists):
            # singleton components are maximum badness.
            if len(component_nodes) == 1:
                component_scores[component_idx] = -np.inf
                continue
            # if not a singleton, score by the edge distance variance.
            for node in component_nodes:
                for (node2, measured_distance) in body_graph[node_names[node]]:
                    # to avoid double counting
                    if node2 > node:
                        expected_distance = skeleton[node_names[node],node_names[node2]][2]
                        component_scores[component_idx] -= (expected_distance - measured_distance) ** 2
        return component_ids, component_scores
        
    @classmethod
    def _select_best_components(
        cls,
        splits: List[Tuple[int,int]],
        fb_data: ForwardBackwardData,
    ) -> List[Tuple[List[int],List[int]]]:
        """
        Given `split` locations from _detect_splits, and ForwardBackwardData
        `fb_data`, a graph is constructed for each split body from peaks of the
        body part clusters. Edges are created only if the distance between
        peaks is permitted by the skeleton. The best component of the graph is
        selected as the one whose edges' distance minimize skeletal variance.
        Returns a list of (parts_in_component, parts_outside_component) tuples.
        """
        num_frames = len(fb_data.frames)
        num_bodies = fb_data.metadata.num_outputs
        num_parts = len(fb_data.metadata.bodyparts)
        down_scaling = fb_data.metadata.down_scaling
        skeleton = fb_data.metadata.skeleton

        num_splits = len(splits)
        best_components = [None] * num_splits
        
        for (split_idx, (frame_idx, body_idx)) in enumerate(splits):
            # create body graph, find connected components
            body_graph = cls._create_body_graph(
                fb_data.frames[frame_idx],
                skeleton,
                body_idx,
                num_bodies,
                num_parts,
                down_scaling)
            body_components = np.array(body_graph.dfs())
            # score components by variance of edges from skeleton mean distance
            component_ids, component_scores = cls._score_components(
                body_components,
                body_graph,
                fb_data.frames[frame_idx],
                skeleton,
                body_idx,
                num_bodies,
                num_parts,
                down_scaling)
            optimal_component_idx = component_ids[np.argmax(component_scores)]
            # create partition of parts into those present in the optimal 
            # component, and everything else.
            optimal_component_parts = np.arange(num_parts)[body_components == optimal_component_idx]
            nonoptimal_parts = np.arange(num_parts)[body_components != optimal_component_idx]

            best_components[split_idx] = (optimal_component_parts, nonoptimal_parts)
        
        return best_components

    @classmethod
    def _recreate_clusters(
        cls,
        splits: List[Tuple[int,int]],
        best_components: List[Tuple[List[int],List[int]]],
        fb_data: ForwardBackwardData,
    ) -> ForwardBackwardData:
        """
        Given `split` locations, the `best_components` within split bodies, and
        ForwardBackwardData `fb_data`, bad clusters are repaired by applying
        skeletal distance frequencies to find the most probable position of
        every misplaced body part. Returns a list of new clusters for each body
        part of each split body.
        """
        # for each split body and each misplaced body part of that body, use 
        # the remaining, well-placed parts in the body's optimal cluster to 
        # assign skeleton-augmented probabilities to the original unclustered 
        # frame data. select as a new position for the misplaced body part the
        # maximum-probability location in this skeleton-augmented frame data. 
        num_splits = len(splits)
        num_bodies = fb_data.metadata.num_outputs
        down_scaling = fb_data.metadata.down_scaling
        assert len(best_components) == num_splits
        for split_idx in range(num_splits):
            frame_idx, body_idx = splits[split_idx]
            optimal_parts, misplaced_parts = best_components[split_idx]
            for part_idx in misplaced_parts:
                # discard the clustering altogether on misplaced parts
                fb_data.frames[frame_idx][(num_bodies * part_idx) + body_idx].src_data = fb_data.frames[frame_idx][(num_bodies * part_idx) + body_idx].orig_data.duplicate()
        return fb_data