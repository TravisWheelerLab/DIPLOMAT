from diplomat.predictors.fpe.frame_pass import FramePass, PassOrderError
from diplomat.predictors.fpe.frame_passes.fix_frame import FixFrame

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

    def run_pass(
        self,
        fb_data: ForwardBackwardData,
        prog_bar: Optional[ProgressBar] = None,
        in_place: bool = True,
        reset_bar: bool = True
    ) -> ForwardBackwardData:
        
        if("is_clustered" not in fb_data.metadata):
            raise PassOrderError(
                "Clustering must be done before cluster repair!"
            )

        if(skeleton is None):
            raise PassOrderError(
                "Skeleton must be created before cluster repair!"
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

    # following two methods are lifted from fix_frame; maybe these should be moved to a utility file instead.

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
    def _detect_split_bodies(
        cls,
        frames: List[ForwardBackwardFrame],
        skeleton: StorageGraph
        num_bodies: int,
        num_parts: int,
        down_scaling: int,
        expected_distance: int = 2,
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
                idx = (3 * part_idx) + body_idx
                assert body_idx, part_idx == divmod(idx, num_bodies)
                part_x, part_y, part_p = cls.get_max_location(frames[idx], down_scaling)
                for part2_idx in range(part_idx + 1, num_parts):
                    # recover location of second part
                    idx2 = (3 * part_idx2) + body_idx
                    assert body_idx, part_idx2 == divmod(idx2, num_bodies)
                    part2_x, part2_y, part2_p = cls.get_max_location(frames[idx2], down_scaling)
                    # measure distance between parts
                    measured_distance = cls.dist((part_x,part_y),(part2_x,part2_y))
                    # compare to skeleton distribution
                    expected_distance = skeleton[part_idx,part_idx2][2]
                    if measured_distance > max_difference_factor * expected_distance:
                        # if past threshold, mark split body,
                        # and break both part loops.
                        split[body_idx] == True
                        break
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
        skeleton = fb_data.metadata.get("skeleton", None)

        split_locations = []
        
        for frame_idx in range(num_frames):
            split_bodies = cls._detect_splits(
                fb_data.frames[frame_idx],
                skeleton,
                num_bodies,
                num_parts,
                down_scaling)
            split_locations.extend(zip(
                [frame_index] * len(split_bodies), 
                split_bodies))
        
        return split_locations

    @classmethod
    def _create_body_graphs(
        cls,
        frames: List[ForwardBackwardFrame],
        skeleton: StorageGraph,
        body_idx: int,
        num_bodies: int,
        num_parts: int,
        down_scaling: int,
        max_difference_factor: int = 2,
    ) -> List[StorageGraph]:
        """
        Construct graphs for each body with body parts as nodes and edges
        between body parts whose distance is less than `max_difference_factor`
        times greater than the expected (skeletal) distance.
        """
        #body_graphs = [StorageGraph(skeleton.node_names()) for _ in range(num_bodies)]
        body_graph = StorageGraph(skeleton.node_names())
        
        #for body_idx in range(num_bodies):
        for part_idx in range(num_parts):
            # recover location of first part
            idx = (3 * part_idx) + body_idx
            assert body_idx, part_idx == divmod(idx, num_bodies)
            part_x, part_y, part_p = cls.get_max_location(frames[idx], down_scaling)
            for part2_idx in range(part_idx + 1, num_parts):
                # recover location of second part
                idx2 = (3 * part_idx2) + body_idx
                assert body_idx, part_idx2 == divmod(idx2, num_bodies)
                part2_x, part2_y, part2_p = cls.get_max_location(frames[idx2], down_scaling)
                # measure distance between parts
                measured_distance = cls.dist((part_x,part_y),(part2_x,part2_y))
                # compare to skeleton distribution
                expected_distance = skeleton[part_idx,part_idx2][2]
                if measured_distance < max_difference_factor * expected_distance:
                    # if below threshold, create an edge.
                    body_graphs[body_idx][part_idx,part_idx2] = measured_distance
        
        #return body graphs
        return body_graph

    @classmethod
    def _score_components(
        cls,
        body_graph: StorageGraph,
        components: List[int],
        frames: List[ForwardBackwardFrame],
        skeleton: StorageGraph,
        body_idx: int,
        num_bodies: int,
        num_parts: int,
        down_scaling: int,
    ) -> List[float]:
        pass

    @classmethod
    def _select_best_components(
        cls,
        splits: List[Tuple[int,int]],
        fb_data: ForwardBackwardData
    ) -> List[Tuple[List[int],List[int]]]:
        """
        Given `split` locations from _detect_splits, and ForwardBackwardData
        `fb_data`, a graph is constructed for each split body from peaks of the
        body part clusters. Edges are created only if the distance between
        peaks is permitted by the skeleton. The best component of the graph is
        selected as the one whose edges' distance are maximally probable.
        Returns a list of (parts_in_component, parts_outside_component) tuples.
        """
        num_frames = len(fb_data.frames)
        num_bodies = fb_data.metadata.num_outputs
        num_parts = len(fb_data.metadata.bodyparts)
        down_scaling = fb_data.metadata.down_scaling
        skeleton = fb_data.metadata.get("skeleton", None)

        num_splits = len(splits)
        best_components = [None] * num_splits
        
        for (split_idx, (frame_idx, body_idx)) in enumerate(splits):
            # create body graph, find components
            body_graph = cls._create_body_graph(
                fb_data.frames[frame_idx],
                skeleton,
                body_idx,
                num_bodies,
                num_parts,
                down_scaling)
            body_components = body_graph.dfs()
            # score components
            component_scores = cls._score_components(
                body_graph,
                components,
                fb_data.frames[frame_idx],
                skeleton,
                body_idx,
                num_bodies,
                num_parts,
                down_scaling
            )
            optimal_component_idx = np.argmax(component_scores)
            # create partition of optimal component parts, nonoptimal parts
            optimal_component_parts = [part for part in range(parts) if body_components[part] == optimal_component_idx]
            nonoptimal_parts = [part for part in range(num_parts) if body_components[part] != optimal_component_idx]
            best_components[split_idx] = (optimal_component_parts, nonoptimal_parts)
        
        return best_components

    @classmethod
    def _recreate_clusters(
        cls,
        splits: List[Tuple[int,int]],
        best_components: List[Tuple[List[int],List[int]]],
        fb_data: ForwardBackwardData
    ) -> List[List[np.ndarray]]:
        """
        Given `split` locations, the `best_components` within split bodies, and
        ForwardBackwardData `fb_data`, bad clusters are repaired by applying
        skeletal distance frequencies to find the most probable position of
        every misplaced body part. Returns a list of new clusters for each body
        part of each split body.
        """
        pass