from diplomat.predictors.fpe.frame_pass import FramePass
from diplomat.predictors.fpe.skeleton_structures import StorageGraph, Histogram
from diplomat.predictors.fpe.sparse_storage import ForwardBackwardData, ForwardBackwardFrame, AttributeDict
from diplomat.processing import ProgressBar, ConfigSpec
import diplomat.processing.type_casters as tc
from typing import Union, Optional, Dict, List, Tuple
import numpy as np


class CreateSkeleton(FramePass):
    """
    Computes optimal skeletal link distances and then constructs a skeleton to be used by :py:plugin:`~diplomat.predictors.frame_passes.MITViterbi`.
    The links can be passed directly to this frame pass or are otherwise inferred from the config file.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skeleton = None
        self._max_locations = None
        self._prior_max_locations = None
        self._current_frame = None

    def run_pass(
        self,
        fb_data: ForwardBackwardData,
        prog_bar: Optional[ProgressBar] = None,
        in_place: bool = True,
        reset_bar: bool = True
    ) -> ForwardBackwardData:
        # Construct a graph to store skeleton values...
        self._frame_data = fb_data
        self._skeleton = StorageGraph(fb_data.metadata.bodyparts)
        self._max_locations = [None] * (fb_data.metadata.num_outputs * len(fb_data.metadata.bodyparts))
        self._prior_max_locations = None
        self._current_frame = 0

        # Build a graph...
        has_skel = self._build_skeleton_graph()

        if(not has_skel):
            return fb_data

        if(self.config.part_weights is not None):
            new_skeleton_info = StorageGraph(fb_data.metadata.bodyparts)
            for node1, node2, val in self.config.part_weights:
                new_skeleton_info[node1, node2] = (val, 1, val)

            new_frame_data = fb_data
        else:
            # Build up skeletal data with frequencies.
            new_frame_data = super().run_pass(fb_data, prog_bar, in_place)

            # Grab max frequency skeletal distances and store them for later passes...
            new_skeleton_info = StorageGraph(fb_data.metadata.bodyparts)
            for edge, hist in self._skeleton.items():
                new_skeleton_info[edge] = hist.get_max()

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
        if(self.config.part_weights is not None):
            lnk_parts = [(a, b) for a, b, weight in self.config.part_weights]
        else:
            lnk_parts = self.config.linked_parts

        if(lnk_parts is None):
            lnk_parts = self._frame_data.metadata.project_skeleton

        if((lnk_parts is not None) and (lnk_parts != False)):
            if(lnk_parts == True):
                lnk_parts = list(self._skeleton.node_names())
            if(not isinstance(lnk_parts[0], (tuple, list))):
                lnk_parts = [(a, b) for a in lnk_parts for b in lnk_parts if(a != b)]

            for (bp1, bp2) in lnk_parts:
                if((bp1 in self._skeleton) and (bp2 in self._skeleton)):
                    self._skeleton[bp1, bp2] = Histogram(self.config.bin_size, self.config.bin_offset)
                else:
                    lst = [bp for bp in (bp1, bp2) if(bp not in self._skeleton)]
                    raise ValueError(
                        f"The skeleton included contains body parts not found in the project: {' and '.join(lst)}"
                    )

            return True
        return False

    def run_step(
        self,
        prior: Optional[ForwardBackwardFrame],
        current: ForwardBackwardFrame,
        frame_index: int,
        bodypart_index: int,
        metadata: AttributeDict
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

                    d = np.inf

                    for bp2 in range(s2, s2 + total_bp):
                        (p2, x2, y2) = self._max_locations[bp2]

                        if(p2 is None):
                            continue

                        d = min(((x - x2) ** 2 + (y - y2) ** 2) ** 0.5, d)

                    if(np.isfinite(d)):
                        hist.add(d)

                for bp1 in range(s2, s2 + total_bp):
                    (p, x, y) = self._prior_max_locations[bp1]

                    if(p is None):
                        continue

                    d = np.inf

                    for bp2 in range(s1, s1 + total_bp):
                        (p2, x2, y2) = self._max_locations[bp2]

                        if(p2 is None):
                            continue

                        d = min(((x - x2) ** 2 + (y - y2) ** 2) ** 0.5, d)

                    if(np.isfinite(d)):
                        hist.add(d)

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
            "linked_parts": (
                None,
                cls.cast_skeleton,
                "None, a boolean, a list of strings, or a list of strings to strings (as tuples). Determines what "
                "parts should be linked together. with a skeleton. If None, attempts to use the skeleton pulled form "
                "the tracking project. If false, specifies no skeleton should be made, basically disabling this pass. "
                "If True, connect all body parts to each other. If a list of strings, connect the body parts in that "
                "list to every other body part in that list. If a list of strings to strings, specifies exact links "
                "that should be made between body parts. Defaults to None."
            ),
            "part_weights": (
                None,
                tc.Optional[tc.List[tc.Tuple[str, str, tc.RangedFloat(0, np.inf)]]],
                "Optional list of tuples. Each tuple contains the edge (two strings) and the distance to use between"
                "those two parts, measured in pixels. This allows for manual specification of the skeleton weights."
                "This value defaults to None, meaning run automated skeleton selection."
            ),
            "bin_size": (
                1 / 4,
                tc.RoundedDecimal(5),
                "A decimal, the size of each bin used in the histogram for computing the mode."
            ),
            "bin_offset": (
                0,
                tc.RoundedDecimal(5),
                "A decimal, the offset of the first bin used in the histogram for computing the mode."
            ),
            "max_amplitude": (
                1, float, "A float, the max amplitude of the skeletal curves."
            ),
            "min_amplitude": (
                0.8, float, "A float the min amplitude of the skeletal curves."
            ),
            "DEBUG": (
                False, bool, "Set to True to print skeleton information to console while this pass is running."
            )
        }