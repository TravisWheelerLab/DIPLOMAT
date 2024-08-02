from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional, BinaryIO

import numpy as np

from diplomat.processing import *
from diplomat.utils import frame_store_api

try:
    from .frame_pass import FramePass, ProgressBar
    from .frame_pass_loader import FramePassBuilder
    from .sparse_storage import ForwardBackwardData, SparseTrackingData, ForwardBackwardFrame, AttributeDict
    from .fpe_help import FPEString
except ImportError:
    __package__ = "diplomat.predictors.fpe"
    from .frame_pass import FramePass, ProgressBar
    from .frame_pass_loader import FramePassBuilder
    from .sparse_storage import ForwardBackwardData, SparseTrackingData, ForwardBackwardFrame, AttributeDict
    from .fpe_help import FPEString


class FramePassEngine(Predictor):
    """
    A predictor that applies a collection of frame passes to the frames
    dumped by deeplabcut, and then predicts poses by selecting maximums.
    Contains a collection of useful prediction algorithms which can be listed
    by calling "get_predictor_settings" on this Predictor.
    """

    def __init__(
        self,
        bodyparts: List[str],
        num_outputs: int,
        num_frames: int,
        settings: Config,
        video_metadata: Config,
    ):
        super().__init__(
            bodyparts, num_outputs, num_frames, settings, video_metadata
        )

        self._num_bp = len(bodyparts)
        self._num_total_bp = self._num_bp * num_outputs

        self._width, self._height = None, None

        self.PASSES = FramePassBuilder.sanitize_pass_config_list(settings.passes)
        self.THRESHOLD = settings.threshold

        p = settings.export_frame_path
        self.EXPORT_LOC = Path(p).resolve() if(p is not None) else None

        self._frame_holder = ForwardBackwardData(num_frames, self._num_total_bp)

        self._frame_holder.metadata.threshold = self.THRESHOLD
        self._frame_holder.metadata.bodyparts = bodyparts
        self._frame_holder.metadata.num_outputs = num_outputs
        self._frame_holder.metadata.project_skeleton = self.video_metadata.get("skeleton", None)

        self._current_frame = 0

    def _sparcify_and_store(self, fb_frame: ForwardBackwardFrame, scmap: TrackingData, frame_idx: int, bp_idx: int):
        fb_frame.orig_data = SparseTrackingData.sparsify(
            scmap,
            frame_idx,
            bp_idx,
            self.THRESHOLD,
            self.settings.max_cells_per_frame,
            SparseTrackingData.SparseModes[self.settings.sparsification_mode]
        )
        fb_frame.src_data = fb_frame.orig_data

    def _on_frames(self, scmap: TrackingData) -> Optional[Pose]:
        if(self._width is None):
            self._width = scmap.get_frame_width()
            self._height = scmap.get_frame_height()
            self._frame_holder.metadata.down_scaling = scmap.get_down_scaling()
            self._frame_holder.metadata.width = scmap.get_frame_width()
            self._frame_holder.metadata.height = scmap.get_frame_height()

        # Store sparsified frames for passes done later.
        for f_idx in range(scmap.get_frame_count()):
            for bp_idx in range(self._num_total_bp):
                if(bp_idx % self.num_outputs == 0):
                    self._sparcify_and_store(
                        self._frame_holder.frames[self._current_frame][bp_idx],
                        scmap,
                        f_idx,
                        bp_idx // self.num_outputs
                    )
                else:
                    dest = self._frame_holder.frames[self._current_frame][bp_idx]
                    src = self._frame_holder.frames[self._current_frame][(bp_idx // self.num_outputs) * self.num_outputs]
                    dest.orig_data = src.orig_data.duplicate()
                    dest.src_data = dest.orig_data

            self._current_frame += 1

        # No frames to return yet!
        return None

    @classmethod
    def get_maximum(
        cls,
        frame: ForwardBackwardFrame,
        relaxed_radius: float = 0,
        verbose = False,
    ) -> Tuple[int, int, float, float, float]:
        """
        PRIVATE: Get the maximum location of a single forward backward frame.
        Returns a tuple containing the values x, y, probability, x offset,
        and y offset in order.
        """
        if verbose: print("get_maximum")
        if (frame.frame_probs is None or frame.src_data.unpack()[0] is None):
            # No frame data, return 3 for no probability and 0 probability...
            if verbose: print("\tno frame data")
            return (-1, -1, 0, 0, 0)
        else:
            # Get the max location in the frame....
            y_coords, x_coords, orig_probs, x_offsets, y_offsets = frame.src_data.unpack()

            max_loc = np.argmax(frame.frame_probs)
            m_y, m_x, m_p = y_coords[max_loc], x_coords[max_loc], frame.frame_probs[max_loc]
            m_offx, m_offy = x_offsets[max_loc], y_offsets[max_loc]

            # Get the max location on the occluded state...
            try:
                max_occluded_loc = np.argmax(frame.occluded_probs)
                m_occluded_prob = frame.occluded_probs[max_occluded_loc]
                m_occ_x, m_occ_y = frame.occluded_coords[max_occluded_loc]
            except (ValueError, TypeError):
                m_occluded_prob = -np.inf
                m_occ_x, m_occ_y = 0, 0

            max_select = np.array([m_p, m_occluded_prob])
            max_of_max = np.argmax(max_select)

            if (max_of_max > 0):
                # Return correct location for occluded, but return a
                # probability of 0.
                if verbose: print("\toccluded loc")
                return (m_occ_x, m_occ_y, 0, 0, 0)
            else:
                if (relaxed_radius <= 0):
                    # If no relaxed radius, just set pose...
                    if verbose: print("\t wout radius")
                    return (m_x, m_y, m_p, m_offx, m_offy)
                else:
                    # Now find locations within the radius...
                    dists = np.sqrt(
                        (x_coords - m_x) ** 2 + (y_coords - m_y) ** 2)
                    res = np.flatnonzero(dists <= relaxed_radius)

                    # No other neighbors, return initially suggested value...
                    if (len(res) <= 0):
                        if verbose: print("\t no neighbors")
                        return (m_x, m_y, m_p, m_offx, m_offy)
                    else:
                        best_idx = res[np.argmax(orig_probs[res])]
                        if verbose: print("\t w neighbors")
                        return (
                            x_coords[best_idx], y_coords[best_idx], m_p,
                            x_offsets[best_idx], y_offsets[best_idx]
                        )

    @classmethod
    def get_maximums(
        cls,
        frame_list: ForwardBackwardData,
        progress_bar: ProgressBar,
        relaxed_radius: float = 0
    ) -> Pose:
        # Our final pose object:
        poses = Pose.empty_pose(frame_list.num_frames, frame_list.num_bodyparts)

        if(progress_bar is not None):
            progress_bar.reset(frame_list.num_frames)

        for f_idx in range(frame_list.num_frames):
            for bp_idx in range(frame_list.num_bodyparts):
                x, y, p, x_off, y_off = cls.get_maximum(
                    frame_list.frames[f_idx][bp_idx], relaxed_radius
                )

                poses.set_at(
                    f_idx, bp_idx, (x, y), (x_off, y_off), p,
                    frame_list.metadata.down_scaling
                )

            if (progress_bar is not None):
                progress_bar.update(1)

        return poses


    class ExportableFields(Enum):
        SOURCE = "Source"
        FRAME = "Frame"
        OCCLUDED_AND_EDGES = "Occluded/Edge"

    @classmethod
    def _get_frame_writer(
        cls,
        num_frames: int,
        frame_metadata: AttributeDict,
        video_metadata: Config,
        file_format: str,
        file: BinaryIO,
        export_all: bool = False
    ) -> frame_store_api.FrameWriter:
        """
        Get a frame writer that can export frames from the Forward Backward instance. Used internally to support frame
        export functionality.

        TODO
        """
        from diplomat.utils import frame_store_fmt

        exporters = {
            "DLFS": frame_store_fmt.DLFSWriter
        }

        if(export_all):
            bp_list = [
                f"{bp}_{track}_{map_type.value}"
                for bp in frame_metadata.bodyparts
                for track in range(frame_metadata.num_outputs)
                for map_type in cls.ExportableFields
            ]
        else:
            bp_list = [
                f"{bp}_{track}" for bp in frame_metadata.bodyparts for track in range(frame_metadata.num_outputs)
            ]

        header = frame_store_fmt.DLFSHeader(
            num_frames,
            (frame_metadata.height + 2) if(export_all) else frame_metadata.height,
            (frame_metadata.width + 2) if(export_all) else frame_metadata.width,
            video_metadata["fps"],
            frame_metadata.down_scaling,
            *video_metadata["size"],
            *((None, None) if (video_metadata["cropping-offset"] is None) else video_metadata["cropping-offset"]),
            bp_list
        )

        return exporters.get(file_format, frame_store_fmt.DLFSWriter)(file, header)

    @classmethod
    def _export_frame(
        cls,
        src_frame: ForwardBackwardFrame,
        metadata: AttributeDict,
        dest_frame: TrackingData,
        dst_idx: Tuple[int, int],
        header: frame_store_api.DLFSHeader,
        selected_field: ExportableFields,
        padding: int = 0
    ):
        start, end = padding, -padding if(padding > 0) else None
        spc = padding * 2

        if(selected_field == cls.ExportableFields.SOURCE):
            res = src_frame.src_data.desparsify(
                header.frame_width - spc, header.frame_height - spc, header.stride
            )

            dest_frame.get_prob_table(*dst_idx)[start:end, start:end] = res.get_prob_table(0, 0)
        elif(selected_field == cls.ExportableFields.FRAME):
            data = src_frame.src_data.unpack()
            probs = src_frame.frame_probs
            if (probs is None):
                probs = np.zeros(len(data[2]), np.float32)

            res = SparseTrackingData()
            res.pack(*data[:2], probs, *data[3:])
            res = res.desparsify(header.frame_width - spc, header.frame_height - spc, header.stride)

            dest_frame.get_prob_table(*dst_idx)[start:end, start:end] = res.get_prob_table(0, 0)
        elif(selected_field == cls.ExportableFields.OCCLUDED_AND_EDGES):
            if(padding < 1):
                raise ValueError("Padding must be included to export edges and the occluded state!")

            probs = src_frame.occluded_probs

            o_x, o_y = tuple(src_frame.occluded_coords.T)
            x, y = o_x + 1, o_y + 1
            off_x = off_y = np.zeros(len(x), dtype=np.float32)

            res = SparseTrackingData()
            res.pack(y, x, probs, off_x, off_y)

            # Add 2 to resolve additional padding as needed for the edges...
            res = res.desparsify(header.frame_width - spc + 2, header.frame_height - spc + 2, header.stride)

            new_start = start - 1
            new_end = end + 1 if(end + 1 < 0) else None

            dest_frame.get_prob_table(*dst_idx)[new_start:new_end, new_start:new_end] = res.get_prob_table(0, 0)

    @classmethod
    def _export_frames(
        cls,
        frames: ForwardBackwardData,
        video_metadata: Config,
        path: Path,
        file_format: str,
        p_bar: Optional[ProgressBar] = None,
        export_final_probs: bool = True,
        export_all: bool = False
    ):
        """
        Private method, exports frames if the user specifies a frame export path.

        TODO
        """
        if(p_bar is not None):
            p_bar.reset(frames.num_frames)

        with path.open("wb") as f:
            with cls._get_frame_writer(
                    frames.num_frames, frames.metadata, video_metadata, file_format, f, export_all
            ) as fw:
                header = fw.get_header()

                for f_idx in range(frames.num_frames):
                    frame_data = TrackingData.empty_tracking_data(
                        1,
                        frames.num_bodyparts * (3 if(export_all) else 1),
                        header.frame_width,
                        header.frame_height,
                        frames.metadata.down_scaling,
                        True
                    )

                    for bp_idx in range(frames.num_bodyparts):
                        if(export_final_probs and frames.frames[f_idx][bp_idx].frame_probs is None):
                            continue

                        if(export_all):
                            for exp_t_i, exp_t in enumerate(cls.ExportableFields):
                                cls._export_frame(
                                    frames.frames[f_idx][bp_idx],
                                    frames.metadata,
                                    frame_data,
                                    (0, bp_idx * len(cls.ExportableFields) + exp_t_i),
                                    header,
                                    exp_t,
                                    padding=1
                                )
                        else:
                            # No padding...
                            cls._export_frame(
                                frames.frames[f_idx][bp_idx],
                                frames.metadata,
                                frame_data,
                                (0, bp_idx),
                                header,
                                cls.ExportableFields.FRAME if(export_final_probs) else cls.ExportableFields.SOURCE
                            )

                    fw.write_data(frame_data)

                    if(p_bar is not None):
                        p_bar.update()

    def _run_frame_passes(self, progress_bar: Optional[ProgressBar], fresh_run: bool = False):
        if(fresh_run):
            if(progress_bar is not None):
                progress_bar.message("Clearing Old ForwardBackward Data...")
                progress_bar.reset(self._frame_holder.num_frames)

            for f in range(self._frame_holder.num_frames):
                for bp in range(self._frame_holder.num_bodyparts):
                    frame = self._frame_holder.frames[f][bp]
                    frame.src_data = frame.orig_data.duplicate()
                    frame.frame_probs = None
                    frame.occluded_probs = None
                    frame.occluded_coords = None

                if(progress_bar is not None):
                    progress_bar.update()

        for i, frame_pass_builder in enumerate(self.PASSES):
            frame_pass = frame_pass_builder(self._width, self._height)

            if(progress_bar is not None):
                progress_bar.message(f"Running Frame Pass {i + 1}/{len(self.PASSES)}: '{frame_pass.get_name()}'")

            self._frame_holder = frame_pass.run_pass(
                self._frame_holder,
                progress_bar,
                True
            )

    def _on_end(self, progress_bar: ProgressBar) -> Optional[Pose]:
        self._run_frame_passes(progress_bar)

        if(self.EXPORT_LOC is not None):
            progress_bar.message(f"Exporting Frames to: '{str(self.EXPORT_LOC)}'")
            self._export_frames(
                self._frame_holder,
                self.video_metadata,
                self.EXPORT_LOC,
                "DLFS",
                progress_bar,
                self.settings.export_final_probs,
                self.settings.export_all_info
            )

        progress_bar.message("Selecting Maximums")
        return self.get_maximums(
            self._frame_holder, progress_bar,
            relaxed_radius=self.settings.relaxed_maximum_radius
        )


    @classmethod
    def get_settings(cls) -> ConfigSpec:
        return {
            "threshold": (
                0.001,
                type_casters.RangedFloat(0, 1),
                "The minimum floating point value a pixel within the probability frame must have "
                "in order to be kept and added to the sparse matrix."
            ),
            "max_cells_per_frame": (
                100,
                type_casters.Optional(type_casters.RangedInteger(1, np.inf)),
                "The maximum number of cells allowed in any frame. Defaults to None, meaning no strict limit is placed on cells"
                "per frame except the minimum threshold. Can be any positive integer, which will limit the number of cells in any"
                "frame score map to that value. Useful in cases where frames generated by models contain too many cells slowing "
                "computation."
            ),
            "passes": (
                [
                    "ClusterFrames",
                    "OptimizeStandardDeviation",
                    "CreateSkeleton",
                    "FixFrame",
                    "MITViterbi",
                ],
                type_casters.List(
                    type_casters.Union(
                        type_casters.Tuple(str, dict),
                        type_casters.Tuple(str),
                        str
                    )
                ),
                FPEString(
                    "A list of lists containing a string (the pass name) and a dictionary (the configuration for "
                    f"the provided plugin). If no configuration is provided, the entry can just be a string."
                )
            ),
            "export_frame_path": (
                None,
                type_casters.Union(type_casters.Literal(None), str),
                "A string or None specifying where to save the post forward backward frames to."
                "If None, does not save the frames to a file. Used for debugging."
            ),
            "export_final_probs": (
                True,
                bool,
                "If true exports the final probabilities as stored in frame_probs. "
                "Otherwise exports the probabilities from src_data. This setting is internal "
                "and for debugging. Defaults to true."
            ),
            "export_all_info": (
                False,
                bool,
                "If true exports all information, both final/pre-fpe probabilities and "
                "the occluded and edge states. This allows for display of several states at once. "
                "Only works if export_frame_path is set, and overrides export_final_probs."
            ),
            "relaxed_maximum_radius": (
                1.8,
                type_casters.RangedFloat(0, np.inf),
                "Determines the radius of relaxed maximum selection."
                "Set to 0 to disable relaxed maximum selection. This value is "
                "measured in cell units, not video units."
            ),
            "sparsification_mode": (
                SparseTrackingData.SparseModes.OFFSET_DOMINATION.name,
                type_casters.Literal(*[mode.name for mode in SparseTrackingData.SparseModes]),
                "The mode to utilize during sparsification."
            ),
            "dipui_file": (
                None, type_casters.Union(type_casters.Literal(None), str), 
                "A path specifying where to save the dipui file"
            ),
        }


    @classmethod
    def get_tests(cls) -> Optional[List[TestFunction]]:
        return [cls.test_plotting, cls.test_sparsification, cls.test_desparsification]

    @classmethod
    def get_test_data(cls) -> TrackingData:
        # Make tracking data...
        track_data = TrackingData.empty_tracking_data(4, 1, 3, 3, 2)

        track_data.set_prob_table(0, 0, np.array([[0, 0, 0],
                                                  [0, 1, 0],
                                                  [0, 0, 0]]))
        track_data.set_prob_table(1, 0, np.array([[0, 1.0, 0],
                                                  [0, 0.5, 0],
                                                  [0, 0.0, 0]]))
        track_data.set_prob_table(2, 0, np.array([[1, 0.5, 0],
                                                  [0, 0.0, 0],
                                                  [0, 0.0, 0]]))
        track_data.set_prob_table(3, 0, np.array([[0.5, 0, 0],
                                                  [1.0, 0, 0],
                                                  [0.0, 0, 0]]))

        return track_data

    @classmethod
    def get_test_instance(cls, track_data: TrackingData, settings: dict = None, num_out: int = 1) -> "FramePassEngine":
        return cls(
            [f"part{i + 1}" for i in range(track_data.get_bodypart_count())],
            num_out,
            track_data.get_frame_count(),
            Config(settings, cls.get_settings()),
            Config()
        )

    @classmethod
    def test_plotting(cls) -> Tuple[bool, str, str]:
        track_data = cls.get_test_data()
        predictor = cls.get_test_instance(track_data, {"passes": [
            "ClusterFrames",
            "FixFrame",
            ["MITViterbi", {"standard_deviation": 5}]
        ]})

        # Probabilities can change quite easily by even very minute changes to the algorithm, so we don't care about
        # them, just the predicted locations of things...
        expected_result = np.array([[3, 3], [3, 1], [1, 1], [1, 3]])

        # Pass it data...
        predictor.on_frames(track_data)

        # Check output
        poses = predictor.on_end(TQDMProgressBar(total=4)).get_all()

        if (np.allclose(poses[:, :2], expected_result)):
            return (True, "\n" + str(expected_result), "\n" + str(np.array(poses[:, :2])))
        else:
            return (False, "\n" + str(expected_result), "\n" + str(np.array(poses[:, :2])))

    @classmethod
    def test_sparsification(cls) -> Tuple[bool, str, str]:
        # Make tracking data...
        track_data = cls.get_test_data()
        predictor = cls.get_test_instance(track_data)

        # Pass it data...
        predictor.on_frames(track_data)

        # Check output
        predictor.on_end(TQDMProgressBar(total=4))

        fb_frames = [data[0].frame_probs for data in predictor._frame_holder.frames]
        orig_frames = [data[0].orig_data for data in predictor._frame_holder.frames]

        if ((None in orig_frames) or np.any([(f is None) for f in fb_frames])):
            return (False, str((fb_frames, orig_frames)), "No None Entries...")
        else:
            return (True, str((fb_frames, orig_frames)), "No None Entries...")

    @classmethod
    def test_desparsification(cls) -> Tuple[bool, str, str]:
        # Make tracking data...
        track_data = cls.get_test_data()
        orig_frame = track_data.get_prob_table(0, 0)
        result_frame = (
            SparseTrackingData.sparsify(track_data, 0, 0, 0.001)
                .desparsify(orig_frame.shape[1], orig_frame.shape[0], 8).get_prob_table(0, 0)
        )

        return (np.allclose(result_frame, orig_frame), str(orig_frame) + "\n", str(result_frame) + "\n")

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True