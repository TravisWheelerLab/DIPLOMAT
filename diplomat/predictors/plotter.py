# For types in methods
from typing import List, Optional
from diplomat.processing import *

# Used specifically by plugin...
import numpy as np
from matplotlib import pyplot
import matplotlib
from matplotlib import colors as mpl_colors
from diplomat.utils.colormaps import to_colormap
from pathlib import Path
import cv2


def optional_dict(val: Optional[dict]) -> dict:
    return {} if(val is None) else dict(val)


def codec_string(val: str) -> int:
    return cv2.VideoWriter_fourcc(*val)


class PlotterArgMax(Predictor):
    """
    Identical to :plugin:`~diplomat.predictors.ArgMax`, but plots probability frames in form of video to the user
    using matplotlib...
    """
    def __init__(
        self,
        bodyparts: List[str],
        num_outputs: int,
        num_frames: int,
        settings: Config,
        video_metadata: Config,
    ):
        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata)

        # Keeps track of how many frames
        self._current_frame = 0

        self._parts_set = set(self.bodyparts) if(settings.parts_to_plot is None) else (set(settings.parts_to_plot) & set(self.bodyparts))
        if(len(self._parts_set) == 0):
            raise ValueError("No parts selected to plot!")
        self._part_idx_list = list(filter(lambda v: self.bodyparts[v] in self._parts_set, range(len(self.bodyparts))))

        # Determines grid size of charts
        self._grid_width = int(np.ceil(np.sqrt(len(self._parts_set))))
        self._grid_height = int(np.ceil(len(self._parts_set) / self._grid_width))
        # Stores opencv video writer...
        self._vid_writer = None

        # Name of the video file to save to
        path = video_metadata["orig-video-path"]
        path = "Unknown" if(path is None) else path
        final_video_name = settings["video_name"].replace(
            "$VIDEO", Path(path).stem
        )
        self.VIDEO_PATH = str(
            (Path(video_metadata["output-file-path"]).parent) / final_video_name
        )

        settings.figure_args.update({"dpi": settings.dpi})

        # Build the subplots...
        if(settings["3d_projection"]):
            from mpl_toolkits.mplot3d import Axes3D
            axes_args = {"projection": "3d", **settings.axes_args}
            self._figure, self._axes = pyplot.subplots(
                self._grid_height,
                self._grid_width,
                subplot_kw=axes_args,
                **settings.figure_args,
                squeeze=False
            )
        else:
            self._figure, self._axes = pyplot.subplots(
                self._grid_height,
                self._grid_width,
                subplot_kw=settings.axes_args,
                **settings.figure_args,
                squeeze=False
            )
        # Hide all axis.....
        if not settings.axis_on:
            for ax in self._axes.flat:
                ax.axis("off")

    @staticmethod
    def _logify(arr: np.ndarray) -> np.ndarray:
        """
        Place the array in log scale, and then place the values between 0 and 1 using simple linear interpolation...
        """
        with np.errstate(divide="ignore"):
            arr_logged = np.log(arr)
            was_zero = np.isneginf(arr_logged)
            not_zero = ~was_zero
            low_val = np.min(arr_logged[not_zero])

            arr_logged[not_zero] = (
                np.abs(low_val) - np.abs(arr_logged[not_zero])
            ) / np.abs(low_val)
            arr_logged[was_zero] = 0

            return arr_logged

    def _get_offset_map_of(self, t: TrackingData, frame: int, bp: int, mode3d: bool = False) -> Optional[tuple]:
        if(t.get_offset_map() is None):
            return None

        stride = t.get_down_scaling()

        x_off, y_off = [
            t.get_offset_map()[frame, :, :, bp, i] / stride for i in range(2)
        ]

        dist = np.sqrt(x_off ** 2 + y_off ** 2)
        mask = dist > self.settings.arrow_displacement_threshold

        x_cent = np.arange(0, x_off.shape[1], 1)
        y_cent = np.arange(0, y_off.shape[0], 1)
        x_cent, y_cent = np.meshgrid(x_cent, y_cent)

        if(mode3d):
            z_cent = np.ones(y_cent.shape)
            z_off = np.zeros(y_cent.shape)
            return (x_cent[mask], y_cent[mask], z_cent[mask], x_off[mask], y_off[mask], z_off[mask])
        else:
            return (x_cent[mask], y_cent[mask], x_off[mask], y_off[mask])

    def _close(self):
        if(self._vid_writer is not None):
            self._vid_writer.release()

    def _on_frames(self, scmap: TrackingData) -> Pose:
        settings = self.settings
        vid_meta = self.video_metadata

        for frame in range(scmap.get_frame_count()):
            self._figure.suptitle(f"Frame: {frame}")
            # Plot all probability maps
            for bp, ax in zip(self._part_idx_list, self._axes.flat):
                ax.clear()
                if(not settings["3d_projection"]):
                    ax.set_aspect("equal")
                if(not settings.axis_on):
                    ax.axis("off")

                ax.set_title(
                    f"Bodypart: {self.bodyparts[bp]}"
                )

                if(settings["3d_projection"]):
                    x, y = (
                        np.arange(scmap.get_frame_width()),
                        np.arange(scmap.get_frame_height()),
                    )
                    x, y = np.meshgrid(x, y)
                    z = (
                        self._logify(scmap.get_prob_table(frame, bp))
                        if(settings.use_log_scale)
                        else scmap.get_prob_table(frame, bp)
                    )
                    ax.plot_surface(x, y, z, cmap=settings.colormap)
                    z_range = ax.get_zlim()[1] - ax.get_zlim()[0]
                    ax.set_zlim(
                        ax.get_zlim()[0],
                        ax.get_zlim()[0] + (z_range * settings.z_shrink_factor)
                    )

                    if(settings.display_offsets):
                        res = self._get_offset_map_of(scmap, frame, bp, True)
                        if(res is not None):
                            ax.quiver(
                                *res,
                                color=settings.arrow_color,
                                arrow_length_ratio=0,
                                normalize=False,
                                **settings.arrow_args
                            )
                else:
                    ax.imshow(
                        self._logify(scmap.get_prob_table(frame, bp))
                        if(settings.use_log_scale)
                        else scmap.get_prob_table(frame, bp),
                        cmap=settings.colormap,
                        origin="lower"
                    )

                    if(settings.display_offsets):
                        res = self._get_offset_map_of(scmap, frame, bp)
                        if(res is not None):
                            ax.quiver(
                                *res,
                                angles='xy',
                                scale_units='xy',
                                scale=1,
                                color=settings.arrow_color,
                                **settings.arrow_args
                            )
                # This reverses the y-axis data, so as probability maps match the video...
                ax.set_ylim(ax.get_ylim()[::-1])

            # Save chart to the buffer
            if(self._current_frame == 0):
                self._figure.tight_layout()
            self._figure.canvas.draw()

            img = np.reshape(
                np.frombuffer(self._figure.canvas.tostring_rgb(), dtype="uint8"),
                self._figure.canvas.get_width_height()[::-1] + (3,),
            )[:, :, ::-1]

            if self._vid_writer is None:
                height, width, colors = img.shape
                self._vid_writer = cv2.VideoWriter(
                    self.VIDEO_PATH, settings.codec, vid_meta.fps, (width, height)
                )

            if(not self._vid_writer.isOpened()):
                raise IOError("Error occurred causing the video writer to close.")
            self._vid_writer.write(img)
            self._current_frame += 1

        return scmap.get_poses_for(
            scmap.get_max_scmap_points(num_max=self.num_outputs)
        )

    def _on_end(self, progress_bar: ProgressBar) -> Optional[Pose]:
        return None

    @classmethod
    def get_settings(cls) -> ConfigSpec:
        return {
            "video_name": (
                "$VIDEO-prob-dlc.mp4",
                str,
                "Name of the video file that plotting data will be saved to. "
                "Can use $VIDEO to place the name of original video somewhere "
                "in the text."
            ),
            "parts_to_plot": (
                None,
                type_casters.Union(type_casters.Literal(None), type_casters.List(str)),
                "A list of body parts to plot. None means plot all the body parts."
            ),
            "codec": (
                "mp4v",
                codec_string,
                "The codec to be used by the opencv library to save info to, "
                "typically a 4-byte string."
            ),
            "use_log_scale": (
                False,
                bool,
                "Boolean, determines whether to apply log scaling to the "
                "frames in the video."
            ),
            "3d_projection": (
                False,
                bool,
                "Boolean, determines if probability frames should be plotted in 3d."
            ),
            "display_offsets": (
                False,
                bool,
                "Boolean, determines if offset vectors are displayed additionally."
            ),
            "arrow_color": (
                "red",
                mpl_colors.to_rgba,
                "Matplotlib color type, the color of the offset arrows if enabled."
            ),
            "arrow_args": (
                None,
                optional_dict,
                "A dictionary, miscellaneous arguments to pass to plt.quiver."
            ),
            "arrow_displacement_threshold": (
                0.1,
                type_casters.RangedFloat(0, np.inf),
                "A float, the threshold under which not to plot an arrow for "
                "a given location."
            ),
            "colormap": (
                "Blues",
                to_colormap,
                "String, determines the underlying colormap to be passed to "
                "matplotlib while plotting the heatmap or mesh."
            ),
            "z_shrink_factor": (
                5,
                type_casters.RangedFloat(1, np.inf),
                "Float, determines how much to shrink the z-axis if in 3D mode..."
            ),
            "dpi": (
                matplotlib.rcParams["figure.dpi"],
                type_casters.RangedInteger(1, np.inf),
                "The dpi of the final video, the higher the dpi the more crisp..."
            ),
            "axis_on": (
                False,
                bool,
                "Boolean, determines if axis, or tick marks and grids of subplots "
                "are shown."
            ),
            "axes_args": (
                None,
                optional_dict,
                "A dictionary, miscellaneous arguments to pass to matplotlib "
                "axes when constructing them."
            ),
            "figure_args": (
                None,
                optional_dict,
                "A dictionary, miscellaneous arguments to pass to matplotlib "
                "figure when constructing it.",
            ),
        }

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True
