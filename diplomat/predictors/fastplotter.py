from typing import List, Optional
import cv2
import math
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import as_strided
from diplomat.processing import *


def optional_dict(val: Optional[dict]) -> dict:
    return {} if(val is None) else dict(val)


def codec_string(val: str) -> int:
    return cv2.VideoWriter_fourcc(*val)


_CV2_FONTS = {
    item: getattr(cv2, item)
    for item in dir(cv2)
    if (item.startswith("FONT_HERSHEY"))
}

_CV2_COLORMAPS = {
    item: getattr(cv2, item)
    for item in dir(cv2)
    if (item.startswith("COLORMAP"))
}


def cv2_font(val: str) -> int:
    return _CV2_FONTS[str(val)]


def cv2_colormap(val: str) -> int:
    return _CV2_COLORMAPS[str(val)]


bgr_color = type_casters.Tuple(
    *([type_casters.RangedInteger(0, 255)] * 3)
)


class FastPlotterArgMax(Predictor):
    """
    Identical to :py:plugin:`~diplomat.predictors.PlotterArgMax`, but avoids using matplotlib to generate probability
    maps, and instead directly uses cv2 to generate the plots. This means it runs much faster,
    but doesn't offer as much customization nor a 3D mode...
    """

    TEST_TEXT = "".join(chr(i) for i in range(32, 127))

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

        self._parts_set = set(self.bodyparts) if (settings.parts_to_plot is None) else (set(settings.parts_to_plot) & set(self.bodyparts))
        if (len(self._parts_set) == 0):
            raise ValueError("No parts selected to plot!")
        self._part_idx_list = list(filter(lambda v: self.bodyparts[v] in self._parts_set, range(len(self.bodyparts))))

        # Determines grid size of charts
        self._grid_width = int(math.ceil(math.sqrt(len(self._parts_set))))
        self._grid_height = int(math.ceil(len(self._parts_set) / self._grid_width))
        # Stores opencv video writer...
        self._vid_writer: Optional[cv2.VideoWriter] = None

        # Name of the video file to save to
        video_metadata["orig-video-path"] = "Unknown" if(video_metadata["orig-video-path"] is None) else video_metadata["orig-video-path"]
        final_video_name = settings["video_name"].replace(
            "$VIDEO", Path(video_metadata["orig-video-path"]).stem
        )

        self.VIDEO_PATH = str(
            (Path(video_metadata["output-file-path"]).parent) / final_video_name
        )

        # Compute the height of the titles and subtitles...
        (__, self._title_font_height), self._title_baseline = cv2.getTextSize(
            self.TEST_TEXT,
            settings.title_font,
            settings.title_font_size,
            settings.font_thickness
        )
        (__, self._subplot_font_height), self._subplot_baseline = cv2.getTextSize(
            self.TEST_TEXT,
            settings.subplot_font,
            settings.subplot_font_size,
            settings.font_thickness,
        )

        # Variable for math later when computing locations of things in the video...
        self._vid_height = None
        self._vid_width = None
        self._subplot_height = None
        self._subplot_width = None
        self._scmap_height = None
        self._scmap_width = None
        self._canvas = None  # The numpy array we will use for drawing...
        # Will store colormap per run to avoid reallocating large arrays over and over....
        self._colormap_temp = None
        self._colormap_view = None

    def _close(self):
        if(self._vid_writer is not None):
            self._vid_writer.release()

    def _compute_video_measurements(self, scmap_width: int, scmap_height: int):
        """
        Compute all required measurements needed to render text/source maps to the correct locations, and also
        initialize the video writer...
        """
        mult = self.settings.source_map_upscale
        padding = self.settings.padding

        self._scmap_width = scmap_width * mult
        self._scmap_height = scmap_height * mult

        self._subplot_width = (padding * 2) + self._scmap_width
        total_subplot_text_height = self._subplot_font_height + self._subplot_baseline
        self._subplot_height = (
            (padding * 2) + total_subplot_text_height + self._scmap_height
        )

        self._vid_width = self._grid_width * self._subplot_width
        self._vid_height = (self._grid_height * self._subplot_height) + (
            self._title_font_height + self._title_baseline
        )

        self._canvas = np.zeros((self._vid_height, self._vid_width, 3), dtype=np.uint8)

        self._vid_writer = cv2.VideoWriter(
            self.VIDEO_PATH,
            self.settings.codec,
            self.video_metadata.fps,
            (self._vid_width, self._vid_height),
        )
        # Array which stores color maps temporarily... Takes advantage of numpy's abilities to make custom strides
        # to access data... The colormap_view maps the
        self._colormap_temp = np.zeros(
            (self._scmap_height, self._scmap_width, 3), dtype=np.uint8
        )
        shape, strides = self._colormap_temp.shape, self._colormap_temp.strides
        view_shape = (
            mult,
            mult,
            shape[0] // mult,
            shape[1] // mult,
            shape[2],
        )
        view_strides = (
            strides[0],
            strides[1],
            strides[0] * mult,
            strides[1] * mult,
            strides[2],
        )
        self._colormap_view = as_strided(
            self._colormap_temp, shape=view_shape, strides=view_strides
        )
        self._unscaled_cmap_temp = np.zeros(
            (scmap_height, scmap_width, 3), dtype=np.uint8
        )

    @staticmethod
    def _probs_to_grayscale(arr: np.ndarray) -> np.ndarray:
        """
        Convert numpy probability array into a grayscale image of unsigned 8 bit integers.
        """
        return (arr * 255).astype(dtype=np.uint8)

    @staticmethod
    def _logify(arr: np.ndarray) -> np.ndarray:
        """
        Place the array in log scale, and then place the values between 0 and 1 using simple linear interpolation...
        """
        # Old code....
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

    @staticmethod
    def _normalize_range(arr: np.ndarray) -> np.ndarray:
        """
        Place values in the array using the range 0-1...
        """
        return arr / np.nanmax(arr)

    def _draw_title(self, text: str):
        """
        Draws the title text to the video frame....
        """
        s = self.settings

        (width, __), __ = cv2.getTextSize(
            text, s.title_font, s.title_font_size, s.font_thickness
        )
        x_in = max(int((self._vid_width - width) / 2), 0)
        y_in = (self._title_font_height + self._title_baseline) - 1
        cv2.putText(
            self._canvas,
            text,
            (x_in, y_in),
            s.title_font,
            s.title_font_size,
            s.title_font_color,
            s.font_thickness,
        )

    def _draw_subplot(
        self, bp_name: str, grid_x: int, grid_y: int, prob_map: np.ndarray
    ):
        """
        Draws a single subplot for the provided frame...

        :param bp_name: The name of the body part...
        :param grid_x: The x grid location to draw this probability map at.
        :param grid_y: The y grid location to draw this probability map at.
        :param prob_map: The probability map, an array of 2D floats...
        """
        s = self.settings

        x_upper_corner = grid_x * self._subplot_width
        y_upper_corner = (grid_y * self._subplot_height) + (
            self._title_font_height + self._title_baseline
        )

        # Convert probabilities to a color image...
        grayscale_img = self._probs_to_grayscale(
            self._logify(self._normalize_range(prob_map)) if (s.use_log_scale) else self._normalize_range(prob_map)
        )
        self._colormap_view[:, :] = cv2.applyColorMap(
            grayscale_img, s.colormap, self._unscaled_cmap_temp
        )
        # Insert the probability map...
        subplot_top_x, subplot_top_y = (
            (x_upper_corner + s.padding) - 1,
            (y_upper_corner + s.padding) - 1,
        )
        subplot_bottom_x, subplot_bottom_y = (
            subplot_top_x + self._scmap_width,
            subplot_top_y + self._scmap_height,
        )
        self._canvas[
            subplot_top_y:subplot_bottom_y, subplot_top_x:subplot_bottom_x
        ] = self._colormap_temp
        # Now insert the text....
        (text_width, __), __ = cv2.getTextSize(
            bp_name, s.subplot_font, s.subplot_font_size, s.font_thickness
        )
        x_text_root = x_upper_corner + max(
            int((self._subplot_width - text_width) / 2), 0
        )
        y_text_root = y_upper_corner + (self._subplot_height - s.padding)
        cv2.putText(
            self._canvas,
            bp_name,
            (x_text_root, y_text_root),
            s.subplot_font,
            s.subplot_font_size,
            s.subplot_font_color,
            s.font_thickness,
        )

    def _on_frames(self, scmap: TrackingData) -> Optional[Pose]:
        # If the video writer has not been created, create it now and compute all needed video dimensions...
        if self._vid_writer is None:
            self._compute_video_measurements(
                scmap.get_frame_width(), scmap.get_frame_height()
            )

        for frame in range(scmap.get_frame_count()):
            # Clear the canvas with the background color...
            self._canvas[:] = self.settings.background_color
            # Drawing the title...
            self._draw_title(f"Frame {self._current_frame}")

            for i, bp in enumerate(self._part_idx_list):
                # Compute the current subplot we are on...
                subplot_y = i // self._grid_width
                subplot_x = i % self._grid_width
                self._draw_subplot(
                    self.bodyparts[bp],
                    subplot_x,
                    subplot_y,
                    scmap.get_prob_table(frame, bp),
                )

            if(not self._vid_writer.isOpened()):
                raise IOError("Error occurred causing the video writer to close.")
            self._vid_writer.write(self._canvas)

            self._current_frame += 1

        # Return just like argmax...
        return scmap.get_poses_for(
            scmap.get_max_scmap_points(num_max=self.num_outputs)
        )

    def _on_end(self, progress_bar: ProgressBar) -> Optional[Pose]:
        return None

    @classmethod
    def get_settings(cls) -> ConfigSpec:
        font_options = "\n".join([f"\t - {key}" for key in _CV2_FONTS])
        colormap_options = "\n".join(f"\t - {key}" for key in _CV2_COLORMAPS)

        return {
            "video_name": (
                "$VIDEO-fast-prob-dlc.mp4",
                str,
                "Name of the video file that plotting data will be saved to. Can use $VIDEO to place the "
                "name of original video somewhere in the text."
            ),
            "parts_to_plot": (
                None,
                type_casters.Union(type_casters.Literal(None), type_casters.List(str)),
                "A list of body parts to plot. None means plot all the body parts."
            ),
            "codec": (
                "mp4v",
                codec_string,
                "The codec to be used by the opencv library to save info to, typically a 4-byte string."
            ),
            "use_log_scale": (
                False,
                bool,
                "Boolean, determines whether to apply log scaling to the frames in the video."
            ),
            "title_font_size": (
                2,
                type_casters.RangedFloat(0.1, np.inf),
                "The font size of the main title"
            ),
            "title_font": (
                "FONT_HERSHEY_SIMPLEX",
                cv2_font,
                f"String, the cv2 font to be used in the title, options for this are:\n{font_options}"
            ),
            "subplot_font_size": (
                1.5,
                type_casters.RangedFloat(0.1, np.inf),
                "The font size of the titles of each subplot."
            ),
            "subplot_font": (
                "FONT_HERSHEY_SIMPLEX",
                cv2_font,
                "String, the cv2 font used in the subplot titles, look at options for 'title_font'."
            ),
            "background_color": (
                (255, 255, 255),
                bgr_color,
                "Tuple of 3 integers, color of the background in BGR format"
            ),
            "title_font_color": (
                (0, 0, 0),
                bgr_color,
                "Tuple of 3 integers, color of the title text in BGR format"
            ),
            "subplot_font_color": (
                (0, 0, 0),
                bgr_color,
                "Tuple of 3 integers, color of the title text in BGR format"
            ),
            "colormap": (
                "COLORMAP_VIRIDIS",
                cv2_colormap,
                f"String, the cv2 colormap to use, options for this are:\n{colormap_options}"
            ),
            "font_thickness": (
                2,
                type_casters.RangedInteger(1, np.inf),
                "Integer, the thickness of the font being drawn."
            ),
            "source_map_upscale": (
                4,
                type_casters.RangedInteger(1, 100),
                "Integer, The amount to upscale the probability maps."
            ),
            "padding": (
                20,
                type_casters.RangedInteger(1, np.inf),
                "Integer, the padding to be applied around plots in pixels."
            )
        }

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True



