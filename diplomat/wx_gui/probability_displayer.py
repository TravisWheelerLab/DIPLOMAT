"""
Provides a plotting widget, which displays a filled line graph. Used for displaying metrics at the bottom of the UI.
"""

from enum import IntEnum
from typing import Iterable, NamedTuple, Optional
import wx
import numpy as np


class DrawMode(IntEnum):
    NORMAL = 0
    POORLY_LABELED = 1
    USER_MODIFIED = 2
    USER_MODIFIED_AND_POORLY_LABELED = 3


class DrawCommand(NamedTuple):
    draw_mode: DrawMode
    points: np.ndarray
    point_before: np.ndarray
    point_after: np.ndarray


class DrawingInfo(NamedTuple):
    x_center: float
    y_center: float
    center_draw_mode: DrawMode
    segment_xs: Iterable[int]
    segment_fix_xs: Iterable[int]
    draw_commands: Iterable[DrawCommand]


class ProbabilityDisplayer(wx.Control):
    """
    A custom wx.Control which displays a list of probabilities in the form of a line segment plot. Uses native colors
    so as to match other native widgets in the UI.
    """
    # Minimum pixels between probabilities....
    MIN_PROB_STEP = 10
    # The number of probabilities to default to displaying on the screen...
    VISIBLE_PROBS = 100
    # Default height, pointer triangle size, and padding values in pixels...
    DEF_HEIGHT = 50
    TRIANGLE_SIZE = 7
    TOP_PADDING = 3

    def __init__(
        self,
        parent,
        data: np.ndarray = None,
        bad_locations: np.ndarray = None,
        text: str = None,
        height: int = DEF_HEIGHT,
        visible_probs: int = VISIBLE_PROBS,
        style=wx.BORDER_DEFAULT,
        name="ProbabilityDisplayer",
        **kwargs
    ):
        """
        Construct a new ProbabilityDisplayer....

        :param parent: The parent widget.
        :param data: The probability data, a 1D numpy array of number type values.
        :param text: The text to display in the top left corner of this probability display, or None to display no text.
        :param w_id: wx ID of the window, and integer. Defaults to wx.ID_ANY.
        :param height: The minimum height of the probability display. Defaults to 50 pixels.
        :param visible_probs: The max number of probabilities to show on screen at once. Defaults to 100.
        :param pos: WX Position of control. Defaults to wx.DefaultPosition.
        :param size: WX Size of the control. Defaults to wx.DefaultSize.
        :param style: WX Control Style. See wx.Control docs for possible options. (Defaults to wx.BORDER_DEFAULT).
        :param validator: WX Validator, defaults to
        :param name: WX internal name of widget.
        """
        super().__init__(parent, style=style | wx.FULL_REPAINT_ON_RESIZE, name=name, **kwargs)
        # This tell WX that we are going to handle background painting ourselves, disabling system background clearing
        # and avoiding glitchy rendering and flickering...
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        if((len(data.shape) != 1)):
            raise ValueError("Invalid data! Must be a numpy array of 1 dimension...")

        self._data = np.copy(data)
        self._bad_locations = bad_locations.astype(np.uint64)
        self._user_modified_from_last_pass = np.array([], dtype=np.uint64)
        self._max_data_point = np.nanmax(self._data)
        self._refresh_bad_locations()
        self._ticks_visible = visible_probs

        self._segment_starts = None
        self._segment_fix_frames = None

        self._best_size = wx.Size(self.MIN_PROB_STEP * 5, max(height, (self.TRIANGLE_SIZE * 4) + self.TOP_PADDING))
        self.SetMinSize(self._best_size)
        self.SetInitialSize(self._best_size)

        self._current_index = 0

        self._text = text

        # Rig up paint event, and also disable erase background event...
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda evt: None)

    def DoGetBestSize(self):
        return self._best_size

    def on_paint(self, event: wx.PaintEvent):
        """
        PRIVATE: Triggered on a wx paint event, redraws the probability display...
        """
        # If the platform already uses double buffering, use a plain old PaintDC, otherwise use a BufferedPaintDC to
        # avoid flickering on unbuffered platforms....
        painter = wx.PaintDC(self) if(self.IsDoubleBuffered()) else wx.BufferedPaintDC(self)
        # Using a GCDC allows for much prettier aliased painting, making plot look nicer.
        painter = wx.GCDC(painter)
        self.on_draw(painter)

    @staticmethod
    def _is_touched(idx, bad_labels, old_user_mods):
        if(len(old_user_mods) == 0):
            return False

        idx2 = np.searchsorted(old_user_mods, idx)
        low_goal = old_user_mods[max(0, idx2 - 1)]
        high_goal = old_user_mods[min(idx2, len(old_user_mods) - 1)]

        low_idx = np.searchsorted(bad_labels, low_goal)
        high_idx = np.searchsorted(bad_labels, high_goal)
        mid_idx = np.searchsorted(bad_labels, idx)

        low_value = int(bad_labels[min(low_idx, len(bad_labels) - 1)])
        high_value = int(bad_labels[min(high_idx, len(bad_labels) - 1)])
        mid_value = int(bad_labels[min(mid_idx, len(bad_labels) - 1)])

        low_gap_match = low_value == low_goal and mid_idx - low_idx == mid_value - low_value
        high_gap_match = high_value == high_goal and high_idx - mid_idx == high_value - mid_value

        return (mid_value == idx) and (low_gap_match or high_gap_match)

    @classmethod
    def _get_draw_commands(
        cls, x_arr, y_arr, mode_arr, low_val, bad_labels, old_user_mods
    ) -> Iterable[DrawCommand]:
        # We compute islands by finding locations where the pairwise
        # difference of modes array is non-zero (Indicating mode change)
        change_locs = np.flatnonzero(mode_arr[1:] - mode_arr[:-1]) + 1
        change_locs = np.concatenate(([0], change_locs, [len(mode_arr)]))

        for start, end in zip(change_locs[:-1], change_locs[1:]):
            before_idx = max(start - 1, 0)
            after_idx = min(end, len(x_arr) - 1)

            if(np.isnan(x_arr[before_idx]) or np.isnan(y_arr[before_idx])):
                before_idx = start
            if(np.isnan(x_arr[after_idx]) or np.isnan(y_arr[after_idx])):
                after_idx -= 1

            mode = DrawMode(mode_arr[start])
            if(mode_arr[start] == DrawMode.POORLY_LABELED):
                if(cls._is_touched(low_val + start, bad_labels, old_user_mods)):
                    mode = DrawMode.USER_MODIFIED_AND_POORLY_LABELED

            yield DrawCommand(
                mode,
                np.stack([x_arr[start:end], y_arr[start:end]], -1),
                np.array([x_arr[before_idx], y_arr[before_idx]]),
                np.array([x_arr[after_idx], y_arr[after_idx]])
            )

    def _compute_points(self, height: int, width: int) -> DrawingInfo:
        """
        PRIVATE: Computes the points to be rendered to the screen given the probability data.

        :param height: The height of the control.
        :param width: The width of the control.

        :returns: An iterable of tuples of (str, int, int, numpy array), which represent:
                      -
                      - The center of the widget horizontally.
                      - The current highlighted index or selected index within the point list returned.
                      - A numpy array of shape (N, 2). Representing the X, Y locations of points. These can be directly
                        drawn to the widget.
        """
        data = self._data
        # Compute the amount of probabilities to display per side based on configured parameters...
        tick_step = max(self.MIN_PROB_STEP, int(width / self._ticks_visible))

        center = (width // 2)
        values_per_side = (center - 1) // tick_step

        # Compute the lowest and highest indexes for probabilities we can show...
        low_val = max(self._current_index - values_per_side, 0)
        high_val = min(self._current_index + values_per_side + 1, len(data))

        # Points are distributed evenly by tick_step on x-axis.... On y axis we set there value by interpolating
        # between the available space for the probabilities within the widget.
        offset = center - ((self._current_index - low_val) * tick_step)

        # Identify "bad" locations as they'll be drawn differently...
        low_bad = np.searchsorted(self._bad_locations, low_val)
        high_bad = np.searchsorted(self._bad_locations, high_val)
        bad_locations = self._bad_locations[low_bad:high_bad] - low_val

        # If there are segments, identify what segments we can see...
        if(self._segment_starts is not None):
            seg_low = np.searchsorted(self._segment_starts[1:], low_val)
            seg_high = np.searchsorted(self._segment_starts[1:], high_val)
            seg_offsets = (self._segment_starts[1:][seg_low:seg_high] - low_val) * tick_step + offset - (tick_step / 2)
        else:
            seg_offsets = np.array([])

        # If there are segments, identify what segments we can see...
        if(self._segment_fix_frames is not None):
            seg_low = np.searchsorted(self._segment_fix_frames, low_val)
            seg_high = np.searchsorted(self._segment_fix_frames, high_val)
            seg_fix_offsets = (self._segment_fix_frames[seg_low:seg_high] - low_val) * tick_step + offset
        else:
            seg_fix_offsets = np.array([])

        x = np.arange(0, high_val - low_val) * tick_step + offset
        y = data[low_val:high_val]
        y = (1 - (y / self._max_data_point)) * (height - ((self.TRIANGLE_SIZE * 2) + self.TOP_PADDING)) + self.TOP_PADDING

        # Build a mode array.
        mode = np.zeros(len(y), dtype=np.int8)
        mode[bad_locations] = DrawMode.POORLY_LABELED
        mode[np.isnan(y)] = DrawMode.USER_MODIFIED

        return DrawingInfo(
            int(center),
            y[self._current_index - low_val],
            DrawMode(mode[self._current_index - low_val]),
            seg_offsets.astype(int),
            seg_fix_offsets.astype(int),
            self._get_draw_commands(
                x, y, mode, low_val, self._bad_locations, self._user_modified_from_last_pass
            )
        )

    def on_draw(self, dc: wx.DC):
        """
        For internal use! Executed on drawing update, redraws the probability display. Expects a wx.DC for drawing
        to.
        """
        width, height = self.GetClientSize()

        if((not width) or (not height)):
            return

        # Clear the background with the default color...
        dc.SetBackground(
            wx.Brush(self.GetBackgroundColour(), wx.BRUSHSTYLE_SOLID)
        )
        dc.Clear()

        # Colors used in pens and brushes below...
        highlight_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT)
        highlight_color2 = wx.Colour(
            *highlight_color[:3],
            int(highlight_color.Alpha() * 0.3)
        )
        # WX widgets doesn't provide an error highlight color. Since the
        # highlight color doesn't typically match the foreground or
        # background, we take the complement of it as a second selection color
        # (This color happens to usually be a Blue, so this typically produces
        #  a Red/Orange)
        error_color = wx.Colour(
            255 - highlight_color.Red(),
            255 - highlight_color.Green(),
            255 - highlight_color.Blue(),
            highlight_color.Alpha()
        )
        error_color2 = wx.Colour(*error_color[:3], int(error_color.Alpha() * 0.3))

        foreground_color = self.GetForegroundColour()

        fixed_error_color = wx.Colour(*(((
            np.asarray(self.GetBackgroundColour(), int)
            + np.asarray(foreground_color, int)
        ) / 2).astype(int)))
        fixed_error_color2 = wx.Colour(
            *fixed_error_color[:3], int(fixed_error_color.Alpha() * 0.3)
        )

        # All the pens and brushes we will need...
        transparent_pen = wx.Pen(highlight_color, 2, wx.PENSTYLE_TRANSPARENT)

        # Primary highlight color (normal plot locations)...
        highlight_pen = wx.Pen(highlight_color, 2, wx.PENSTYLE_SOLID)
        highlight_pen2 = wx.Pen(highlight_color, 5, wx.PENSTYLE_SOLID)
        highlight_brush = wx.Brush(highlight_color2, wx.BRUSHSTYLE_SOLID)

        error_pen = wx.Pen(error_color, 2, wx.PENSTYLE_SOLID)
        error_pen2 = wx.Pen(error_color, 5, wx.PENSTYLE_SOLID)
        error_brush = wx.Brush(error_color2, wx.BRUSHSTYLE_SOLID)

        fixed_error_pen = wx.Pen(fixed_error_color, 2, wx.PENSTYLE_SOLID)
        fixed_error_pen2 = wx.Pen(fixed_error_color, 5, wx.PENSTYLE_SOLID)
        fixed_error_brush = wx.Brush(fixed_error_color2, wx.BRUSHSTYLE_SOLID)

        indicator_brush = wx.Brush(foreground_color, wx.BRUSHSTYLE_SOLID)
        indicator_pen = wx.Pen(foreground_color, 5, wx.PENSTYLE_SOLID)
        indicator_pen2 = wx.Pen(foreground_color, 1, wx.PENSTYLE_SOLID)

        # This patches the point drawing for the latest versions of wxWidgets, which don't respect the pen's width correctly...
        def draw_points(points, pen):
            top = np.round((np.asarray(points) - pen.GetWidth() / 2)).astype(int)
            args = np.concatenate([top, np.full(top.shape, pen.GetWidth(), dtype=int)], axis=-1)
            dc.DrawEllipseList(args, transparent_pen, wx.Brush(pen.GetColour(), wx.BRUSHSTYLE_SOLID))

        # Compute the center and points to place on the line...
        draw_info = self._compute_points(height, width)

        for seg_x in draw_info.segment_xs:
            seg_x = int(seg_x)
            dc.DrawLineList([[seg_x, 0, seg_x, height]], fixed_error_pen)
            dc.DrawPolygonList([[
                [seg_x - int(self.TRIANGLE_SIZE / 2), 0],
                [seg_x + int(self.TRIANGLE_SIZE / 2), 0],
                [seg_x, int(self.TRIANGLE_SIZE)]
            ]], fixed_error_pen, fixed_error_brush)

        for seg_x in draw_info.segment_fix_xs:
            seg_x = int(seg_x)
            dc.DrawPolygonList([[
                [seg_x - int(self.TRIANGLE_SIZE / 2), 0],
                [seg_x + int(self.TRIANGLE_SIZE / 2), 0],
                [seg_x, int(self.TRIANGLE_SIZE)]
            ]], indicator_pen2, highlight_brush)

        # Plot all of the points the filled-in polygon underneath, and the line connecting the points...
        for draw_command in draw_info.draw_commands:
            if(draw_command.draw_mode == DrawMode.USER_MODIFIED):
                continue

            if(draw_command.draw_mode == DrawMode.NORMAL):
                pen = highlight_pen
                pen2 = highlight_pen2
                brush = highlight_brush
            elif(draw_command.draw_mode == DrawMode.POORLY_LABELED):
                pen = error_pen
                pen2 = error_pen2
                brush = error_brush
            else:
                pen = fixed_error_pen
                pen2 = fixed_error_pen2
                brush = fixed_error_brush

            poly_begin_point = (draw_command.points[0] + draw_command.point_before) / 2
            poly_end_point = (draw_command.points[-1] + draw_command.point_after) / 2

            wrap_polygon_points = np.array([
                poly_end_point,
                [poly_end_point[0], height],
                [poly_begin_point[0], height],
                poly_begin_point
            ])

            dc.DrawPolygonList(
                [np.concatenate((draw_command.points, wrap_polygon_points))],
                transparent_pen,
                brush
            )

            all_points = np.concatenate(
                ([poly_begin_point], draw_command.points, [poly_end_point])
            )
            dc.DrawLineList(np.concatenate((all_points[1:], all_points[:-1]), 1).astype(int), pen)
            draw_points(draw_command.points.astype(int), pen2)

        # Draw the current location indicating line, point and arrow, indicates which data point we are currently on.
        dc.DrawLineList([[int(draw_info.x_center), 0, int(draw_info.x_center), height]], indicator_pen2)
        dc.DrawPolygonList([[
            [int(draw_info.x_center - self.TRIANGLE_SIZE), height],
            [int(draw_info.x_center + self.TRIANGLE_SIZE), height],
            [int(draw_info.x_center), height - int(self.TRIANGLE_SIZE * 1.5)]
        ]], indicator_pen2, indicator_brush)
        if(draw_info.center_draw_mode != DrawMode.USER_MODIFIED):
            draw_points([[int(draw_info.x_center), int(draw_info.y_center)]], indicator_pen)

        # If the user set the name of this probability display plot, write it to the top-left corner...
        if(self._text is not None):
            back_pen = wx.Pen(self.GetBackgroundColour(), 3, wx.PENSTYLE_SOLID)
            back_brush = wx.Brush(self.GetBackgroundColour(), wx.BRUSHSTYLE_SOLID)
            dc.SetTextBackground(self.GetBackgroundColour())
            dc.SetTextForeground(self.GetForegroundColour())

            dc.SetFont(self.GetFont())
            size: wx.Size = dc.GetTextExtent(self._text)
            width, height = size.GetWidth(), size.GetHeight()
            dc.DrawRectangleList([(0, 0, width, height)], back_pen, back_brush)
            dc.DrawText(self._text, 0, 0)

    def set_location(self, location: int):
        """
        Set the current location of the probability display, or the index to which the arrow is pointing.

        :param location: A integer, being the frame or index to make this probability display center and point to.
        """
        if(not (0 <= location < self._data.shape[0])):
            raise ValueError(f"Location {location} is not within the range: 0 through {self._data.shape[0]}.")
        self._current_index = location
        self.Refresh()

    def get_location(self) -> int:
        """
        Get the current location of the probability display, or the index to which the arrow is pointing.

        :returns: A integer, being the frame or index of the currently pointed to location.
        """
        return self._current_index

    def set_data(self, data: np.ndarray):
        """
        Set all of the data.

        :param data: Numpy array of numbers, will be copied over into the
                     internal data store and displayed on next
                     redraw.
        """
        self._data[:] = data
        self._max_data_point = np.nanmax(self._data)
        self._refresh_bad_locations()
        self.Refresh()

    def get_data(self) -> np.ndarray:
        """
        Get all the data.

        :returns: Numpy array of numbers, the data of this probability display.
                  The returned array is a read only view...
        """
        view = self._data.view()
        view.flags.writeable = False
        return view

    def set_data_at(self, frame: int, value: float):
        """
        Set the data at the given frame to the given value.

        :param frame: The frame or index to set the data value at.
        :param value: A float or number, the value to assign at the data point.
        """
        self._data[frame] = value
        self._max_data_point = np.nanmax([self._max_data_point, value])
        self._refresh_bad_locations()
        self.Refresh()

    def get_data_at(self, frame: int) -> float:
        """
        Get the data at a given index or frame within the probability display.

        :param frame: The index or frame to get the probability data of.

        :returns: A float, being the data at the given frame.
        """
        return self._data[frame]

    def _refresh_bad_locations(self):
        # Remove nan values...
        self._bad_locations = self._bad_locations[
            np.isfinite(self._data[self._bad_locations])
        ]

    def get_user_modified_locations(self) -> np.ndarray:
        """
        Get the current user modified locations...
        """
        return np.flatnonzero(np.isnan(self._data))

    def set_prior_modified_user_locations(self, value: np.ndarray):
        self._user_modified_from_last_pass = np.asarray(value, dtype=np.uint64)

    def get_prior_modified_user_locations(self) -> np.ndarray:
        return self._user_modified_from_last_pass

    def set_bad_locations(self, locations: np.ndarray):
        """
        Set the list of indexes specifying poorly annotated locations during
        the video.

        :param locations: List of integers, the indexes of poorly annotated
                          locations within the video.
        """
        self._bad_locations = locations.astype(np.uint64)
        self._refresh_bad_locations()

    def get_bad_locations(self) -> np.ndarray:
        """
        Get the list of indexes specifying poorly annotated locations during
        the video.

        :returns: A read only view of the numpy array storing indexes of
                  poorly annotated frames.
        """
        view = self._bad_locations.view()
        view.flags.writeable = False
        return view

    def get_prev_bad_location(self, location: int = None, orig_location = None, moves_done = 0) -> int:
        """
        Get the previous bad location based on the current location in the
        probability display.

        :returns: An integer, the index of the nearest previous bad location.
        """
        if(location is None):
            location = self.get_location()
        if(orig_location is None):
            orig_location = location

        if(len(self._bad_locations) == 0):
            return location

        idx = np.searchsorted(self._bad_locations, location, side="left")
        is_bad_spot = self._bad_locations[idx % len(self._bad_locations)] == location
        idx -= 1

        if(is_bad_spot):
            while(location - int(self._bad_locations[idx]) == 1):
                location = int(self._bad_locations[idx])
                idx -= 1

        if(self._is_touched(
            self._bad_locations[idx],
            self._bad_locations,
            self._user_modified_from_last_pass
        )):
            val = self._bad_locations[idx]
            if(val >= location):
                val = -len(self._data) + val
            moves_done += location - val
            if(moves_done >= len(self._data)):
                return orig_location

            return self.get_prev_bad_location(
                self._bad_locations[idx],
                orig_location,
                moves_done
            )

        return int(self._bad_locations[idx])

    def get_next_bad_location(self, location: int = None, orig_location = None, moves_done = 0) -> int:
        """
        Get the next bad location based on the current location in the
        probability display.

        :returns: An integer, the index of the nearest next bad location.
        """
        if(location is None):
            location = self.get_location()
        if(orig_location is None):
            orig_location = location

        if(len(self._bad_locations) == 0):
            return location

        idx = np.searchsorted(self._bad_locations, location, side="right")
        is_bad_spot = self._bad_locations[idx - 1] == location
        idx = idx % len(self._bad_locations)

        if(is_bad_spot):
            while(int(self._bad_locations[idx]) - location == 1):
                location = int(self._bad_locations[idx])
                idx = (idx + 1) % len(self._bad_locations)

        if(self._is_touched(
                self._bad_locations[idx],
                self._bad_locations,
                self._user_modified_from_last_pass
        )):
            val = self._bad_locations[idx]
            if(val <= location):
                val = len(self._data) + val
            moves_done += val - location
            if(moves_done >= len(self._data)):
                return orig_location

            return self.get_prev_bad_location(
                self._bad_locations[idx],
                orig_location,
                moves_done
            )

        return int(self._bad_locations[idx])

    def get_text(self) -> str:
        """
        Get the display text for this probability display.

        :returns: A string, being the display text.
        """
        return self._text

    def set_text(self, value: str):
        """
        Set the display text of this probability display.

        :param value: The string value to set the display text to...
        """
        self._text = value

    def set_segment_starts(self, value: Optional[np.ndarray]):
        self._segment_starts = value

    def set_segment_fix_frames(self, value: Optional[np.ndarray]):
        self._segment_fix_frames = value


def test_demo_displayer():
    app = wx.App()
    print_all_sys_colors()

    frame = wx.Frame(None, wx.ID_ANY, "Test Window")
    layout = wx.BoxSizer(wx.VERTICAL)

    data = np.random.rand(100)
    data[np.random.randint(0, 100, 5)] = np.nan

    prob_display = ProbabilityDisplayer(frame, data, np.flatnonzero(data < 0.1), text="Test1")
    prob_display.set_segment_starts(np.unique(np.random.randint(0, 100, 5)))
    prob_display.set_segment_fix_frames(np.unique(np.random.randint(0, 100, 5)))
    layout.Add(prob_display, 1, wx.EXPAND)

    prob_display2 = ProbabilityDisplayer(frame, data, np.flatnonzero(data < 0.1), text="Test2")
    layout.Add(prob_display2, 1, wx.EXPAND)

    slider = wx.Slider(frame, minValue=0, maxValue=len(data) - 1)
    layout.Add(slider, 0, wx.EXPAND)

    def do(evt):
        prob_display.set_location(slider.GetValue())
        prob_display2.set_location(slider.GetValue())

    slider.Bind(wx.EVT_SLIDER, do)

    frame.SetSizerAndFit(layout)
    frame.SetSize(500, 100)
    frame.Show()

    app.MainLoop()


def print_all_sys_colors():
    for attr in dir(wx):
        if(attr.startswith("SYS_COLOUR")):
            color = wx.SystemSettings.GetColour(getattr(wx, attr))
            red, green, blue, alpha = color
            print(f"{attr}: {color} \033[48;2;{red};{green};{blue}m  \033[0m")


if(__name__ == "__main__"):
    test_demo_displayer()