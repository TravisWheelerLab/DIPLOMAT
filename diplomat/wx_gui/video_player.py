"""
Module contains a wx video player widget and a wx video controller widget. Uses multi-threading to load frames to a
deque while playing them, allowing for smoother playback...
"""

import dataclasses
import time
from typing import Optional, Tuple
import wx
from wx.lib.newevent import NewCommandEvent
import cv2
from collections import deque
import numpy as np

from diplomat.utils.video_info import get_frame_count_robust_fast
from diplomat.utils.video_io import ContextVideoCapture


def read_frame(
    video_hdl: cv2.VideoCapture, frame_idx: Optional[int] = None
) -> Tuple[bool, int, np.ndarray]:
    valid_frame = False
    frame = None

    if frame_idx is not None:
        if tell_frame(video_hdl) != frame_idx:
            seek_frame(video_hdl, frame_idx)

    if video_hdl.isOpened():
        valid_frame, frame = video_hdl.read()

    if not valid_frame:
        frame = np.zeros(
            (
                int(video_hdl.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(video_hdl.get(cv2.CAP_PROP_FRAME_WIDTH)),
                3,
            ),
            dtype=np.uint8,
        )

    return valid_frame, int(video_hdl.get(cv2.CAP_PROP_POS_FRAMES)), frame


def seek_frame(video_hdl: cv2.VideoCapture, new_loc: int):
    video_hdl.set(cv2.CAP_PROP_POS_FRAMES, new_loc)


def tell_frame(video_hdl: cv2.VideoCapture) -> int:
    return int(video_hdl.get(cv2.CAP_PROP_POS_FRAMES))


# Represents (x, y, width, height)
Box = Tuple[int, int, int, int]
Coord = Tuple[float, float]
IntCoord = Tuple[int, int]


class VideoTransform:
    def __init__(
        self,
        video_dims: IntCoord,
        widget_dims: IntCoord,
        crop_box: Optional[Box] = None,
        offset: Coord = (0, 0),
        scale: float = 1,
    ):
        self._params = dict(
            video_dims=video_dims,
            widget_dims=widget_dims,
            crop_box=crop_box,
            offset=offset,
            scale=scale,
        )
        crop_box = self.check_crop_box(crop_box, *video_dims)
        self._cropped_video_size = tuple(
            crop_box[2:] if crop_box is not None else video_dims
        )
        self._widget_size = tuple(widget_dims)
        self._video_offset, self._video_scale = self._get_video_pos_and_scale(
            *self._cropped_video_size, *self._widget_size
        )
        self._post_offset = tuple(offset)
        self._post_scale = float(scale)

    @classmethod
    def check_crop_box(cls, box: Optional[Box], vid_width: int, vid_height: int):
        """
        PRIVATE: Validate that the passed cropping box is valid.
        """
        if box is None:
            return None

        x, y, w, h = box
        if (0 <= x < vid_width) and (0 <= y < vid_height):
            if (h > 0) and (w > 0):
                if (x + w <= vid_width) and (y + h <= vid_height):
                    return box

        raise ValueError("Invalid cropping box!!!!")

    def update(
        self,
        video_dims: Optional[IntCoord] = None,
        widget_dims: Optional[IntCoord] = None,
        crop_box: Optional[Box] = None,
        offset: Optional[Coord] = None,
        scale: Optional[float] = None,
    ):
        obj = dict(
            video_dims=video_dims,
            widget_dims=widget_dims,
            crop_box=crop_box,
            offset=offset,
            scale=scale,
        )
        obj = {k: self._params[k] if v is None else v for k, v in obj.items()}
        # If object is already up to date, do nothing...
        if all(obj[k] == self._params[k] for k in obj):
            return
        self.__init__(**obj)

    @property
    def offset(self) -> Coord:
        return self._post_offset

    @property
    def scale(self) -> float:
        return self._post_scale

    def adjust(self, offset: Coord, scale: float):
        self.update(offset=offset, scale=scale)

    @classmethod
    def _get_resize_dims(
        cls, frame_w: int, frame_h: int, width: int, height: int
    ) -> Tuple[int, int]:
        """
        PRIVATE: Get the dimensions to resize the video to in order to fit the widget.
        """
        frame_aspect = frame_h / frame_w  # <-- Height / Width
        passed_aspect = height / width

        if passed_aspect <= frame_aspect:
            # Passed aspect has less height per unit width, so height is the limiting dimension
            return int(height / frame_aspect), height
        else:
            # Otherwise the width is the limiting dimension
            return width, int(width * frame_aspect)

    @classmethod
    def _get_video_pos_and_scale(
        cls, frame_w: int, frame_h: int, width: int, height: int
    ) -> Tuple[IntCoord, Coord]:
        """
        PRIVATE: Get the video bounding box within the widget...
        """
        v_w, v_h = cls._get_resize_dims(frame_w, frame_h, width, height)
        return (
            ((width - v_w) // 2, (height - v_h) // 2),
            (v_w / frame_w, v_h / frame_h),
        )

    def video_crop_to_widget(self, xy: Coord) -> Coord:
        ps = self._post_scale
        x, y = tuple(
            ((v * vs + vo) - (po * ws)) * ps
            for v, vs, vo, po, ws in zip(
                xy,
                self._video_scale,
                self._video_offset,
                self._post_offset,
                self._widget_size,
            )
        )
        return x, y

    def widget_to_video_crop(self, xy: Coord) -> Coord:
        ps = self._post_scale
        x, y = tuple(
            (((w / ps) + (po * ws)) - vo) / vs
            for w, vs, vo, po, ws in zip(
                xy,
                self._video_scale,
                self._video_offset,
                self._post_offset,
                self._widget_size,
            )
        )
        return x, y

    def get_cropped_image(self, img: np.ndarray) -> np.ndarray:
        c_box = self._params["crop_box"]
        if c_box is None:
            return img
        x, y, w, h = c_box
        return img[y : y + h, x : x + w]

    def transform_image(
        self,
        img: np.ndarray,
        img_scale: float = 1.0,
        interpolation: int = cv2.INTER_AREA,
    ) -> Tuple[np.ndarray, Coord]:
        # Get visible bounds of the video...
        video_top_left = [int(v / img_scale) for v in self.widget_to_video_crop((0, 0))]
        video_bottom_right = [
            int(np.ceil(v / img_scale))
            for v in self.widget_to_video_crop(self._widget_size)
        ]

        video_top_left = [
            np.clip(v, 0, mx - 1)
            for v, mx in zip(video_top_left, self._cropped_video_size)
        ]
        video_bottom_right = [
            np.clip(v, 1, mx)
            for v, mx in zip(video_bottom_right, self._cropped_video_size)
        ]

        # Grab the subsection of the image...
        img = img[
            video_top_left[1] : video_bottom_right[1],
            video_top_left[0] : video_bottom_right[0],
        ]
        img_scaled = cv2.resize(
            img,
            (
                int(img.shape[1] * img_scale * self._video_scale[0] * self._post_scale),
                int(img.shape[0] * img_scale * self._video_scale[1] * self._post_scale),
            ),
            interpolation=interpolation,
        )

        return (
            img_scaled,
            self.video_crop_to_widget(
                (video_top_left[0] * img_scale, video_top_left[1] * img_scale)
            ),
        )


@dataclasses.dataclass
class ZoomConfig:
    key: Optional[wx.KeyCode] = None
    min_zoom: float = 1
    max_zoom: float = 20
    zoom_slow_down: float = 1000
    min_move_refresh: float = 10


class VideoPlayer(wx.Control):
    """
    A video player for wx Widgets, Using cv2 for solid cross-platform video support. Can play video, but no audio.
    """

    # The number of frames to store in the backward buffer...
    BACK_LOAD_AMT = 50
    MAX_FAST_FORWARD_MODE = 10

    # Events for the VideoPlayer class, one triggered for every frame change, and one triggered for every change in
    # play state (starting, stopping, pausing, etc....)
    FrameChangeEvent, EVT_FRAME_CHANGE = NewCommandEvent()
    PlayStateChangeEvent, EVT_PLAY_STATE_CHANGE = NewCommandEvent()

    def __init__(
        self,
        parent,
        w_id=wx.ID_ANY,
        video_hdl: cv2.VideoCapture = None,
        crop_box: Optional[Box] = None,
        zoom_config: Optional[ZoomConfig] = None,
        pos=wx.DefaultPosition,
        size=wx.DefaultSize,
        style=wx.BORDER_DEFAULT,
        validator=wx.DefaultValidator,
        name="VideoPlayer",
    ):
        """
        Create a new VideoPlayer

        :param parent: The wx Control Parent.
        :param w_id: The wx ID.
        :param video_hdl: The cv2 VideoCapture to play video from. One should avoid never manipulate the video capture
                          once passed to this constructor, as the handle will be passed to another thread for fast
                          video loading.
        :param crop_box: Tuple of ints, x, y, width, height, being the area of the video to show instead of the entire video.
                         if set to None, just shows the entire video...
        :param pos: The position of the widget.
        :param size: The size of the widget.
        :param style: The style of the widget.
        :param validator: The widgets validator.
        :param name: The name of the widget.
        """
        super().__init__(
            parent, w_id, pos, size, style | wx.FULL_REPAINT_ON_RESIZE, validator, name
        )
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self._width = int(video_hdl.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(video_hdl.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = video_hdl.get(cv2.CAP_PROP_FPS)
        self._crop_box = VideoTransform.check_crop_box(
            crop_box, self._width, self._height
        )

        self._num_frames = get_frame_count(video_hdl)

        # Useful indicator variables...
        self._playing = False
        self._frozen = False

        self._prior_frames = deque(maxlen=self.BACK_LOAD_AMT)
        self._current_loc = 0

        size = self._compute_min_size()
        self.SetMinSize(size)
        self.SetInitialSize(size)

        self._core_timer = wx.Timer(self)

        # Create the video loader to start loading frames:
        self._video_hdl = video_hdl
        self._loaded_frame = None
        self._max_video_load_rate = (1 / self._fps) / 2
        self._last_frame_read = time.monotonic() - self._max_video_load_rate * 2
        self._video_transform = None
        self._zoom_config = zoom_config
        self._is_pressed = False
        self._prior_mouse_location = (0, 0)

        self.Bind(wx.EVT_TIMER, self._on_timer)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_MOUSEWHEEL, self._on_wheel)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_mouse_down)
        self.Bind(wx.EVT_LEFT_UP, self._on_mouse_up)
        self.Bind(wx.EVT_MOTION, self._on_mouse_move)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda evt: None)

    @property
    def video_transform(self) -> VideoTransform:
        if self._loaded_frame is None:
            raise ValueError("No frame is loaded!")
        if self._video_transform is None:
            fh, fw = self._loaded_frame[1].shape[:2]
            self._video_transform = VideoTransform(
                (fw, fh), self.GetSize().Get(), self._crop_box
            )
        else:
            fh, fw = self._loaded_frame[1].shape[:2]
            self._video_transform.update((fw, fh), self.GetSize().Get(), self._crop_box)
        return self._video_transform

    def _on_wheel(self, evt):
        if self._zoom_config is None:
            evt.Skip()
            return
        if self._zoom_config.key is not None and not wx.GetKeyState(
            self._zoom_config.key
        ):
            evt.Skip()
            return
        if self._loaded_frame is None:
            return

        vt = self.video_transform

        w, h = self.GetClientSize()
        x, y = self.ScreenToClient(wx.GetMousePosition())
        scale = vt.scale
        offset = vt.offset

        ix = (x + (scale * w * offset[0])) / (scale * w)
        iy = (y + (scale * h * offset[1])) / (scale * h)

        scale = min(
            self._zoom_config.max_zoom,
            max(
                self._zoom_config.min_zoom,
                scale + evt.GetWheelRotation() / self._zoom_config.zoom_slow_down,
            ),
        )
        offset = (ix - (x / (scale * w)), iy - (y / (scale * h)))

        if scale <= 1:
            offset = (0, 0)

        vt.update(offset=offset, scale=scale)
        self.Refresh()
        evt.Skip()

    def _on_mouse_down(self, evt):
        if self._zoom_config is None or self._is_pressed:
            evt.Skip()
            return
        self._is_pressed = True
        self._prior_mouse_location = tuple(evt.GetPosition())
        self._net_dist = 0
        evt.Skip()

    def _on_mouse_up(self, evt):
        if self._zoom_config is None or not self._is_pressed:
            evt.Skip()
            return
        self._on_mouse_move(evt, True)
        self._is_pressed = False
        evt.Skip()

    def _on_mouse_move(self, evt, force_move=False):
        if self._zoom_config is None or not self._is_pressed:
            evt.Skip()
            return
        if self._loaded_frame is None:
            return

        vt = self.video_transform

        nx, ny = evt.GetPosition()
        px, py = self._prior_mouse_location
        w, h = evt.GetEventObject().GetClientSize()
        offset = vt.offset
        scale = vt.scale

        offset = (
            offset[0] + ((px - nx) / (scale * w)),
            offset[1] + ((py - ny) / (scale * h)),
        )
        self._net_dist += np.sqrt((px - nx) ** 2 + (py - ny) ** 2)

        if force_move or self._net_dist > self._zoom_config.min_move_refresh:
            self._prior_mouse_location = (nx, ny)
            self._net_dist = 0

            if scale <= 1:
                offset = (0, 0)

            vt.update(offset=offset)
            self.Refresh()

        evt.Skip()

    def _compute_min_size(self) -> wx.Size:
        displays = (wx.Display(i) for i in range(wx.Display.GetCount()))
        sizes = [display.GetGeometry().GetSize() for display in displays]

        w = int(min(self._width / 2, *(s.GetWidth() / 3 for s in sizes)))
        h = int(min(self._height / 2, *(s.GetHeight() / 3 for s in sizes)))

        return wx.Size(w, h)

    def on_paint(self, event):
        """
        Run on a paint event, redraws the widget.
        """
        painter = (
            wx.PaintDC(self) if (self.IsDoubleBuffered()) else wx.BufferedPaintDC(self)
        )
        self.on_draw(painter)

    def _attempt_frame_load(self):
        now = time.monotonic()
        if now - self._last_frame_read < self._max_video_load_rate:
            # self.Refresh()  # is this needed?
            return

        if self._loaded_frame is None:
            self._prior_frames.clear()
            self._loaded_frame = read_frame(self._video_hdl, self._current_loc)[1:]
            return

        offset = self._current_loc - self._loaded_frame[0]
        if offset == 0:
            return
        elif offset > 0 and offset <= self.MAX_FAST_FORWARD_MODE:
            while self._loaded_frame[0] < self._current_loc:
                self._prior_frames.append(self._loaded_frame)
                self._loaded_frame = read_frame(self._video_hdl)[1:]
        elif offset < 0 and offset >= len(self._prior_frames):
            while (
                self._loaded_frame[0] > self._current_loc
                and len(self._prior_frames) > 0
            ):
                self._loaded_frame = self._prior_frames.pop()
        else:
            self._prior_frames.clear()
            self._loaded_frame = read_frame(self._video_hdl, self._current_loc)[1:]

        self._last_frame_read = now

    def on_draw(self, dc: wx.DC):
        """
        Draws the widget.

        :param dc: The wx DC to use for drawing.
        """
        width, height = self.GetClientSize()

        if (not width) or (not height):
            return

        dc.SetBackground(wx.Brush(self.GetBackgroundColour(), wx.BRUSHSTYLE_SOLID))
        dc.Clear()

        self._attempt_frame_load()

        if self._loaded_frame is None:
            return

        vt = self.video_transform
        resized_frame, (loc_x, loc_y) = vt.transform_image(
            vt.get_cropped_image(self._loaded_frame[1]),
            img_scale=1,
            interpolation=cv2.INTER_LINEAR,
        )

        # Draw the video background
        b_h, b_w = resized_frame.shape[:2]
        bitmap = wx.Bitmap.FromBuffer(
            b_w, b_h, resized_frame[:, :, ::-1].astype(dtype=np.uint8)
        )

        dc.DrawBitmap(bitmap, int(loc_x), int(loc_y))

    def _push_time_change_event(self):
        """PRIVATE: Used to specify how long the event should"""
        new_event = self.FrameChangeEvent(id=self.Id, frame=self.get_offset_count())
        wx.PostEvent(self, new_event)

    def _on_timer(self, event, trigger_run=True):
        """
        PRIVATE: Executed whenever a timer event occurs, which triggers a video frame update if the video is playing.
        """
        if self._playing:
            if self._frozen:
                self.pause()
                return
            # If we have reached the end of the video, pause the video and don't perform a frame update as
            # we will deadlock the system by waiting for a frame forever...
            if self._current_loc >= (self._num_frames - 1):
                self.pause()
                return

            # Get the next frame and set it as the current frame
            self._current_loc += 1
            # Post a frame change event.
            self._push_time_change_event()
            # Trigger a redraw on the next pass through the loop and start the timer to play the next frame...
            if trigger_run:
                self._core_timer.StartOnce(int(1000 / self._fps))
        self.Refresh()  # Force a redraw....

    def play(self):
        """
        Play the video.
        """
        if not self.is_playing():
            self._playing = True
            wx.PostEvent(
                self,
                self.PlayStateChangeEvent(
                    id=self.Id, playing=True, stop_triggered=False
                ),
            )
            self._on_timer(None)

    def stop(self):
        """
        Stop the video.
        """
        self._playing = False
        wx.PostEvent(
            self,
            self.PlayStateChangeEvent(id=self.Id, playing=False, stop_triggered=True),
        )
        self.set_offset_frames(0)

    def pause(self):
        """
        Pause the video.
        """
        self._playing = False
        wx.PostEvent(
            self,
            self.PlayStateChangeEvent(id=self.Id, playing=False, stop_triggered=False),
        )

    def is_playing(self) -> bool:
        """
        Returns whether or not the video is currently playing.
        """
        return self._playing

    def freeze(self):
        """
        Freeze the video player, immediately pausing the video and making it unresponsive to play/pause/stop commands,
        and also frame changing methods.
        """
        self.pause()
        self._frozen = True

    def unfreeze(self):
        """
        Unfreeze the video, allowing controls to work again.
        """
        self.pause()
        self._frozen = False

    def is_frozen(self) -> bool:
        """
        Check whether this video is frozen.

        :returns: True is this video is frozen and therefore will not respond to any play/pause/stop commands, or
                  False otherwise.
        """
        return self._frozen

    def get_offset_count(self):
        """
        Get the current frame index we are at in the video.

        :returns: An integer, the frame offset.
        """
        return self._current_loc

    def get_total_frames(self):
        """
        Get the total number of frames in this video.

        :returns: An integer being the total frame count of the video.
        """
        return int(self._num_frames)

    def move_back(self, amount: int = 1):
        """
        Move backward a given amount of frames.

        :param amount: A non-negative integer. Being how many frames to move backward. Defaults to 1.
                       Can be 0, does nothing if so.
        """
        # Check if movement is valid...
        if amount < 0:
            raise ValueError("Offset must be positive!")
        elif amount == 0:
            return
        self.set_offset_frames(self._current_loc - amount)

    def move_forward(self, amount: int = 1):
        """
        Move forward a given amount of frames.

        :param amount: A non-negative integer. Being how many frames to move forward. Defaults to 1.
                       Can be 0, does nothing if so.
        """
        # Check if movement is valid...
        if amount < 0:
            raise ValueError("Offset must be positive!")
        elif amount == 0:
            return
        self.set_offset_frames(self._current_loc + amount)

    def set_offset_frames(self, value: int):
        """
        Set the current frame offset location into the video.

        :param value: An integer index, being the frame to move to in the video.
        """
        # Is this a valid frame value?
        if not (0 <= value < self._num_frames):
            raise ValueError(
                f"Can't set frame index to {value}, there is only {self._num_frames} frames."
            )
        if self._frozen:
            return

        # current_state = self._playing
        # self._playing = False
        self._current_loc = value
        # Restore play state prior to frame change...
        # self._playing = current_state
        self._push_time_change_event()
        self.Refresh()
        # self._core_timer.StartOnce(int(1000 / self._fps))

    def __del__(self):
        """
        Delete this video player, deleting its video reading thread.
        """
        self._prior_frames.clear()
        self._video_hdl.release()


class VideoController(wx.Panel):
    """
    Provides a set of video controls for controlling a VideoPlayer. Provides some play back controls.
    """

    PLAY_SYMBOL = "\u25b6"
    PAUSE_SYMBOL = "\u23f8"
    STOP_SYMBOL = "\u23f9"
    FRAME_BACK_SYMBOL = "\u21b6"
    FRAME_FORWARD_SYMBOL = "\u21b7"

    def __init__(
        self,
        parent,
        video_player: VideoPlayer,
        w_id=wx.ID_ANY,
        pos=wx.DefaultPosition,
        size=wx.DefaultSize,
        style=wx.TAB_TRAVERSAL,
        name="VideoController",
    ):
        """
        Construct a new VideoController.

        :param parent: The parent WX widget.
        :param video_player: The VideoPlayer to control. Will automatically hook into the video player's events.
        :param w_id: The WX ID. Defaults to wx.ID_ANY
        :param pos: The WX Position. Defaults to wx.DefaultPosition.
        :param size: The WX Size. Defaults to wx.DefaultSize.
        :param style: A wx.Panel style. Look at wx.Panel docs to see supported styles. Defaults to wx.TAB_TRAVERSAL.
        :param name: The WX internal name.
        """
        super().__init__(parent, w_id, pos, size, style, name)

        if video_player is None:
            raise ValueError("Have to pass a VideoPlayer!!!")

        self._video_player = video_player

        self._sizer = wx.BoxSizer(wx.HORIZONTAL)

        self._back_btn = wx.Button(self, label=self.FRAME_BACK_SYMBOL)
        self._play_pause_btn = wx.Button(self, label=self.PLAY_SYMBOL)
        self._stop_btn = wx.Button(self, label=self.STOP_SYMBOL)
        self._forward_btn = wx.Button(self, label=self.FRAME_FORWARD_SYMBOL)

        self._slider_control = wx.Slider(
            self,
            value=0,
            minValue=0,
            maxValue=video_player.get_total_frames() - 1,
            style=wx.SL_HORIZONTAL | wx.SL_LABELS,
        )

        self._sizer.Add(self._back_btn, 0, wx.EXPAND | wx.ALL)
        self._sizer.Add(self._play_pause_btn, 0, wx.EXPAND | wx.ALL)
        self._sizer.Add(self._stop_btn, 0, wx.EXPAND | wx.ALL)
        self._sizer.Add(self._forward_btn, 0, wx.EXPAND | wx.ALL)
        self._sizer.Add(self._slider_control, 1, wx.EXPAND)

        self._sizer.SetSizeHints(self)
        self.SetSizer(self._sizer)

        self._video_player.Bind(VideoPlayer.EVT_FRAME_CHANGE, self.frame_change)
        self._video_player.Bind(VideoPlayer.EVT_PLAY_STATE_CHANGE, self.on_play_switch)
        self._slider_control.Bind(wx.EVT_SLIDER, self.on_slide)
        self._play_pause_btn.Bind(wx.EVT_BUTTON, self.on_play_pause_press)
        self._back_btn.Bind(wx.EVT_BUTTON, self.on_back_press)
        self._forward_btn.Bind(wx.EVT_BUTTON, self.on_forward_press)
        self._stop_btn.Bind(wx.EVT_BUTTON, lambda evt: self._video_player.stop())

    def on_char(self, evt: wx.KeyEvent):
        """
        PRIVATE: Handles optional keyboard events....
        """
        if not self.IsEnabled() and not self._video_player.is_frozen():
            return

        # Is the control were working with some type of text input?
        # If so don't process this event...
        window: wx.Window = wx.GetTopLevelParent(self)
        foc_widget = window.FindFocus()
        from wx.lib.agw.floatspin import FloatSpin

        if isinstance(foc_widget, (wx.SpinCtrl, wx.TextEntry, FloatSpin)):
            evt.Skip()
            return

        if evt.GetModifiers() != 0:
            evt.Skip()
            return

        elif evt.GetKeyCode() == wx.WXK_SPACE:
            self.on_play_pause_press(None)
            # If it was the space key we eat the event, to stop buttons from triggering. User can still use enter key
            # to activate buttons in the UI....
            return
        elif evt.GetKeyCode() == wx.WXK_LEFT:
            self.on_back_press(None)
        elif evt.GetKeyCode() == wx.WXK_RIGHT:
            self.on_forward_press(None)
        elif evt.GetKeyCode() == wx.WXK_BACK:
            self._video_player.stop()

        evt.Skip()

    def set_keyboard_listener(self, control: wx.Window):
        """
        Set the keyboard listener, which enables keyboard shortcuts for this video controller.

        :param control: The wx.Window to bind listen for keyboard events from.
        """
        control.Bind(wx.EVT_CHAR_HOOK, self.on_char)

    def frame_change(self, event):
        """
        PRIVATE: Triggered when video player frame changes.
        """
        frame = event.frame
        self._slider_control.SetValue(frame)
        self._back_btn.Enable(frame > 0)
        self._forward_btn.Enable(frame < (self._video_player.get_total_frames() - 1))
        wx.PostEvent(self, event)

    def on_play_switch(self, event):
        """
        PRIVATE: Triggered when video player is paused/played.
        """
        self._play_pause_btn.SetLabel(
            self.PAUSE_SYMBOL if (event.playing) else self.PLAY_SYMBOL
        )

    def on_slide(self, event):
        """
        PRIVATE: Triggered when slider is moved.
        """
        self._video_player.set_offset_frames(self._slider_control.GetValue())

    def on_play_pause_press(self, event):
        """
        PRIVATE: Triggered when the play/pause button is pressed.
        """
        if self._video_player.is_playing():
            self._video_player.pause()
        else:
            if (
                self._video_player.get_offset_count() + 1
                == self._video_player.get_total_frames()
            ):
                self._video_player.set_offset_frames(0)
            self._video_player.play()

    def on_back_press(self, event):
        """
        PRIVATE: Triggered when go back 1 frame button is pressed.
        """
        if self._video_player.get_offset_count() > 0:
            self._video_player.move_back()

    def on_forward_press(self, event):
        """
        PRIVATE: Triggered when go forward 1 frame button has been pressed.
        """
        if self._video_player.get_offset_count() < (
            self._video_player.get_total_frames() - 1
        ):
            self._video_player.move_forward()


get_frame_count = get_frame_count_robust_fast


def _main_test():
    from diplomat.wx_gui.probability_displayer import ProbabilityDisplayer

    # We test the video player by playing a video with it.
    vid_path = input("Enter a video path: ")

    print(get_frame_count(ContextVideoCapture(vid_path)))

    app = wx.App()
    wid_frame = wx.Frame(None, title="Test...")
    panel = wx.Panel(parent=wid_frame)

    sizer = wx.BoxSizer(wx.VERTICAL)

    wid = VideoPlayer(
        panel, video_hdl=ContextVideoCapture(vid_path), zoom_config=ZoomConfig()
    )
    obj3 = ProbabilityDisplayer(
        panel,
        data=np.random.randint(0, 10, (wid.get_total_frames())),
        bad_locations=np.array([], np.uint64),
    )
    obj2 = VideoController(panel, video_player=wid)

    obj2.set_keyboard_listener(wid_frame)

    obj2.Bind(wid.EVT_FRAME_CHANGE, lambda evt: obj3.set_location(evt.frame))

    sizer.Add(wid, 1, wx.EXPAND)
    sizer.Add(obj3, 0, wx.EXPAND)
    sizer.Add(obj2, 0, wx.EXPAND)

    panel.SetSizerAndFit(sizer)

    wid_frame.Fit()
    wid_frame.Show(True)

    def destroy(evt):
        wid_frame.Destroy()

    wid_frame.Bind(wx.EVT_CLOSE, destroy)
    app.MainLoop()


if __name__ == "__main__":
    _main_test()
