"""
Module contains a wx video player widget and a wx video controller widget. Uses multi-threading to load frames to a
deque while playing them, allowing for smoother playback...
"""
from typing import Optional, Tuple
import wx
from wx.lib.newevent import NewCommandEvent
import cv2
from collections import deque
import numpy as np
from diplomat.utils.video_io import ContextVideoCapture


def read_frame(video_hdl: cv2.VideoCapture, frame_idx: Optional[int] = None) -> Tuple[bool, np.ndarray]:
    valid_frame = False
    frame = None

    if(frame_idx is not None):
        if(tell_frame(video_hdl) != frame_idx):
            seek_frame(video_hdl, frame_idx)

    if video_hdl.isOpened():
        valid_frame, frame = video_hdl.read()

    if not valid_frame:
        frame = np.zeros((
            int(video_hdl.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(video_hdl.get(cv2.CAP_PROP_FRAME_WIDTH)),
            3
        ), dtype=np.uint8)

    return valid_frame, frame


def seek_frame(video_hdl: cv2.VideoCapture, new_loc: int):
    video_hdl.set(cv2.CAP_PROP_POS_FRAMES, new_loc)


def tell_frame(video_hdl: cv2.VideoCapture) -> int:
    return int(video_hdl.get(cv2.CAP_PROP_POS_FRAMES))


# Represents (x, y, width, height)
Box = Tuple[int, int, int, int]


class VideoPlayer(wx.Control):
    """
    A video player for wx Widgets, Using cv2 for solid cross-platform video support. Can play video, but no audio.
    """

    # The number of frames to store in the backward buffer...
    BACK_LOAD_AMT = 50
    MAX_FAST_FORWARD_MODE = 20

    # Events for the VideoPlayer class, one triggered for every frame change, and one triggered for every change in
    # play state (starting, stopping, pausing, etc....)
    FrameChangeEvent, EVT_FRAME_CHANGE = NewCommandEvent()
    PlayStateChangeEvent, EVT_PLAY_STATE_CHANGE = NewCommandEvent()

    def __init__(self, parent, w_id=wx.ID_ANY, video_hdl: cv2.VideoCapture = None, crop_box: Optional[Box]=None,
                 pos=wx.DefaultPosition, size=wx.DefaultSize, style=wx.BORDER_DEFAULT, validator=wx.DefaultValidator,
                 name="VideoPlayer"):
        """
        Create a new VideoPlayer

        :param parent: The wx Control Parent.
        :param w_id: The wx ID.
        :param video_hdl: The cv2 VideoCapture to play video from. One should avoid never manipulate the video capture
                          once passed to this constructor, as the handle will be passed to another thread for fast
                          video loading.
        :param pos: The position of the widget.
        :param size: The size of the widget.
        :param style: The style of the widget.
        :param validator: The widgets validator.
        :param name: The name of the widget.
        """
        super().__init__(parent, w_id, pos, size, style | wx.FULL_REPAINT_ON_RESIZE, validator, name)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self._width = video_hdl.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._height = video_hdl.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._fps = video_hdl.get(cv2.CAP_PROP_FPS)
        self._crop_box = self._check_crop_box(crop_box, self._width, self._height)

        try:
            self._num_frames = int(video_hdl.get(cv2.CAP_PROP_FRAME_COUNT))
            if(self._num_frames == 0):
                self._num_frames = get_frame_count(video_hdl)
        except:
            self._num_frames = get_frame_count(video_hdl)

        # Useful indicator variables...
        self._playing = False
        self._current_frame = None
        self._frozen = False

        self._prior_frames = deque(maxlen=self.BACK_LOAD_AMT)
        self._current_loc = 0

        size = self._compute_min_size()
        self.SetMinSize(size)
        self.SetInitialSize(size)

        self._core_timer = wx.Timer(self)

        # Create the video loader to start loading frames:
        self._video_hdl = video_hdl
        self._current_frame = read_frame(video_hdl)[1]

        self.Bind(wx.EVT_TIMER, self._on_timer)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda evt: None)

    def _compute_min_size(self) -> wx.Size:
        displays = (wx.Display(i) for i in range(wx.Display.GetCount()))
        sizes = [display.GetGeometry().GetSize() for display in displays]

        w = int(min(self._width / 2, *(s.GetWidth() / 3 for s in sizes)))
        h = int(min(self._height / 2, *(s.GetHeight() / 3 for s in sizes)))

        return wx.Size(w, h)

    @classmethod
    def _get_resize_dims(cls, frame: np.ndarray, width: int, height: int) -> Tuple[int, int]:
        """
        PRIVATE: Get the dimensions to resize the video to in order to fit the widget.
        """
        frame_aspect = frame.shape[0] / frame.shape[1]  # <-- Height / Width
        passed_aspect = height / width

        if(passed_aspect <= frame_aspect):
            # Passed aspect has less height per unit width, so height is the limiting dimension
            return (int(height / frame_aspect), height)
        else:
            # Otherwise the width is the limiting dimension
            return (width, int(width * frame_aspect))

    @classmethod
    def _check_crop_box(cls, box: Optional[Box], vid_width: int, vid_height: int):
        """
        PRIVATE: Validate that the passed cropping box is valid.
        """
        if(box is None):
            return None

        x, y, w, h = box
        if((0 <= x < vid_width) and (0 <= y < vid_height)):
            if((h > 0) and (w > 0)):
                if((x + w <= vid_width) and (y + h <= vid_height)):
                    return box

        raise ValueError("Invalid cropping box!!!!")

    @classmethod
    def _get_video_bbox(cls, frame: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        PRIVATE: Get the video bounding box within the widget...
        """
        v_w, v_h = cls._get_resize_dims(frame, width, height)
        return ((width - v_w) // 2, (height - v_h) // 2, v_w, v_h)

    @classmethod
    def _resize_video(cls, frame: np.ndarray, width: int, height: int, crop_box: Optional[Box]) -> np.ndarray:
        """
        PRIVATE: Resizes the passed frame to optimally fit into the specified width and height, while maintaining
        aspect ratio.

        :param frame: The frame (cv2 image which is really a numpy array) to resize.
        :param width: The desired width of the resized frame.
        :param height: The desired height of the resized frame.
        :return: A new numpy array, being the resized version of the frame.
        """
        # If we have a valid crop box, crop the frame...
        if(crop_box is not None):
            x, y, w, h = crop_box
            frame = frame[y:y+h, x:x+w]

        return cv2.resize(frame, cls._get_resize_dims(frame, width, height), interpolation=cv2.INTER_LINEAR)

    def on_paint(self, event):
        """
        Run on a paint event, redraws the widget.
        """
        painter = wx.PaintDC(self) if(self.IsDoubleBuffered()) else wx.BufferedPaintDC(self)
        self.on_draw(painter)

    def on_draw(self, dc: wx.DC):
        """
        Draws the widget.

        :param dc: The wx DC to use for drawing.
        """
        width, height = self.GetClientSize()

        if((not width) or (not height)):
            return

        dc.SetBackground(wx.Brush(self.GetBackgroundColour(), wx.BRUSHSTYLE_SOLID))
        dc.Clear()

        if(self._current_frame is None):
            return

        resized_frame = self._resize_video(self._current_frame, width, height, self._crop_box)

        # Draw the video background
        b_h, b_w = resized_frame.shape[:2]
        bitmap = wx.Bitmap.FromBuffer(b_w, b_h, resized_frame[:, :, ::-1].astype(dtype=np.uint8))

        loc_x = (width - b_w) // 2
        loc_y = (height - b_h) // 2

        dc.DrawBitmap(bitmap, loc_x, loc_y)

    def _push_time_change_event(self):
        """ PRIVATE: Used to specify how long the event should """
        new_event = self.FrameChangeEvent(id=self.Id, frame=self.get_offset_count())
        wx.PostEvent(self, new_event)

    def _on_timer(self, event, trigger_run = True):
        """
        PRIVATE: Executed whenever a timer event occurs, which triggers a video frame update if the video is playing.
        """
        if(self._playing):
            if(self._frozen):
                self.pause()
                return
            # If we have reached the end of the video, pause the video and don't perform a frame update as
            # we will deadlock the system by waiting for a frame forever...
            if(self._current_loc >= (self._num_frames - 1)):
                self.pause()
                return

            # Get the next frame and set it as the current frame
            self._prior_frames.append(self._current_frame)
            self._current_loc += 1
            self._current_frame = read_frame(self._video_hdl, self._current_loc)[1]
            # Post a frame change event.
            self._push_time_change_event()
            # Trigger a redraw on the next pass through the loop and start the timer to play the next frame...
            if(trigger_run):
                self._core_timer.StartOnce(int(1000 / self._fps))
        self.Refresh()  # Force a redraw....

    def play(self):
        """
        Play the video.
        """
        if(not self.is_playing()):
            self._playing = True
            wx.PostEvent(self, self.PlayStateChangeEvent(id=self.Id, playing=True, stop_triggered = False))
            self._on_timer(None)

    def stop(self):
        """
        Stop the video.
        """
        self._playing = False
        wx.PostEvent(self, self.PlayStateChangeEvent(id=self.Id, playing=False, stop_triggered = True))
        self.set_offset_frames(0)

    def pause(self):
        """
        Pause the video.
        """
        self._playing = False
        wx.PostEvent(self, self.PlayStateChangeEvent(id=self.Id, playing=False, stop_triggered = False))

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

    def _full_jump(self, value: int):
        """
        PRIVATE: Executes a full jump, clearing both internal deques and refilling them with frames. This is done
        whenever we need to do a large jump within a video. Should never be called outside of this class.
        """
        if(self._frozen):
            return

        current_state = self.is_playing()
        self._playing = False
        self._current_loc = value

        # Completely wipe the prior frames...
        self._prior_frames.clear()
        self._current_frame = read_frame(self._video_hdl, self._current_loc)[1]

        self._push_time_change_event()
        # Restore play state prior to frame change...
        self._playing = current_state
        self.Refresh()
        self._core_timer.StartOnce(int(1000 / self._fps))

    def _fast_back(self, amount: int):
        """
        PRIVATE: Used to efficiently go back a given amount of frames, popping frames of the back deque instead of
        clearing both the front and back deques and refilling them. Should never be called outside this class, as it
        performs no checks itself. Used when jump back is small.
        """
        if(self._frozen):
            return

        current_state = self.is_playing()
        self._playing = False

        # Move back the passed amount of frames.
        for i in range(amount):
            self._current_frame = self._prior_frames.pop()
        self._current_loc = self._current_loc - amount

        self._push_time_change_event()

        self._playing = current_state
        self.Refresh()
        self._core_timer.StartOnce(int(1000 / self._fps))

    def _fast_forward(self, amount: int):
        """
        PRIVATE: Used to efficiently move forward a given amount of frames, popping frames of the front deque instead
        of clearing both the front and back deques and refilling them. Should never be called outside this class, as
        it performs no checks itself. Used when jump forward is small.
        """
        if(self._frozen):
            return

        current_state = self.is_playing()
        self._playing = False

        # Move the passed amount of frames forward. Video reader will automatically move forward with us...
        for i in range(amount):
            self._prior_frames.append(self._current_frame)
            self._current_loc += 1
            self._current_frame = read_frame(self._video_hdl, self._current_loc)[1]

        self._push_time_change_event()

        self._playing = current_state
        self.Refresh()
        self._core_timer.StartOnce(int(1000 / self._fps))

    def move_back(self, amount: int = 1):
        """
        Move backward a given amount of frames.

        :param amount: A non-negative integer. Being how many frames to move backward. Defaults to 1.
                       Can be 0, does nothing if so.
        """
        # Check if movement is valid...
        if(amount < 0):
            raise ValueError("Offset must be positive!")
        elif(amount == 0):
            return
        if(self._current_loc - amount < 0):
            raise ValueError(f"Can't go back {amount} frames when at frame {self._current_loc}.")
        # Check if we can perform a 'fast' backtrack, where we have all of the frames in the queue. If not perform
        # a more computationally expensive full jump.
        if(amount > len(self._prior_frames)):
            self._full_jump(self._current_loc - amount)
        else:
            self._fast_back(amount)

    def move_forward(self, amount: int = 1):
        """
        Move forward a given amount of frames.

        :param amount: A non-negative integer. Being how many frames to move forward. Defaults to 1.
                       Can be 0, does nothing if so.
        """
        # Check if movement is valid...
        if(amount < 0):
            raise ValueError("Offset must be positive!")
        elif(amount == 0):
            return
        if(self._current_loc + amount >= self._num_frames):
            raise ValueError(f"Can't go forward {amount} frames when at frame {self._current_loc}.")
        # Check if we can do a fast forward, which is basically the same as moving through frames normally...
        # Otherwise we perform a more expensive full jump.
        if(amount > self.MAX_FAST_FORWARD_MODE):
            self._full_jump(self._current_loc + amount)
        else:
            self._fast_forward(amount)

    def set_offset_frames(self, value: int):
        """
        Set the current frame offset location into the video.

        :param value: An integer index, being the frame to move to in the video.
        """
        # Is this a valid frame value?
        if(not (0 <= value < self._num_frames)):
            raise ValueError(f"Can't set frame index to {value}, there is only {self._num_frames} frames.")
        # Determine which way the value is moving the current video location, and move backward/forward based on that.
        if(value < self._current_loc):
            self.move_back(self._current_loc - value)
        elif(value > self._current_loc):
            self.move_forward(value - self._current_loc)

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

    PLAY_SYMBOL = "\u25B6"
    PAUSE_SYMBOL = "\u23F8"
    STOP_SYMBOL = "\u23F9"
    FRAME_BACK_SYMBOL = "\u21b6"
    FRAME_FORWARD_SYMBOL = "\u21b7"

    def __init__(self, parent, video_player: VideoPlayer, w_id = wx.ID_ANY, pos = wx.DefaultPosition,
                 size = wx.DefaultSize, style = wx.TAB_TRAVERSAL, name = "VideoController"):
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

        if(video_player is None):
            raise ValueError("Have to pass a VideoPlayer!!!")

        self._video_player = video_player

        self._sizer = wx.BoxSizer(wx.HORIZONTAL)

        self._back_btn = wx.Button(self, label=self.FRAME_BACK_SYMBOL)
        self._play_pause_btn = wx.Button(self, label=self.PLAY_SYMBOL)
        self._stop_btn = wx.Button(self, label=self.STOP_SYMBOL)
        self._forward_btn = wx.Button(self, label=self.FRAME_FORWARD_SYMBOL)

        self._slider_control = wx.Slider(self, value=0, minValue=0, maxValue=video_player.get_total_frames() - 1,
                                         style=wx.SL_HORIZONTAL | wx.SL_LABELS)

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
        if(not self.IsEnabled() and not self._video_player.is_frozen()):
            return

        # Is the control were working with some type of text input?
        # If so don't process this event...
        window: wx.Window = wx.GetTopLevelParent(self)
        foc_widget = window.FindFocus()
        from wx.lib.agw.floatspin import FloatSpin
        if(isinstance(foc_widget, (wx.SpinCtrl, wx.TextEntry, FloatSpin))):
            evt.Skip()
            return

        if(evt.GetModifiers() != 0):
            evt.Skip()
            return

        elif(evt.GetKeyCode() == wx.WXK_SPACE):
            self.on_play_pause_press(None)
            # If it was the space key we eat the event, to stop buttons from triggering. User can still use enter key
            # to activate buttons in the UI....
            return
        elif(evt.GetKeyCode() == wx.WXK_LEFT):
            self.on_back_press(None)
        elif(evt.GetKeyCode() == wx.WXK_RIGHT):
            self.on_forward_press(None)
        elif(evt.GetKeyCode() == wx.WXK_BACK):
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
        self._play_pause_btn.SetLabel(self.PAUSE_SYMBOL if(event.playing) else self.PLAY_SYMBOL)

    def on_slide(self, event):
        """
        PRIVATE: Triggered when slider is moved.
        """
        self._video_player.set_offset_frames(self._slider_control.GetValue())

    def on_play_pause_press(self, event):
        """
        PRIVATE: Triggered when the play/pause button is pressed.
        """
        if(self._video_player.is_playing()):
            self._video_player.pause()
        else:
            if(self._video_player.get_offset_count() + 1 == self._video_player.get_total_frames()):
                self._video_player.set_offset_frames(0)
            self._video_player.play()

    def on_back_press(self, event):
        """
        PRIVATE: Triggered when go back 1 frame button is pressed.
        """
        if(self._video_player.get_offset_count() > 0):
            self._video_player.move_back()

    def on_forward_press(self, event):
        """
        PRIVATE: Triggered when go forward 1 frame button has been pressed.
        """
        if(self._video_player.get_offset_count() < (self._video_player.get_total_frames() - 1)):
            self._video_player.move_forward()


def get_frame_count(video_hdl):
    i = 0
    while(video_hdl.isOpened() and video_hdl.grab()):
        i += 1

    video_hdl.set(cv2.CAP_PROP_POS_MSEC, 0)
    return i


def _main_test():
    from diplomat.wx_gui.probability_displayer import ProbabilityDisplayer
    # We test the video player by playing a video with it.
    vid_path = input("Enter a video path: ")

    print(get_frame_count(ContextVideoCapture(vid_path)))

    app = wx.App()
    wid_frame = wx.Frame(None, title="Test...")
    panel = wx.Panel(parent=wid_frame)

    sizer = wx.BoxSizer(wx.VERTICAL)

    wid = VideoPlayer(panel, video_hdl=ContextVideoCapture(vid_path))
    obj3 = ProbabilityDisplayer(panel, data=np.random.randint(0, 10, (wid.get_total_frames())), bad_locations=np.array([], np.uint64))
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


if(__name__ == "__main__"):
    _main_test()
