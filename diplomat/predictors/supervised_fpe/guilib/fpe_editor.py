from pathlib import Path
import wx
import cv2
import numpy as np
from typing import List, Any, Tuple, Optional, Callable, Mapping
from .point_edit import PointEditor, PointViewNEdit, PoseLabeler
from .progress_dialog import FBProgressDialog
from .score_lib import ScoreEngine, ScoreEngineDisplayer
from .scroll_image_list import ScrollImageList
from diplomat.processing import Pose, ProgressBar
from .helpdialog import HelpDialog
from .video_player import VideoController
from wx.lib.scrolledpanel import ScrolledPanel
from collections import deque
from . import icons


class History:
    """
    History Object, represents a navigable history, allowing for doing, undoing, and redoing actions using little doers.
    """
    class Element:
        """
        A history 'element'. Used internally.
        """
        def __init__(self, name: str, value: Any):
            """
            Create a new element for placement in the history queue.

            :param name: A string which identifies type of this action.
            :param value: The data required to redo or undo this action.
            """
            self.name = name
            self.value = value

    def __init__(self, max_size: int = 100):
        """
        Construct a new history object.

        :param max_size: The size of the history, or the size until items begin being deleted from the history.
        """
        self.history = deque(maxlen=max_size)
        self.future = deque(maxlen=max_size)
        # :)
        self._little_redoers = {}
        self._little_undoers = {}
        self._little_confirmers = {}

        self._on_chg = None

    def _change(self):
        if(self._on_chg is not None):
            self._on_chg(self)

    def set_change_handler(self, func: Optional[Callable[["History"], None]]):
        """
        Set the function to be called whenever this history object is updated. Can be used to update ui widgets and etc.
        """
        self._on_chg = func

    def do(self, name: str, value: Any):
        """
        Do an action, adding it to the history, clearing the future.

        :param name: The name of this action, or classifier of it.
        :param value: The state before this action was done, can be anything....
        """
        self.future.clear()
        self.history.append(self.Element(name, value))
        self._change()

    def register_undoer(self, name: str, func: Callable[[Any], Optional[Any]]):
        """
        Register a little doer, which undoes an action of a specific type.

        :param name: The type of action this little doer undoes.
        :param func: A callable which accepts the stored state to change to and also returns the state prior to
                    applying the change. Data type will depend on what was passed to the do method of the history
                    object. Returning None adds no entry to the future.
        """
        self._little_undoers[name] = func

    def register_redoer(self, name: str, func: Callable[[Any], Optional[Any]]):
        """
        Register a little doer, which redoes an action of a specific type.

        :param name: The type of action this little doer redoes.
        :param func: A callable which accepts the stored state to change to and also returns the state prior to
                    applying the change. Data type will depend on what was passed to the do method of the history
                    object. Returning None adds no entry to the history.
        """
        self._little_redoers[name] = func

    def register_confirmer(self, name: str, func: Callable[[bool], bool]):
        """
        Register a confirmer, which confirms if an action should actually be done before doing it.

        :param name: The type of action this confirmer confirms.
        :param func: A callable which accepts a boolean which is true if performing an undo and false if performing a
                     redo. The callable should return a boolean, True if user confirmed the action or false if the user
                     canceled the action.
        """
        self._little_confirmers[name] = func

    def undo(self):
        """
        Undo the most recent change in history, using one of the registered little undoers.
        """
        if(self.can_undo()):
            result = self.history.pop()
            if(result.name in self._little_confirmers):
                if(not self._little_confirmers[result.name](True)):
                    self.history.append(result)
                    return

            if(result.name in self._little_undoers):
                new_result = self.Element(result.name, self._little_undoers[result.name](result.value))
                if(new_result.value is not None):
                    self.future.appendleft(new_result)
        self._change()

    def redo(self):
        """
        Redo the most recent change in history, using one of the registered little redoers.
        """
        if(self.can_redo()):
            result = self.future.popleft()
            if(result.name in self._little_confirmers):
                if(not self._little_confirmers[result.name](False)):
                    self.future.appendleft(result)
                    return

            if(result.name in self._little_redoers):
                new_result = self.Element(result.name, self._little_undoers[result.name](result.value))
                if(new_result.value is not None):
                    self.history.append(new_result)
        self._change()

    def clear(self):
        """
        Clear the history, wiping out all entries...
        """
        self.future.clear()
        self.history.clear()
        self._change()

    def can_undo(self) -> bool:
        """
        Returns True if the history is not empty, otherwise False.
        """
        return (len(self.history) > 0)

    def can_redo(self) -> bool:
        """
        Returns False if the future is not empty, otherwise False.
        """
        return (len(self.future) > 0)

Box = Optional[Tuple[int, int, int, int]]

class FPEEditor(wx.Frame):
    """
    Main Forward-Backward Editor Frame.
    """

    TOOLBAR_ICON_SIZE = (32, 32)
    HIST_POSE_CHANGE = "pose_change"

    def __init__(
        self,
        parent,
        video_hdl: cv2.VideoCapture,
        poses: Pose,
        names: List[str],
        plot_settings: Mapping[str, Any],
        crop_box: Box,
        labeling_modes: List[PoseLabeler],
        score_engines: List[ScoreEngine],
        w_id=wx.ID_ANY,
        title="",
        pos=wx.DefaultPosition,
        size=wx.DefaultSize,
        style=wx.DEFAULT_FRAME_STYLE,
        name="FPEEditor"
    ):
        """
        Construct a new FBEditor UI.

        :param parent: The parent window, can be None.
        :param video_hdl: A cv2.VideoCapture, the video to display to the user.
        :param data: A list of 1D numpy arrays, probability data for each body part...
        :param poses: Pose object, being the poses produced by the FB Algorithm...
        :param names: A list of strings, being the names of the body parts...
        :param plot_settings: The video_metadata object from the predictor plugin, includes important point and video
                              settings.
        :param crop_box: The cropping box of the video which poses were actually predicted on.
        :param w_id: The WX ID of the window. Defaults to wx.ID_ANY
        :param title: String title of the window. Defaults to "".
        :param pos: WX Position of the window. Defaults to wx.DefaultPosition.
        :param size: The size of the window. Defaults to wx.DefaultSize.
        :param style: The style of the WX Frame. Look at wx.Frame docs for supported styles.
        :param name: The WX internal name of the window.
        """
        super().__init__(parent, w_id, title, pos, size, style | wx.WANTS_CHARS, name)

        self._history = History()
        self._history.set_change_handler(self._update_hist_btns)
        self._history.register_undoer(self.HIST_POSE_CHANGE, self._pose_doer)
        self._history.register_redoer(self.HIST_POSE_CHANGE, self._pose_doer)
        self._fb_runner = None
        self._frame_exporter = None

        self._main_panel = wx.Panel(self, style=wx.WANTS_CHARS | wx.TAB_TRAVERSAL)
        self._main_sizer = wx.BoxSizer(wx.VERTICAL)
        self._main_sizer.Add(self._main_panel, 1, wx.EXPAND)
        self._splitter_sizer = wx.BoxSizer(wx.VERTICAL)

        self._main_splitter = wx.SplitterWindow(self._main_panel)
        self._sub_sizer = wx.BoxSizer(wx.VERTICAL)
        self._video_splitter = wx.SplitterWindow(self._main_splitter)
        self._side_sizer = wx.BoxSizer(wx.VERTICAL)
        self._sub_panel = wx.Panel(self._main_splitter)

        # Splitter specific settings...
        self._video_splitter.SetSashGravity(0.0)
        self._video_splitter.SetMinimumPaneSize(20)
        self._main_splitter.SetSashGravity(1.0)
        self._main_splitter.SetMinimumPaneSize(20)

        ps = {
            "colormap": plot_settings["colormap"],
            "shape_list": plot_settings["shape_list"],
            "plot_threshold": plot_settings["pcutoff"],
            "point_radius": plot_settings["dotsize"],
            "point_alpha": plot_settings["alphavalue"],
            "line_thickness": plot_settings["line_thickness"]
        }
        self.video_player = PointEditor(
            self._video_splitter,
            video_hdl=video_hdl,
            crop_box=crop_box,
            poses=poses,
            bp_names=names,
            labeling_modes=labeling_modes,
            **ps
        )
        self.video_controls = VideoController(self._sub_panel, video_player=self.video_player.video_viewer)

        with FBProgressDialog(self, inner_msg="Calculating Scores...") as dlg:
            dlg.Show()
            self._score_disp = MultiScoreDisplay(
                self._sub_panel, score_engines, poses, dlg.progress_bar
            )

        self._plot_panel = wx.Panel(self._video_splitter)

        self.plot_button = wx.Button(self._plot_panel, label="Plot This Frame")
        plot_imgs = [wx.Bitmap.FromRGBA(100, 100, 0, 0, 0, 0) for __ in range(poses.get_bodypart_count())]
        self.plot_list = ScrollImageList(self._plot_panel, plot_imgs, wx.VERTICAL, size=wx.Size(200, -1))

        self._side_sizer.Add(self.plot_button, 0, wx.ALIGN_CENTER)
        self._side_sizer.Add(self.plot_list, 1, wx.EXPAND)
        self._plot_panel.SetSizerAndFit(self._side_sizer)

        self._video_splitter.SplitVertically(self._plot_panel, self.video_player, self._plot_panel.GetMinSize().GetWidth())

        self._sub_sizer.Add(self._score_disp, 1, wx.EXPAND)
        self._sub_sizer.Add(self.video_controls, 0, wx.EXPAND)

        self._sub_panel.SetSizerAndFit(self._sub_sizer)

        self._main_splitter.SplitHorizontally(self._video_splitter, self._sub_panel, -self._sub_panel.GetMinSize().GetHeight())
        self._splitter_sizer.Add(self._main_splitter, 1, wx.EXPAND)

        self._build_toolbar()

        self._main_panel.SetSizerAndFit(self._splitter_sizer)
        self.SetSizerAndFit(self._main_sizer)

        self.video_player.video_viewer.set_keyboard_listener(self)
        self.video_controls.set_keyboard_listener(self)
        self._setup_keyboard()

        self.Bind(PointViewNEdit.EVT_POINT_INIT, lambda a: self.video_controls.Enable(False))
        self.Bind(PointViewNEdit.EVT_POINT_END, lambda a: self._refocus(a))
        self.Bind(PointViewNEdit.EVT_POINT_CHANGE, self._on_prob_chg)

        self.video_controls.Bind(PointViewNEdit.EVT_FRAME_CHANGE, self._on_frame_chg)

    def _refocus(self, evt):
        """
        PRIVATE: Refocuses the FBEditor window for accepting keyboard events. Used after disabling all controls during
        a point edit event.
        """
        self.video_controls.Enable(True)
        self.video_controls.SetFocus()

    def _setup_keyboard(self):
        """
        PRIVATE: Connects keyboard events to toolbar actions.
        """
        keyboard_shortcuts = [
            (wx.ACCEL_ALT, ord("Z"), self._undo.GetId()),
            (wx.ACCEL_ALT | wx.ACCEL_SHIFT, ord("Z"), self._redo.GetId()),
            (wx.ACCEL_ALT, ord("S"), self._save.GetId()),
            (wx.ACCEL_ALT, ord("R"), self._run.GetId()),
            (wx.ACCEL_ALT, ord("H"), self._help.GetId()),
            (wx.ACCEL_ALT, wx.WXK_RIGHT, self._f_frame.GetId()),
            (wx.ACCEL_ALT, wx.WXK_LEFT, self._b_frame.GetId()),
            (wx.ACCEL_ALT, ord("E"), self._export_btn.GetId())
        ]

        self._id_to_key = {wid: (mod_key, norm_key) for mod_key, norm_key, wid in keyboard_shortcuts}

        self.SetAcceleratorTable(wx.AcceleratorTable([wx.AcceleratorEntry(*s) for s in keyboard_shortcuts]))

    def _launch_help(self):
        """
        PRIVATE: Launches the help dialog for the FBEditor, which describes how to use the UI and lists all tools
        and keyboard shortcuts.
        """
        entries = []

        for tool, bmp in zip(self._tools, self._bitmaps):
            entries.append((bmp, self._id_to_key.get(tool.GetId(), None),
                            self._toolbar.GetToolShortHelp(tool.GetId())))

        empty_bitmap = wx.Bitmap.FromRGBA(32, 32)

        other_entries = [
            (empty_bitmap, (wx.ACCEL_CTRL, None), "Enter Fast Labeling Mode. Hover over the video to label the "
                                                   "selected point. Hover outside the video to indicate the point is not in the frame."),
            (empty_bitmap, (wx.ACCEL_CTRL | wx.ACCEL_SHIFT, None),
             f"Pressing SHIFT while in fast labeling mode will jump back {PointViewNEdit.JUMP_BACK_AMT} frames."),
            (empty_bitmap, (wx.ACCEL_NORMAL, wx.WXK_SPACE), "Play/Pause the video."),
            (empty_bitmap, (wx.ACCEL_NORMAL, wx.WXK_BACK), "Stop the video."),
            (empty_bitmap, (wx.ACCEL_NORMAL, wx.WXK_RIGHT), "Move 1 frame forward in the video."),
            (empty_bitmap, (wx.ACCEL_NORMAL, wx.WXK_LEFT), "Move 1 frame back in the video."),
            (empty_bitmap, "Left Click/Drag", "Label the selected point within the video."),
            (empty_bitmap, "Right Click", "Mark the selected point as not being in the current frame of the video.")
        ]
        entries.extend(other_entries)


        with HelpDialog(self, entries, self.TOOLBAR_ICON_SIZE) as d:
            d.ShowModal()

    def _build_toolbar(self):
        """
        PRIVATE: Constructs the toolbar, adds all tools to the toolbar, and sets up toolbar events to trigger actions
        within the UI.
        """
        try:
            if wx.GetApp().GetComCtl32Version() >= 600 and wx.DisplayDepth() >= 32:
                # Use the 32-bit images
                wx.SystemOptions.SetOption("msw.remap", 2)
        except Exception:
            pass

        self._toolbar: wx.ToolBar = self.CreateToolBar()

        p_f_bmp = icons.to_wx_bitmap(icons.JUMP_BACK_ICON, icons.JUMP_BACK_ICON_SIZE,
                                     self.GetForegroundColour(), self.TOOLBAR_ICON_SIZE)
        n_f_bmp = icons.to_wx_bitmap(icons.JUMP_FORWARD_ICON, icons.JUMP_FORWARD_ICON_SIZE,
                                     self.GetForegroundColour(), self.TOOLBAR_ICON_SIZE)

        b_bmp = icons.to_wx_bitmap(icons.BACK_ICON, icons.BACK_ICON_SIZE,
                                   self.GetForegroundColour(), self.TOOLBAR_ICON_SIZE)
        f_bmp = icons.to_wx_bitmap(icons.FORWARD_ICON, icons.FORWARD_ICON_SIZE,
                                   self.GetForegroundColour(), self.TOOLBAR_ICON_SIZE)

        run_bmp = icons.to_wx_bitmap(icons.RUN_ICON, icons.RUN_ICON_SIZE,
                                     self.GetForegroundColour(), self.TOOLBAR_ICON_SIZE)
        save_bmp = icons.to_wx_bitmap(icons.SAVE_ICON, icons.SAVE_ICON_SIZE,
                                      self.GetForegroundColour(), self.TOOLBAR_ICON_SIZE)

        turtle_bmp = icons.to_wx_bitmap(icons.TURTLE_ICON, icons.TURTLE_ICON_SIZE,
                                        self.GetForegroundColour(), self.TOOLBAR_ICON_SIZE)

        export_bmp = icons.to_wx_bitmap(icons.DUMP_FRAMES_ICON, icons.DUMP_FRAMES_SIZE,
                                        self.GetForegroundColour(), self.TOOLBAR_ICON_SIZE)

        spin_ctrl = wx.SpinCtrl(self._toolbar, min=1, max=50, initial=PointViewNEdit.DEF_FAST_MODE_SPEED_FRACTION)
        spin_ctrl.SetMaxSize(wx.Size(-1, self.TOOLBAR_ICON_SIZE[1]))
        spin_ctrl.Bind(wx.EVT_SPINCTRL, self._on_spin)

        help_bmp = icons.to_wx_bitmap(icons.HELP_ICON, icons.HELP_ICON_SIZE,
                                      self.GetForegroundColour(), self.TOOLBAR_ICON_SIZE)

        self._b_frame = self._toolbar.CreateTool(wx.ID_ANY, "Prior Detected Frame", p_f_bmp,
                                                 shortHelp="Jump to the prior detected frame.")
        self._toolbar.AddTool(self._b_frame)
        self._f_frame = self._toolbar.CreateTool(wx.ID_ANY, "Next Detected Frame", n_f_bmp,
                                                 shortHelp="Jump to the next detected frame.")
        self._toolbar.AddTool(self._f_frame)
        self._toolbar.AddSeparator()

        self._undo = self._toolbar.CreateTool(wx.ID_ANY, "Undo", b_bmp, shortHelp="Undo the last action.")
        self._toolbar.AddTool(self._undo)
        self._redo = self._toolbar.CreateTool(wx.ID_ANY, "Redo", f_bmp, shortHelp="Redo the last action.")
        self._toolbar.AddTool(self._redo)
        self._toolbar.AddSeparator()

        self._run = self._toolbar.CreateTool(wx.ID_ANY, "Run Frame Passes", run_bmp,
                                             shortHelp="Rerun the frame passes on user modified results.")
        self._toolbar.AddTool(self._run)
        self._save = self._toolbar.CreateTool(wx.ID_ANY, "Save Results", save_bmp,
                                              shortHelp="Save the current results to file.")
        self._toolbar.AddTool(self._save)
        self._toolbar.AddSeparator()

        self._turtle = self._toolbar.CreateTool(wx.ID_ANY, "Edit CTRL Speed: ", turtle_bmp, turtle_bmp,
                                                shortHelp="Modify the labeling speed when CTRL Key is pressed (fast labeling mode).")
        self._toolbar.AddTool(self._turtle)
        self._toolbar.EnableTool(self._turtle.GetId(), False)
        self._toolbar.AddControl(spin_ctrl)
        self._toolbar.AddSeparator()

        self._export_btn = self._toolbar.CreateTool(wx.ID_ANY, "Export Frames", export_bmp,
                                                    shortHelp="Export the current modified frames from the UI.")
        self._toolbar.AddTool(self._export_btn)

        self._help = self._toolbar.CreateTool(wx.ID_ANY, "Help", help_bmp, shortHelp="Display the help dialog.")
        self._toolbar.AddTool(self._help)

        self._update_hist_btns(self._history)
        self.Bind(wx.EVT_TOOL, self.on_tool)

        self._tools = [self._b_frame, self._f_frame, self._undo, self._redo, self._run, self._save, self._turtle,
                       self._export_btn, self._help]
        self._bitmaps = [p_f_bmp, n_f_bmp, b_bmp, f_bmp, run_bmp, save_bmp, turtle_bmp, export_bmp, help_bmp]

        self._toolbar.Realize()

    def _on_spin(self, evt: wx.SpinEvent):
        """
        PRIVATE: Triggered when the value in the wx.SpinStrl in the toolbar is changed by the user.
        """
        self.video_player.video_viewer.set_ctrl_speed_divider(evt.GetPosition())

    def _on_frame_chg(self, evt: PointViewNEdit.FrameChangeEvent):
        """
        PRIVATE: Triggered when the frame in the point edit is changed...
        """
        for prob_disp in self.score_displays:
            prob_disp.set_location(evt.frame)

    def _on_prob_chg(self, evt: PointViewNEdit.PointChangeEvent):
        """
        PRIVATE: Triggered when a probability is changed...
        """
        # Get the new location.
        new_x, new_y, new_prob = evt.new_location

        # Update the probability in the probability displayer and also the point editor...
        old_scores = [
            score.get_data_at(evt.frame) for score in self.score_displays
        ]
        for score in self.score_displays:
            score.update_at(evt.frame, np.nan)

        self.video_player.video_viewer.set_pose(evt.frame, evt.part, evt.new_location)

        self.video_player.Refresh()

        self._history.do(
            self.HIST_POSE_CHANGE,
            (
                evt.labeler,
                (evt.frame, evt.part, *evt.old_location),
                old_scores,
                evt.labeler_data,
                True
            )
        )

    def _update_hist_btns(self, hist_elm):
        """
        PRIVATE: Update the history buttons to match history.
        """
        self._toolbar.EnableTool(self._undo.GetId(), hist_elm.can_undo())
        self._toolbar.EnableTool(self._redo.GetId(), hist_elm.can_redo())

    def set_fb_runner(self, func: Optional[Callable[[], bool]]):
        """
        Set the Forward/Backward runner function, which runs FB on the entire dataset.

        :param func: A callable which accepts no arguments, and return a boolean determining if the history should be
                     cleared. It is assumed this method already has access to all data to rerun FB on the data, and
                     it is also assumed this function will manipulate the widgets of the FB Editor to have the updated
                     data. (Specifically the score_displays, and the video_player.video_viewer).
        """
        self._fb_runner = func

    def set_frame_exporter(self, func: Optional[Callable[[int, str, Path], Tuple[bool, str]]]):
        """
        Set the frame exporting function, which exports current modified frames UI is showing...

        :param func: The function to handle frame exporting. It will be passed an integer being the type of frames to
                     export (0 for pre frame passes and 1 for post frame passes), a string being the format to
                     save the file to ('DLFS' or 'HDF5') and a Path to where the user selected to save the frames, and
                     is expected to return a boolean being if the export succeeded and a string being the error message
                     displayed if it did not.
        """
        self._frame_exporter = func

    @property
    def history(self) -> History:
        """
        Get the history object of this FB Editor, allows the user to add there own custom history events, and also
        manipulate this history.
        """
        return self._history

    def _pose_doer(self, pose_data):
        """
        PRIVATE: Handles pose undo and redo events via the history doer api. Accepts an older/newer state, applies it,
                 and returns the current state.
        """
        labeler, old_loc, old_scores, old_labeler_data, undo = pose_data
        frm, bp, x, y, prob = old_loc

        self.video_player.video_viewer.pause()
        self.video_player.video_viewer.set_offset_frames(frm)
        self.video_player.set_body_part(bp)
        cur_loc = self.video_player.video_viewer.get_pose(frm, bp)
        self.video_player.video_viewer.set_pose(frm, bp, (x, y, prob))

        new_old_scores = [score.get_data_at(frm) for score in self.score_displays]
        for score, value in zip(self.score_displays, old_scores):
            score.update_at(frm, value)

        if(undo):
            labeler_data = labeler.undo(old_labeler_data)
        else:
            labeler_data = labeler.redo(old_labeler_data)

        self.video_player.Refresh()

        return (labeler, (frm, bp, *cur_loc), new_old_scores, labeler_data, not undo)

    def _on_export(self):
        """
        PRIVATE: Triggered when user clicks the export frame toolbar button...
        """
        selection = [
            "Original Frames with Latest User Edits",
            "Frames after Latest Frame Pass Run",
            "Frames after Latest Frame Pass Run, All Data."
        ]

        if(self._frame_exporter is not None):
            with wx.SingleChoiceDialog(self, "Select Frames to Export.", "Frame Type Selection", selection) as sd:
                if(sd.ShowModal() == wx.ID_CANCEL):
                    return

                frame_exp_type = sd.GetSelection()

            with wx.FileDialog(self, "Select FrameStore Save Location",
                               wildcard="DLFS File (*.dlfs)|H5 File (*.h5)",
                               style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fd:
                if(fd.ShowModal() == wx.ID_CANCEL):
                    return

                file_format, ext = ("DLFS", ".dlfs") if(fd.GetFilterIndex() == 0) else ("HDF5", ".h5")
                path = Path(fd.GetPath()).with_suffix(ext)

                res, msg = self._frame_exporter(frame_exp_type, file_format, path)
                if(not res):
                    with wx.MessageDialog(self, msg, "File Export Error", wx.ICON_ERROR | wx.OK) as msgd:
                        msgd.ShowModal()

    @property
    def score_displays(self):
        """
        Get the score engine displays.

        :returns: A list of ScoreEngineDisplayer, being all the probability displays of this editor window.
        """
        return self._score_disp.displays

    def on_tool(self, evt: wx.CommandEvent):
        """
        PRIVATE: Triggered whenever a tool is clicked on in the toolbar....
        """
        if(evt.GetId() == self._undo.GetId()):
            self._history.undo()
        elif(evt.GetId() == self._redo.GetId()):
            self._history.redo()
        elif(evt.GetId() == self._save.GetId()):
            self.Destroy()
        elif(evt.GetId() == self._run.GetId()):
            if(self._fb_runner is not None):
                if(self._fb_runner()):
                    self._history.clear()
        elif(evt.GetId() == self._help.GetId()):
            self._launch_help()
        elif(evt.GetId() == self._export_btn.GetId()):
            self._on_export()
        elif(evt.GetId() == self._f_frame.GetId()):
            self._move_to_poor_label(True)
        elif(evt.GetId() == self._b_frame.GetId()):
            self._move_to_poor_label(False)

    def _move_to_poor_label(self, forward: bool):
        self.video_player.video_viewer.pause()
        current_offset = self.video_player.video_viewer.get_offset_count()
        frame_count = self.video_player.video_viewer.get_total_frames()

        def dist_forward(val):
            if(val <= current_offset):
                val = frame_count + val
            return val - current_offset

        def dist_backward(val):
            if(val >= current_offset):
                val = -frame_count + val
            return current_offset - val

        def get_frame(score):
            if(forward):
                return score.get_next_bad_location()
            else:
                return score.get_prev_bad_location()

        res = int(min(
            (get_frame(score) for score in self.score_displays),
            key=dist_forward if(forward) else dist_backward
        ))
        self.video_player.video_viewer.set_offset_frames(res)


class MultiScoreDisplay(wx.Panel):
    """
    Internal-ish Class:

    A MultiScoreDisplay. Is simply a scrollable list of ScoreEngineDisplayer.
    Convenience class used by the FBEditor class.
    """
    # The number of probability displays to allow at max...
    MAX_HEIGHT_IN_WIDGETS = 4

    def __init__(
        self,
        parent,
        score_engines: List[ScoreEngine],
        poses: Pose,
        progress_bar: ProgressBar,
        w_id=wx.ID_ANY,
        **kwargs
    ):
        """
        Construct a new MultiScoreDisplay.

        :param parent: Parent WX Window...
        :param bp_names: A list of strings, being the names of the body parts.
        :param data: A list of 1D numpy arrays, being the probabilities for each body part.
        :param w_id: The WX window ID, defaults to wx.ID_ANY.
        :param **kwargs: All other arguments are passed to the wx.Panel parent class constructor....
        """
        super().__init__(parent, w_id, **kwargs)

        self._main_sizer = wx.BoxSizer(wx.VERTICAL)

        self._scroll_panel = ScrolledPanel(self, style=wx.VSCROLL)
        self._scroll_sizer = wx.BoxSizer(wx.VERTICAL)

        self.displays = [
            ScoreEngineDisplayer(
                engine, poses, progress_bar, self._scroll_panel
            ) for engine in score_engines
        ]

        for display in self.displays:
            self._scroll_sizer.Add(display, 0, wx.EXPAND)

        self._scroll_panel.SetSizer(self._scroll_sizer)
        self._scroll_panel.SetAutoLayout(True)
        self._scroll_panel.SetupScrolling()

        self._main_sizer.Add(self._scroll_panel, 1, wx.EXPAND)

        self.SetSizer(self._main_sizer)
        self.SetMinSize(wx.Size(
            max(disp.GetMinSize().GetWidth() for disp in self.displays),
            sum(disp.GetMinSize().GetHeight() for disp in self.displays[:self.MAX_HEIGHT_IN_WIDGETS]))
        )