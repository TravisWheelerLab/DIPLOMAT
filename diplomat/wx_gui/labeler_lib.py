"""
Provides abstract and utility classes for creating GUI labelers. These allow the user to edit the source confidence
maps by clicking and dragging over the video.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Callable, Dict
import wx
from diplomat.processing import Config
from wx.lib.agw import floatspin
from diplomat.utils._bit_or import _bit_or


class SettingWidget(ABC):
    """
    Represents a setting that can be configured by the user via a wx Control.
    """
    @abstractmethod
    def set_hook(self, hook: Callable[[], None]):
        """
        Set the hook function for this setting, this is called whenever the
        setting is changed.
        """
        pass

    @abstractmethod
    def get_new_widget(self, parent = None) -> wx.Control:
        """
        Get a widget capable of changing this setting.

        :param parent: The parent container this wxWidget will be placed in.

        :returns: A wx.Control, to be used to update this setting in the
                  user interface.
        """
        pass

    @abstractmethod
    def get_value(self) -> Any:
        """
        Get the current value of this setting.

        :returns: Any, the current set value of this setting as based on the
                  wx control.
        """
        pass


class Slider(SettingWidget):
    """
    A setting which displays a slider for the user to interact with.
    Allows the user to select over a range of integers.
    """
    def __init__(
        self,
        minimum: int,
        maximum: int,
        default: int = None,
        style: int = _bit_or(wx.SL_HORIZONTAL, wx.SL_LABELS),
        **kwargs
    ):
        """
        Create a new slider setting.

        :param minimum: An integer, the minimum value of the slider.
        :param maximum: An integer, the maximum bound of the slider.
        :param default: An optional integer, the initial value of the slider.
                        If None, the default value is the minimum value of the
                        slider.
        :param style: A wxWidgets style flag (integer), adjust the style of the
                      slider as displayed in the UI.
        """
        self._value = default if(default is not None) else minimum
        self._params = (minimum, maximum)
        self._style = style
        self._kwargs = kwargs
        self._hook = None

    def set_hook(self, hook: Callable[[Any], None]):
        self._hook = hook

    def get_new_widget(self, parent = None) -> wx.Control:
        slider = wx.Slider(
            parent, value=self._value,
            minValue=self._params[0],
            maxValue=self._params[1],
            style=self._style,
            **self._kwargs
        )

        def val_change(evt):
            self._value = slider.GetValue()
            if(self._hook is not None):
                self._hook(self._value)

        slider.Bind(wx.EVT_SLIDER, val_change)
        return slider

    def get_value(self) -> int:
        return self._value


def first_non_none(*vals):
    return next((v for v in vals if(v is not None)), None)


class FloatSpin(SettingWidget):
    """
    A setting which displays a spin control for the user to interact with.
    Can handle any floating point values, both bounded and unbounded.
    """
    def __init__(
        self,
        minimum: float = None,
        maximum: float = None,
        default: float = None,
        increment: float = 1,
        digits: int = -1,
        **kwargs
    ):
        """
        Create a new floating point spinner setting.

        :param minimum: Minimum value of the spinner, or None if unbounded.
        :param maximum: Maximum value of the spinner, or None if unbounded.
        :param default: The default value. If None, uses the minimum if it is
                        not None, otherwise the maximum if it is not None, and
                        otherwise 0.
        :param increment: The value the change the spin box by when one of
                          the increment buttons is clicked, defaults to 1
        :param digits: The number of decimal places to resolve numbers to.
                       Defaults to -1 or the maximum possible number of
                       decimal places.
        """
        self._value = first_non_none(default, minimum, maximum, 0)
        self._hook = None
        self._args = dict(
            min_val=minimum,
            max_val=maximum,
            value=self._value,
            increment=increment,
            digits=digits,
            **kwargs
        )

    def set_hook(self, hook: Callable[[Any], None]):
        self._hook = hook

    def get_new_widget(self, parent=None) -> wx.Control:
        self._args["value"] = self._value
        float_spin = floatspin.FloatSpin(parent, **self._args)

        def update(evt):
            self._value = float_spin.GetValue()
            if(self._hook is not None):
                self._hook(self._value)

        float_spin.Bind(floatspin.EVT_FLOATSPIN, update)
        return float_spin

    def get_value(self) -> Any:
        return self._value


class SettingCollection:
    """
    Represents a collection of named SettingWidgets. Widget values can be
    extacted from the ui using get_values.
    """
    def __init__(self, **values):
        """
        Create a new Setting Collection.

        :param values: A set of keyword arguments, the name of the argument
                       is the name of the setting, and the value should be a
                       SettingWidget.
        """
        for name, setting_widget in values.items():
            if(not isinstance(setting_widget, SettingWidget)):
                raise ValueError("Must pass arguments that are names to setting widgets!")
        self.widgets: Dict[str, SettingWidget] = values

    def get_values(self) -> Config:
        """
        Get the values of all of the settings.

        :returns: A Config object, or a dictionary with attribute style lookup
                  of the settings in this SettingCollection. The names match
                  those pased to the constructor, and the values are the
                  values currently stored in the SettingWidgets...
        """
        return Config({k: w.get_value() for k, w in self.widgets.items()})


class PoseLabeler(ABC):
    """
    A PoseLabeler represents a labeling mode in the UI.

    A pose labeler takes a user input at a given location in the video, and
    returns a new pose prediction based on the user input and additonal
    internal information. This allows for 'smart' labelers to be created.
    """
    @abstractmethod
    def predict_location(
        self,
        frame_idx: int,
        bp_idx: int,
        x: float,
        y: float,
        probability: float
    ) -> Tuple[Any, Tuple[float, float, float]]:
        """
        Predict the location of a user input, while not changing the internal
        state of the frames. Used to display the next location of the point
        as the user drags their mouse around the screen.

        :param frame_idx: The index of the current frame the pose labeling UI is on.
        :param bp_idx: The index of the current body part the pose labeling UI is on.
        :param x: The x location the user clicked in video coordinates, or None if
                  the user indicated the body part is not in this frame.
        :param y: The y location the user clicked in video, or None if
                  the user indicated the body part is not in this frame.
        :param probability: The probability of this prediction, 1 if the user
                            selected a location in the video frame, None
                            otherwise.

        :returns: A tuple containing the following information:

                   - Any data, representing the new state this pose labeler
                     would set if this prediction is eventually finalized,
                     is passed to the pose_change method on finalization.
                   - A tuple of 3 floats, being the location (x, y) of the
                     prediction in the video, and the probability.
                     This is where the point is displayed in the UI. Must be
                     floats, set the probability to 0 to avoid plotting a
                     point.
        """
        pass

    @abstractmethod
    def pose_change(
        self,
        new_state: Any
    ) -> Any:
        """
        Finalize a user change, updating any internal state or frame storage
        to enforce the user labeling.

        :param new_state: The state returned by 'predict_location' to finalize.

        :returns: Any data, information needed if the user ever want to undo
                  this labeling, passed to the 'undo' method.
        """
        pass

    @abstractmethod
    def undo(self, data: Any) -> Any:
        """
        Undo a pose change handled by this pose labeler.

        :param data: Data returned from 'pose_change' or 'redo' to handle this
                     undo event if it ever happened.

        :return: Any data, which will be passed to 'redo' if the user decides
                 to redo this labeling in the UI.
        """
        pass

    @abstractmethod
    def redo(self, data: Any) -> Any:
        """
        Redo a pose change handled by this pose labeler.

        :param data: Data returned from 'undo' to handle this
                     redo event if it ever happened.

        :return: Any data, which will be passed to 'undo' if the user decides
                 to undo this labeling in the UI.
        """
        pass

    def get_settings(self) -> Optional[SettingCollection]:
        """
        Get the settings for this pose labeler. Should return None or a
        SettingCollection, which contains SettingWidgets. These will be
        automatically added to the UI when this labeling mode is selected.

        :returns: A SettingCollection or None. The default implementation
                  returns None, indication this plugin has no configurable
                  settings to place into the UI.
        """
        return None

    def get_display_name(self):
        """
        Get the display name of the pose labeler as to be displayed in the UI.

        :returns: The display name. The default implementation returns the
                  class name with a space inserted before every capital letter.
        """
        return "".join([
            f" {c}" if(65 <= ord(c) <= 90) else c for c in type(self).__name__
        ]).strip()

    @classmethod
    def supports_multi_label(cls) -> bool:
        """
        Get if this pose labeler supports editing multiple parts at once...

        :return: A boolean, true if this labeler wants to allow the user to manipulate multiple parts at once. Defaults
                 to false.
        """
        return False


class SettingCollectionWidget(wx.Control):
    """
    A widget for displaying the settings of a pose labeler. Defaults
    to an empty widget, as no pose labeler is set.
    """
    EXPAND_CHAR = "  ▼"
    RETRACT_CHAR = "  ▲"

    def __init__(
        self,
        *args,
        title: str = "Advanced Settings",
        collapsable: bool = True,
        **kwargs
    ):
        """
        Create a new empty settings displaying widget.
        """
        super().__init__(*args, **kwargs)
        self._title = title
        self._shown = not collapsable

        self._collapse_button = wx.Button(self, label=title + self.EXPAND_CHAR)
        if(not collapsable):
            self._collapse_button.SetLabel(title)
            self._collapse_button.Enable(False)

        self._collapse_button.SetFont(
            self._collapse_button.GetFont().MakeSmaller()
        )
        w, h = self._collapse_button.GetSize()
        self._collapse_button.SetMinSize(wx.Size(w, int(h * 0.8)))

        self._selected_settings = None

        self._main_layout = wx.BoxSizer(wx.VERTICAL)
        self._main_layout.Add(self._collapse_button, 0, wx.EXPAND)
        self.Bind(wx.EVT_BUTTON, self._expand_panel)
        self.SetSizerAndFit(self._main_layout)

        self.Layout()

    def is_shown(self) -> bool:
        return self._shown

    def clear(self):
        for i in range(1, self._main_layout.GetItemCount()):
            item = self._main_layout.GetItem(1)
            item.DeleteWindows()
            self._main_layout.Detach(1)

    def _show(self):
        self._shown = True
        for i in range(1, self._main_layout.GetItemCount()):
            self._main_layout.Show(i)

    def _hide(self):
        self._shown = False
        for i in range(1, self._main_layout.GetItemCount()):
            self._main_layout.Hide(i)

    def _expand_panel(self, evt):
        if(self.is_shown()):
            self._collapse_button.SetLabel(self._title + self.EXPAND_CHAR)
            self._hide()
        else:
            self._collapse_button.SetLabel(self._title + self.RETRACT_CHAR)
            self._show()
        self._force_layout_fix(evt)

    def _force_layout_fix(self, evt):
        # This is stupid, but by setting the sizer again we force the
        # window to resize to fit everything...
        self.SetSizerAndFit(self._main_layout)
        self.Layout()
        # This is also stupid...
        window = wx.GetTopLevelParent(self)
        w, h = window.GetSize()
        window.Layout()
        window.SetSize(w + 1, h + 1)
        window.SetSize(w, h)

    def set_setting_collection(self, collection: SettingCollection):
        """
        Set the setting collection to display in this setting collection
        widget...

        :param collection: The SettingCollection to display the settings of.
        """
        self._selected_settings = collection

        self.clear()

        # Load the new plugin...
        if(collection is not None):
            for name, widget_gen in self._selected_settings.widgets.items():
                nice_name = " ".join(w.capitalize() for w in name.split("_")) + ":"
                label = wx.StaticText(self, label=nice_name)
                control = widget_gen.get_new_widget(self)
                self._main_layout.Add(label, 0, wx.EXPAND)
                self._main_layout.Add(control, 0, wx.EXPAND)

        if(self.is_shown()):
            self._show()
        else:
            self._hide()
        self._force_layout_fix(None)


def _test_setting_viewer():
    app = wx.App()

    setting_collection = SettingCollection(
        first_setting = Slider(0, 100, 50),
        second_setting = FloatSpin(0, 100, 10, 1, 3)
    )

    window = wx.Frame(None, wx.ID_ANY, "Hello World!")
    sizer = wx.BoxSizer(wx.VERTICAL)

    setting_widget = SettingCollectionWidget(window, title="Settings:")
    setting_widget.set_setting_collection(setting_collection)
    setting_widget.set_setting_collection(setting_collection)
    label = wx.StaticText(window, label="Example Text Below...")

    sizer.Add(setting_widget, 0, wx.EXPAND)
    sizer.Add(label, 0, wx.EXPAND)
    window.SetSizerAndFit(sizer)

    window.Show()
    app.MainLoop()


if(__name__ == "__main__"):
    _test_setting_viewer()
