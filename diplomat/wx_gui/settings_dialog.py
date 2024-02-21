"""
Provides a dialog for displaying an arbitrary set of configurable settings to the user to be changed. Utilizes
the :class:`~diplomat.wx_gui.labeler_lib.SettingWidget` API for specifying dialog settings and retrieving results.
"""
from typing import Any, Callable, List, Optional
import wx
from diplomat.processing import Config
from diplomat.wx_gui.labeler_lib import SettingWidget, SettingCollection, SettingCollectionWidget
import platform


class DropDown(SettingWidget):
    """
    A SettingWidget for representing a drop-down, or selection widget. Allows the user to select from a list of
    options.
    """
    def __init__(self, options: List[Any], option_names: Optional[List[str]] = None, default: int = 0, **kwargs):
        """
        Create a new drop-down widget.

        :param options: The list of objects to select from.
        :param option_names: Optional, a list of names to actually display in the selection box. If not set or set to
                             None, this widget gets display names by calling the `str` function on each object in the
                             options list.
        :param default: The index of the default selected value when the widget is first loaded. Defaults to 0, or the
                        first element in the selection box.
        """
        if(len(options) == 0):
            raise ValueError("No options offered!")
        if(option_names is None):
            option_names = [str(o) for o in options]
        if(len(option_names) != len(options)):
            raise ValueError("Options and name arrays don't have the same length.")
        self._options = list(options)
        self._option_names = option_names
        if(not (0 <= default < len(options))):
            raise ValueError("Default index is out of bounds!")
        self._default = int(default)
        self._kwargs = kwargs
        self._hook = None
        self._value = default

    def set_hook(self, hook: Callable[[str], None]):
        self._hook = hook

    def get_new_widget(self, parent=None) -> wx.Control:
                # Check if the platform is Windows
        if platform.system() != 'Windows':
            # If not Windows, add the style flag to kwargs
            self._kwargs['style'] = wx.LB_SINGLE
        text_list = wx.Choice(parent, choices=self._option_names, **self._kwargs)
        text_list.SetSelection(self._default)

        def val_change(evt):
            sel = text_list.GetSelection()
            if(sel == wx.NOT_FOUND):
                text_list.SetSelection(self._default)
                self._value = self._default
            else:
                self._value = sel

            if(self._hook is not None):
                self._hook(self._options[self._value])

        text_list.Bind(wx.EVT_CHOICE, val_change)

        return text_list

    def get_value(self) -> Any:
        return self._options[self._value]


class SettingsDialog(wx.Dialog):
    """
    A dialog of settings. Allows displaying a :class:`~diplomat.wx_gui.labeler_lib.SettingCollection` to a user in
    a dialog.
    """
    def __init__(self, *args, title: str = "Settings", settings: SettingCollection, **kwargs):
        """
        Create a new dialog.

        :param title: The title of the dialog.
        :param settings: The settings to display.

        Additional positional and keyword arguments are passed directly to :class:`wx.Dialog` constructor.
        """
        super().__init__(*args, title=title, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER, **kwargs)

        self._parent_layout = wx.BoxSizer(wx.VERTICAL)

        self._settings = settings
        self._setting_widget = SettingCollectionWidget(self, title=title, collapsable=False)
        self._setting_widget.set_setting_collection(settings)
        self._parent_layout.Add(self._setting_widget, proportion=1, flag=wx.EXPAND | wx.ALL)

        self._buttons = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        self._parent_layout.Add(self._buttons, proportion=0, flag=wx.EXPAND | wx.ALL)

        self.SetSizerAndFit(self._parent_layout)

    def get_values(self) -> Config:
        """
        Get the values selected by the user in the settings dialog.

        :return: A :class:`~diplomat.processing.containers.Config` object, containing the values selected by the user.
        """
        return self._settings.get_values()
