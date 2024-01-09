from typing import Any, Callable, List, Optional
import wx
from diplomat.processing import Config
from diplomat.wx_gui.labeler_lib import SettingWidget, SettingCollection, SettingCollectionWidget


class DropDown(SettingWidget):
    def __init__(self, options: List[Any], option_names: Optional[List[str]] = None, default: int = 0, **kwargs):
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
        text_list = wx.Choice(parent, choices=self._option_names, style=wx.LB_SINGLE, **self._kwargs)
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
    def __init__(self, *args, title: str = "Settings", settings: SettingCollection, **kwargs):
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
        return self._settings.get_values()
