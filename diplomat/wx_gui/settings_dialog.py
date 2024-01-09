import wx

from diplomat.processing import Config
from diplomat.wx_gui.labeler_lib import SettingCollection, SettingCollectionWidget


class SettingsDialog(wx.Dialog):
    def __init__(self, *args, title: str = "Settings", settings: SettingCollection, **kwargs):
        super().__init__(*args, title=title, **kwargs)

        self._parent_layout = wx.BoxSizer(wx.VERTICAL)

        self._settings = settings
        self._setting_widget = SettingCollectionWidget(self, title=title, collapsable=False)
        self._setting_widget.set_setting_collection(settings)
        self._parent_layout.Add(self._setting_widget, proportion=1, flag=wx.EXPAND | wx.ALL)

        self._buttons = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        self._parent_layout.Add(self._buttons)

        self.SetSizerAndFit(self._parent_layout)
        self.SetSize(self.GetMinSize())

    def get_values(self) -> Config:
        return self._settings.get_values()
