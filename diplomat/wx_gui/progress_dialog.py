"""
Provides a dialog wrapping :class:`~diplomat.wx_gui.progress_bar.TqdmWxPanel`. Allows for displaying progress in
a standalone manner (no additional GUI elements are needed).
"""

import wx
from diplomat.wx_gui.progress_bar import TqdmWxPanel


class FBProgressDialog(wx.Dialog):
    """
    A custom wx.Dialog for displaying progress. Uses a wx.TqdmWxPanel internally, which can be accessed via the
    .progress_bar property to display progress...
    """
    def __init__(self, parent=None, wid=wx.ID_ANY, title="Progress", inner_msg: str = "Rerunning Frame Pass Engine:",
                 pos=wx.DefaultPosition, size=wx.DefaultSize, style=wx.DEFAULT_DIALOG_STYLE, name="ProgressDialog"):
        """
        Construct a new FBProgressDialog.

        :param parent: The parent window.
        :param wid: wx ID of the window, and integer. Defaults to wx.ID_ANY.
        :param title: The string title of the progress dialog, defaults to "Progress".
        :param inner_msg: A string, the inner message to display inside the dialog. Defaults to
                          "Rerunning FB Algorithm:"
        :param pos: WX Position of dialog. Defaults to wx.DefaultPosition.
        :param size: WX Size of the dialog. Defaults to wx.DefaultSize.
        :param style: WX Dialog Style. See wx.Dialog docs for possible options.
        :param name: WX internal name of widget.
        """
        super().__init__(parent, wid, title, pos, size, style, name)

        self._sizer = wx.BoxSizer(wx.VERTICAL)

        self._label1 = wx.StaticText(self, label=inner_msg)
        self.progress_bar = TqdmWxPanel(self)

        self._sizer.Add(self._label1, 0, wx.ALIGN_CENTER)
        self._sizer.Add(self.progress_bar, 0, wx.EXPAND)

        self.SetSizerAndFit(self._sizer)
        min_size = self.GetMinSize()
        self._sizer.SetMinSize(wx.Size(self.progress_bar.GetTextExtent("0" * 80).GetWidth(), min_size.GetHeight()))
        self.SetSize(self.GetMinSize())
        self.SendSizeEvent()

        # We bind the close event to nothing so the user can't close the dialog mid-progress causing an exception,
        # code needs to use Destroy() method to close this dialog... or rebind to close to allow the user to close the
        # dialog...
        self.Bind(wx.EVT_CLOSE, lambda evt: None)

    def set_inner_message(self, text: str):
        self._label1.SetLabel(text)