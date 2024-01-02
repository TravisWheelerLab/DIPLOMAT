"""
Includes the help dialog for DIPLOMAT's main gui editor. Displays toolbar actions and keyboard shortcuts.
"""

from typing import Tuple, Optional, List
from diplomat.utils._bit_or import _bit_or
import wx

Opt = Optional


def is_ascii(s):
    """
    Meant for internal use: Checks if the passed string is made of only ascii characters.
    """
    return all(ord(c) < 128 for c in s)


class HelpDialog(wx.Dialog):
    """
    A custom wx.Dialog. This dialog can be used to describe UI shortcuts and their related keyboard shortcuts, allowing
    the user to see how to use a UI.
    """
    # Modifier key to string for displaying in the help dialog.
    MOD_TO_STR = {
        wx.ACCEL_CTRL: "Ctrl",
        wx.ACCEL_ALT: "Alt",
        wx.ACCEL_SHIFT: "Shift"
    }

    # These get displayed for keyboard shortcuts which are not alphanumeric...
    MIS_KEYS = {
        wx.WXK_RIGHT: "\u2192",
        wx.WXK_LEFT: "\u2190",
        wx.WXK_UP: "\u2191",
        wx.WXK_DOWN: "\u2193",
        wx.WXK_SPACE: "Space",
        wx.WXK_BACK: "Backspace",
    }

    def __init__(self, parent, entries: List[Tuple[wx.Bitmap, Opt[Tuple[int, int]], str]], image_sizes: Tuple[int, int],
                 wid=wx.ID_ANY, title="Help", pos=wx.DefaultPosition, size=wx.DefaultSize,
                 style=_bit_or(wx.DEFAULT_DIALOG_STYLE, wx.RESIZE_BORDER), name="helpDialog"):
        """
        Construct a new help dialog.

        :param parent: The parent window.
        :param entries: A list of tuples, each tuple containing a wx.Bitmap being the action icon, an optional
                        wx.AcceleratorTable style shortcut (None to disable), and a string description describing the
                        action. These will be parsed and shown to the user in the help dialog as a table.
        :param image_sizes: The size of all of the images passed to this method, a tuple of two integers.
        :param wid: wx ID of the window, and integer. Defaults to wx.ID_ANY.
        :param title: The string title of the help dialog, defaults to "Help".
        :param pos: WX Position of dialog. Defaults to wx.DefaultPosition.
        :param size: WX Size of the dialog. Defaults to wx.DefaultSize.
        :param style: WX Dialog Style. See wx.Dialog docs for possible options.
        :param name: WX internal name of widget.
        """
        super().__init__(parent, wid, title, pos, size, style, name)

        self._main_sizer = wx.BoxSizer(wx.VERTICAL)
        self._main_panel = wx.Panel(self)

        self._sub_sizer = wx.BoxSizer(wx.VERTICAL)
        self._list = wx.ListCtrl(self._main_panel, style=wx.LC_REPORT | wx.LC_ALIGN_TOP)

        self._list.AppendColumn("Tool Icon")
        self._list.AppendColumn("Shortcut")
        self._list.AppendColumn("Description")

        self._img_lst = wx.ImageList(*image_sizes, True)
        for icon, shortcut, desc in entries:
            self._img_lst.Add(icon)
        self._list.SetImageList(self._img_lst, wx.IMAGE_LIST_SMALL)

        for i, (icon, shortcut, desc) in enumerate(entries):
            self._list.InsertItem(i, i)
            self._list.SetItem(i, 1, self.shortcut_to_string(shortcut))
            self._list.SetItem(i, 2, desc)

        # Resize the columns:
        for i in range(self._list.GetColumnCount()):
            self._list.SetColumnWidth(i, wx.LIST_AUTOSIZE)

        self._sub_sizer.Add(self._list, 1, wx.EXPAND)
        self._main_panel.SetSizerAndFit(self._sub_sizer)

        self._btns = self.CreateButtonSizer(wx.CLOSE)

        self._main_sizer.Add(self._main_panel, 1, wx.EXPAND)
        if(self._btns is not None):
            self._main_sizer.Add(self._btns, 0, wx.EXPAND)
        self.SetSizerAndFit(self._main_sizer)

    @classmethod
    def shortcut_to_string(cls, shortcut: Opt[Tuple[int, int]]):
        """
        Meant for internal use: Converts a wx.AcceleratorTable style shortcut to a string for display to the user.
        """
        if(shortcut is None):
            return ""
        if(isinstance(shortcut, str)):
            return shortcut

        accel, letter = shortcut

        key_list = [string for accel_btn, string in cls.MOD_TO_STR.items() if((accel & accel_btn) == accel_btn)]

        if(letter is not None):
            letter_c = chr(letter)
            if(letter_c.isalnum() and is_ascii(letter_c) and (not letter_c.isspace())):
                key_list.append(letter_c)
            else:
                letter_c = cls.MIS_KEYS.get(letter, None)
                if(letter_c is not None):
                    key_list.append(letter_c)

        if(len(key_list) == 0):
            return ""

        return "+".join(key_list)