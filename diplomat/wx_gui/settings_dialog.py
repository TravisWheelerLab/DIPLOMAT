"""
Provides a dialog for displaying an arbitrary set of configurable settings to the user to be changed. Utilizes
the :class:`~diplomat.wx_gui.labeler_lib.SettingWidget` API for specifying dialog settings and retrieving results.
"""

from typing import Any, Callable, List, Optional, Union
import numpy as np
import wx
from diplomat.processing import Config
from diplomat.utils.colormaps import DiplomatColormap, to_colormap
from diplomat.wx_gui.labeler_lib import (
    SettingWidget,
    SettingCollection,
    SettingCollectionWidget,
)
import platform
import matplotlib.colors as mpl_colors


class DropDown(SettingWidget):
    """
    A SettingWidget for representing a drop-down, or selection widget. Allows the user to select from a list of
    options.
    """

    def __init__(
        self,
        options: List[Any],
        option_names: Optional[List[str]] = None,
        default: int = 0,
        **kwargs
    ):
        """
        Create a new drop-down widget.

        :param options: The list of objects to select from.
        :param option_names: Optional, a list of names to actually display in the selection box. If not set or set to
                             None, this widget gets display names by calling the `str` function on each object in the
                             options list.
        :param default: The index of the default selected value when the widget is first loaded. Defaults to 0, or the
                        first element in the selection box.
        """
        if len(options) == 0:
            raise ValueError("No options offered!")
        if option_names is None:
            option_names = [str(o) for o in options]
        if len(option_names) != len(options):
            raise ValueError("Options and name arrays don't have the same length.")
        self._options = list(options)
        self._option_names = option_names
        if not (0 <= default < len(options)):
            raise ValueError("Default index is out of bounds!")
        self._default = int(default)
        self._kwargs = kwargs
        self._hook = None
        self._value = default

    def set_hook(self, hook: Callable[[str], None]):
        self._hook = hook

    def get_new_widget(self, parent=None) -> wx.Control:
        # Check if the platform is Windows
        if platform.system() != "Windows":
            # If not Windows, add the style flag to kwargs
            self._kwargs["style"] = wx.LB_SINGLE
        text_list = wx.Choice(parent, choices=self._option_names, **self._kwargs)
        text_list.SetSelection(self._default)

        def val_change(evt):
            sel = text_list.GetSelection()
            if sel == wx.NOT_FOUND:
                text_list.SetSelection(self._default)
                self._value = self._default
            else:
                self._value = sel

            if self._hook is not None:
                self._hook(self._options[self._value])

        text_list.Bind(wx.EVT_CHOICE, val_change)

        return text_list

    def get_value(self) -> Any:
        return self._options[self._value]


def _draw_colormap_entry(dc, rect, cmap):
    fw, fh = dc.GetFont().GetPixelSize()
    x, y, w, h = rect.Get()
    if h <= 0 or w <= 0:
        return
    img_w = fw * 8
    img_h = int(fh * 1.5)
    pad_w = fw

    img = cmap(np.linspace(0, 1, img_w), bytes=True)[..., :3]
    img = np.repeat(img[None], img_h, axis=0)
    bitmap = wx.Bitmap.FromBuffer(img_w, img_h, img.tobytes())

    dc.DrawBitmap(bitmap, x + pad_w, y + (h - img_h) // 2)
    dc.DrawText(cmap.name, x + pad_w * 2 + img_w, y + (h - fh) // 2)


class ColormapListBox(wx.VListBox):
    def __init__(
        self,
        parent,
        id=wx.ID_ANY,
        colormaps: List[DiplomatColormap] = (),
        pos=wx.DefaultPosition,
        size=wx.DefaultSize,
        style=0,
        name="ColormapListBox",
    ):
        super().__init__(parent, id, pos, size, style, name)
        self._colormaps = tuple(colormaps)
        self._colormaps_enabled = [True] * len(colormaps)
        self.SetItemCount(len(self._colormaps))

    def apply_filter(self, filt: Optional[Callable[[DiplomatColormap], bool]] = None):
        self._colormaps_enabled = [
            True if filt is None else filt(cmap) for cmap in self._colormaps
        ]
        self.SetItemCount(len(self._colormaps))
        self.Update()
        self.Refresh()

    def get_selected_colormap(self):
        res = self.GetSelection()
        if res != wx.NOT_FOUND:
            return self._colormaps[res]
        return None

    def get_colormap(self, index: int):
        return self._colormaps[index]

    @property
    def colormaps(self):
        return self._colormaps

    def GetItemRect(self, item):
        return super().GetItemRect(item) if self._colormaps_enabled[item] else wx.Rect()

    def OnMeasureItem(self, n):
        w, h = self.GetFont().GetPixelSize()
        return h * 2 if self._colormaps_enabled[n] else 0

    def OnDrawItem(self, dc, rect, n):
        if not self._colormaps_enabled[n]:
            return
        _draw_colormap_entry(dc, rect, self._colormaps[n])


class ColormapSearch(wx.Panel):

    def generate_search_bitmap(self, size: int, thickness: int):
        bitmap = wx.Bitmap(size, size)
        bc = self.GetBackgroundColour()
        fc = self.GetForegroundColour()
        dc = wx.GCDC(wx.MemoryDC(bitmap))
        dc.SetBackground(wx.Brush(bc, wx.BRUSHSTYLE_SOLID))
        dc.Clear()
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.SetPen(wx.Pen(fc, thickness, wx.PENSTYLE_SOLID))
        tsize = size - thickness * 2
        circ_off = thickness + tsize * 3 // 8
        dc.DrawCircle(circ_off, circ_off, tsize * 3 // 8)
        line_start = circ_off + int(tsize * (3 / 8) / np.sqrt(2))
        line_end = size - thickness
        dc.DrawLine(line_start, line_start, line_end, line_end)
        return bitmap

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fw, fh = self.GetFont().GetPixelSize()
        search_bmp_size = int(fh * 1.5)
        search_bmp_thickness = max(1, fw // 3)
        self.search_bitmap = wx.StaticBitmap(
            self,
            bitmap=self.generate_search_bitmap(search_bmp_size, search_bmp_thickness),
        )
        self.text = wx.TextCtrl(self)
        self._sizer = wx.BoxSizer(wx.HORIZONTAL)
        self._sizer.Add(self.search_bitmap, 0, wx.EXPAND)
        self._sizer.Add(self.text, 1, wx.ALL | wx.EXPAND)
        self.SetSizerAndFit(self._sizer)


class ColormapPopup(wx.ComboPopup):
    def __init__(self, colormaps: List[DiplomatColormap]):
        super().__init__()
        self.colormaps = [to_colormap(c) for c in colormaps]
        self.frame = None
        self.search = None
        self.list = None
        self.value = -1
        self.curent_item = -1

    def OnMotion(self, evt):
        offset = self.list.CalcUnscrolledPosition(evt.GetPosition()[1])
        selected_item = -1
        for i in range(self.list.GetItemCount()):
            item_dim = self.list.OnMeasureItem(i)
            if item_dim > offset:
                selected_item = i
                break
            offset -= item_dim
        if selected_item >= 0:
            self.list.SetSelection(selected_item)
            self.curent_item = selected_item

    def OnLeftDown(self, evt):
        self.value = self.curent_item
        self.Dismiss()

    def OnSearch(self, evt):
        value = self.search.text.GetValue().strip().lower()
        self.list.apply_filter(lambda c: value in c.name.lower())

    def Init(self):
        self.value = -1
        self.curent_item = -1

    def Create(self, parent):
        self.frame = wx.Panel(parent, style=wx.SIMPLE_BORDER)
        self.search = ColormapSearch(self.frame)
        self.list = ColormapListBox(
            self.frame, colormaps=self.colormaps, style=wx.LC_LIST | wx.LC_SINGLE_SEL
        )
        l1 = wx.BoxSizer(wx.VERTICAL)
        l1.Add(self.search, 0, wx.ALL | wx.EXPAND)
        l1.Add(self.list, 1, wx.ALL | wx.EXPAND)
        self.frame.SetSizer(l1)

        self.list.Bind(wx.EVT_MOTION, self.OnMotion)
        self.list.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.search.text.Bind(wx.EVT_TEXT, self.OnSearch)

        return True

    def GetStringValue(self):
        if self.value >= 0:
            return self.list.get_colormap(self.value).name
        return ""

    def SetStringValue(self, value):
        if self.list is None:
            return
        for i, cmap in enumerate(self.list.colormaps):
            if value == cmap.name:
                self.list.SetSelection(i)
                self.value = i
                break

    def PaintComboControl(self, dc, rect):
        if self.value < 0:
            return
        _draw_colormap_entry(dc, rect, self.colormaps[self.value])

    def GetControl(self):
        return self.frame


class ColormapChoice(wx.ComboCtrl):
    def __init__(
        self,
        parent,
        id=wx.ID_ANY,
        colormaps: List[DiplomatColormap] = None,
        pos=wx.DefaultPosition,
        size=wx.DefaultSize,
        style=wx.CB_READONLY,
        validator=wx.DefaultValidator,
        name="ColormapChoice",
    ):
        if colormaps is None:
            from matplotlib import colormaps

            colormaps = sorted(colormaps)
        colormaps = [to_colormap(c) for c in colormaps]
        super().__init__(
            parent,
            id,
            colormaps[0].name if len(colormaps) > 0 else "",
            pos,
            size,
            style,
            validator,
            name,
        )
        self.UseAltPopupWindow(True)
        self._popup = ColormapPopup(colormaps)
        self.SetPopupControl(self._popup)
        min_size = self.GetMinSize()
        w, h = self.GetFont().GetPixelSize()
        self.SetMinSize(wx.Size(min_size.width, h * 2))


class ColormapSelector(SettingWidget):
    """
    A SettingWidget for representing a colormap dropdown, for selecting a colormap.
    """

    def __init__(
        self,
        options: Optional[
            List[Union[DiplomatColormap, str, mpl_colors.Colormap]]
        ] = None,
        default: Optional[Union[DiplomatColormap, str, mpl_colors.Colormap]] = None,
        **kwargs
    ):
        """
        Create a new drop-down widget.

        :param options: The list of objects to select from.
        :param option_names: Optional, a list of names to actually display in the selection box. If not set or set to
                             None, this widget gets display names by calling the `str` function on each object in the
                             options list.
        :param default: The index of the default selected value when the widget is first loaded. Defaults to 0, or the
                        first element in the selection box.
        """
        self._options = []

        if default is not None:
            self._options.append(to_colormap(default))

        if options is None:
            from matplotlib import colormaps

            options = sorted(colormaps)
        if len(options) == 0:
            raise ValueError("No options offered!")
        self._options.extend([to_colormap(op) for op in options])
        self._default = 0
        self._kwargs = kwargs
        self._hook = None
        self._value = self._default

    def set_hook(self, hook: Callable[[DiplomatColormap], None]):
        self._hook = hook

    def get_new_widget(self, parent=None) -> wx.Control:
        text_list = ColormapChoice(parent, wx.ID_ANY, colormaps=self._options)
        text_list.SetValue(self._options[self._value].name)

        def val_change(evt):
            print(evt)
            try:
                sel = text_list.GetValue()
            except RuntimeError:
                return
            for i, cmap in enumerate(self._options):
                if cmap.name == sel:
                    self._value = i
                    break
            else:
                text_list.SetValue(self._options[self._default].name)
                self._value = self._default

            if self._hook is not None:
                self._hook(self._options[self._value])

        text_list.Bind(wx.EVT_COMBOBOX_CLOSEUP, val_change)

        return text_list

    def get_value(self) -> Any:
        return to_colormap(self._options[self._value])


class SettingsDialog(wx.Dialog):
    """
    A dialog of settings. Allows displaying a :class:`~diplomat.wx_gui.labeler_lib.SettingCollection` to a user in
    a dialog.
    """

    def __init__(
        self, *args, title: str = "Settings", settings: SettingCollection, **kwargs
    ):
        """
        Create a new dialog.

        :param title: The title of the dialog.
        :param settings: The settings to display.

        Additional positional and keyword arguments are passed directly to :class:`wx.Dialog` constructor.
        """
        super().__init__(
            *args,
            title=title,
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
            **kwargs
        )

        self._parent_layout = wx.BoxSizer(wx.VERTICAL)

        self._settings = settings
        self._setting_widget = SettingCollectionWidget(
            self, title=title, collapsable=False
        )
        self._setting_widget.set_setting_collection(settings)
        self._parent_layout.Add(
            self._setting_widget, proportion=1, flag=wx.EXPAND | wx.ALL
        )

        self._buttons = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        self._parent_layout.Add(self._buttons, proportion=0, flag=wx.EXPAND | wx.ALL)

        self.SetSizerAndFit(self._parent_layout)

    def get_values(self) -> Config:
        """
        Get the values selected by the user in the settings dialog.

        :return: A :class:`~diplomat.processing.containers.Config` object, containing the values selected by the user.
        """
        return self._settings.get_values()


def _test_colormap_selector():
    app = wx.App()

    with SettingsDialog(
        None, title="Settings", settings=SettingCollection(colormap=ColormapSelector())
    ) as dlg:
        if dlg.ShowModal() == wx.ID_OK:
            print(dlg.get_values())
            print(dlg.get_values().colormap.name)
        else:
            print("Canceled...")


if __name__ == "__main__":
    _test_colormap_selector()
