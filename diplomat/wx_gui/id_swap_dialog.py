"""
Contains the identity swap dialog. Displayed in the stripped down version of DIPLOMAT's UI (shown when running
:cli:`diplomat tweak`).
"""

import wx
from typing import List, Iterable, Tuple
from dataclasses import dataclass
from diplomat.wx_gui.point_edit import WxDotShapeDrawer


class IdSwapDialog(wx.Dialog):
    """
    Identity swap dialog. Provides a dialog which can get reordered parts and individuals.
    """
    def __init__(
        self,
        *args,
        num_outputs: int,
        labels: List[str],
        colors: List[Tuple[int, int, int, int]],
        shapes: List[str],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.SetWindowStyle(self.GetWindowStyle() | wx.RESIZE_BORDER)

        self._num_outputs = num_outputs
        self._labels = labels

        self._outer_sizer = wx.BoxSizer(wx.VERTICAL)
        self._scroller = wx.ScrolledWindow(self, wx.ID_ANY, size=wx.Size(200, 200), style=wx.VSCROLL)
        self._outer_sizer.Add(self._scroller, 1, wx.ALL | wx.EXPAND)

        self._sizer = wx.BoxSizer(wx.VERTICAL)
        # Build the dialog....
        self._individuals = DragZone(num_outputs, labels, shapes, colors, parent=self._scroller, id=wx.ID_ANY)
        self._sizer.Add(self._individuals, 1, wx.EXPAND | wx.CENTER)

        self._btn_sizer = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        if(self._btn_sizer is not None):
            self._outer_sizer.Add(self._btn_sizer, 0, wx.ALL | wx.EXPAND)

        self._scroller.SetSizer(self._sizer)
        self._scroller.EnableScrolling(True, True)
        self._scroller.SetScrollbars(1, 1, 1, 1)
        self.SetSizerAndFit(self._outer_sizer)

    def get_proposed_order(self) -> List[int]:
        return [part.index for part in self._individuals.iter_parts()]


_PADDING = 3


@dataclass
class Part:
    x: int
    y: int
    index: int
    name: str
    shape: str
    color: Tuple[int, int, int, int]

    def __post_init__(self):
        self.dragging = False
        self.hover = False
        self.highlight = False
        self.background_color = (0, 0, 0, 0)

    def draw(self, dc: wx.DC, canvas_width: int, canvas_height: int):
        w, h = dc.GetTextExtent(self.name)

        highlight_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT)
        background_color = highlight_color if(self.dragging or self.highlight) else wx.Colour(*self.background_color)
        dc.SetBrush(wx.Brush(background_color))
        border_color = wx.SYS_COLOUR_BTNSHADOW if(not self.hover or self.dragging) else wx.SYS_COLOUR_HIGHLIGHT
        dc.SetPen(wx.Pen(wx.SystemSettings.GetColour(border_color), 1, wx.PENSTYLE_SOLID))
        dc.DrawRoundedRectangle(self.x, self.y, w + int(h * 2.5), h * 2, _PADDING)

        dc.DrawText(self.name, self.x + h * 2, self.y + h // 2)

        dc.SetBrush(wx.Brush(wx.Colour(*self.color), wx.BRUSHSTYLE_SOLID))
        dc.SetPen(wx.TRANSPARENT_PEN)
        WxDotShapeDrawer(dc)[self.shape](self.x + h, self.y + h, h // 2)

    def size(self, dc: wx.DC, canvas_width: int, canvas_height: int) -> Tuple[int, int]:
        w, h = dc.GetTextExtent(self.name)
        return w + int(h * 2.5), h * 2

    def get_mouseover(self, dc: wx.DC, canvas_width: int, canvas_height: int, mx: int, my: int):
        w, h = self.size(dc, canvas_width, canvas_height)
        if((self.x <= mx <= self.x + w) and (self.y <= my <= self.y + h)):
            return self
        return None


@dataclass
class Body:
    x: int
    y: int
    name: str
    shape: str
    parts: List[Part]

    def __post_init__(self):
        self.dragging = False
        self.hover = False
        self.highlight = False
        self.background_color = (0, 0, 0, 0)

    def draw(self, dc: wx.DC, canvas_width: int, canvas_height: int):
        w, h = dc.GetTextExtent(self.name)

        x_cur, y_cur = self.x + _PADDING, self.y + _PADDING + h * 2
        h_jump = 0

        # Part layout engine code...
        for part in self.parts:
            pw, ph = part.size(dc, canvas_width, canvas_height)
            if ((x_cur + pw + _PADDING) > (self.x + canvas_width)):
                y_cur += h_jump + _PADDING
                x_cur = self.x + _PADDING

            if(not part.dragging):
                part.x = x_cur
                part.y = y_cur

            h_jump = max(h_jump, ph)
            x_cur += pw + _PADDING

        highlight_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT)
        background_color = highlight_color if (self.dragging or self.highlight) else wx.Colour(*self.background_color)
        dc.SetBrush(wx.Brush(background_color))
        border_color = wx.SYS_COLOUR_BTNSHADOW if(not self.hover or self.dragging) else wx.SYS_COLOUR_HIGHLIGHT
        dc.SetPen(wx.Pen(wx.SystemSettings.GetColour(border_color), 1, wx.PENSTYLE_SOLID))
        dc.DrawRoundedRectangle(self.x, self.y, canvas_width, y_cur + h_jump + _PADDING - self.y, _PADDING)

        dc.DrawText(self.name, self.x + canvas_width // 2 - w // 2, self.y + h // 2)

        dc.SetBrush(wx.Brush(dc.GetTextForeground(), wx.BRUSHSTYLE_SOLID))
        dc.SetPen(wx.TRANSPARENT_PEN)
        WxDotShapeDrawer(dc)[self.shape](self.x + canvas_width // 2 - w // 2 - h // 2 - _PADDING, self.y + h, h // 2)

        for part in sorted(self.parts, key=lambda a: a.dragging):
            part.draw(dc, canvas_width, canvas_height)

    def size(self, dc: wx.DC, canvas_width: int, canvas_height: int):
        w, h = dc.GetTextExtent(self.name)

        x_cur, y_cur = _PADDING, _PADDING
        h_jump = 0

        for part in self.parts:
            pw, ph = part.size(dc, canvas_width, canvas_height)
            if((x_cur + pw + _PADDING) > canvas_width):
                y_cur += h_jump + _PADDING
                x_cur = _PADDING
            h_jump = max(h_jump, ph)
            x_cur += pw + _PADDING

        return canvas_width, y_cur + h * 2 + h_jump + _PADDING

    def get_mouseover(self, dc: wx.DC, canvas_width: int, canvas_height: int, mx: int, my: int):
        for part in self.parts:
            obj = part.get_mouseover(dc, canvas_width, canvas_height, mx, my)
            if(obj is not None):
                return obj

        w, h = self.size(dc, canvas_width, canvas_height)
        if((self.x <= mx <= self.x + w) and (self.y <= my <= self.y + h)):
            return self
        return None


class DragZone(wx.Control):
    def __init__(self, num_outputs: int, parts: List[str], shapes: List[str], colors: List, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bodies = []
        self._part_count = len(parts)
        self._num_outputs = num_outputs

        self._pressed_obj = None
        self._owner_body = None
        self._hover_obj = None
        self._highlight_obj = None
        self._press_offset = (0, 0)
        self._offset_shapes = []

        for i in range(num_outputs):
            body = Body(0, 0, f"Body {i}", shapes[i], [])
            self._offset_shapes.append(shapes[i])
            body.background_color = self.GetBackgroundColour()

            for j in range(i, len(parts), num_outputs):
                part = Part(0, 0, j, parts[j], shapes[j], colors[j])
                part.background_color = self.GetBackgroundColour()
                body.parts.append(part)

            self._bodies.append(body)

        self.SetMinSize(self._get_min_size(200, 200))

        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_press)
        self.Bind(wx.EVT_LEFT_UP, self._on_release)
        self.Bind(wx.EVT_MOTION, self._on_move)
        self.Bind(wx.EVT_SIZE, self._on_size)

    def iter_parts(self) -> Iterable[Part]:
        for j in range(self._part_count // self._num_outputs):
            for i in range(self._num_outputs):
                yield self._bodies[i].parts[j]

    def _on_paint(self, evt: wx.PaintEvent):
        dc = wx.PaintDC(self)
        self._on_draw(dc)

    def _on_size(self, evt: wx.SizeEvent):
        w, h = self._get_min_size(*evt.GetSize())
        self.SetSizeHints(w, h)

    def _get_min_size(self, w, h):
        dc = wx.ClientDC(self)
        new_h = _PADDING

        for body in self._bodies:
            bw, bh = body.size(dc, *self._get_internal_size(w, h))
            new_h += bh + _PADDING

        return w, new_h

    def _get_internal_size(self, w: int, h: int) -> Tuple[int, int]:
        __, fh = self.GetFont().GetPixelSize()
        return (w - _PADDING * 4 - fh, h - _PADDING * 2)

    def _on_draw(self, dc: wx.DC):
        width, height = self.GetClientSize()
        y_off = _PADDING
        __, fh = self.GetFont().GetPixelSize()

        y_info = []

        for body in self._bodies:
            if(not body.dragging):
                body.x = _PADDING
                body.y = y_off

            bw, bh = body.size(dc, *self._get_internal_size(width, height))
            y_info.append((y_off, bh))
            y_off += bh + _PADDING

        for (by_off, bh), orig_shape in zip(y_info, self._offset_shapes):
            dc.SetPen(wx.Pen(dc.GetTextForeground(), _PADDING // 2, wx.PENSTYLE_SOLID))
            dc.DrawLine(
                width - fh - _PADDING * 2,
                by_off + _PADDING,
                width - fh - _PADDING * 2,
                by_off + bh - _PADDING * 2
            )
            dc.SetPen(wx.TRANSPARENT_PEN)
            dc.SetBrush(wx.Brush(dc.GetTextForeground(), wx.BRUSHSTYLE_SOLID))
            WxDotShapeDrawer(dc)[orig_shape](width - fh // 2 - _PADDING, by_off + bh // 2, fh // 2)

        for body in sorted(self._bodies, key=lambda a: a.dragging or any(b.dragging for b in a.parts)):
            body.draw(dc, *self._get_internal_size(width, height))

    def _on_press(self, evt: wx.MouseEvent):
        if(self._pressed_obj is None):
            dc = wx.ClientDC(self)
            w, h = self.GetClientSize()
            mx, my = evt.GetPosition()

            for body in self._bodies:
                res = body.get_mouseover(dc, *self._get_internal_size(w, h), mx, my)
                if(res is not None):
                    self._pressed_obj = res
                    self._owner_body = body
                    self._press_offset = (mx - res.x, my - res.y)
                    res.dragging = True

    def _handle_drag(self, dc: wx.DC, w: int, h: int, x: int, y: int, is_release: bool = False):
        if(self._highlight_obj is not None):
            self._highlight_obj.highlight = False
            self._highlight_obj = None

        if(isinstance(self._pressed_obj, Part)):
            for body in self._bodies:
                res = body.get_mouseover(dc, *self._get_internal_size(w, h), x, y)
                if(res is not None and res is not self._pressed_obj):
                    index = self._pressed_obj.index // self._num_outputs
                    if(is_release):
                        self._owner_body.parts[index] = body.parts[index]
                        body.parts[index] = self._pressed_obj
                    else:
                        body.parts[index].highlight = True
                        self._highlight_obj = body.parts[index]
                    return
        elif(isinstance(self._pressed_obj, Body)):
            body = None

            for i, body in enumerate(self._bodies):
                if(body is self._pressed_obj):
                    continue
                h = body.size(dc, *self._get_internal_size(w, h))[1]
                end_h = body.y + h
                if(end_h >= y):
                    break

            if(body is None or body.y > y):
                return

            if(is_release):
                j = self._bodies.index(self._pressed_obj)
                self._bodies[j] = self._bodies[i]
                self._bodies[i] = self._pressed_obj
            else:
                self._bodies[i].highlight = True
                self._highlight_obj = self._bodies[i]

    def _on_release(self, evt: wx.MouseEvent):
        if(self._pressed_obj is not None):
            dc = wx.ClientDC(self)
            w, h = self.GetClientSize()
            x, y = evt.GetPosition()
            self._handle_drag(dc, w, h, x, y, True)
            self._pressed_obj.dragging = False
            self._pressed_obj = None
            self._owner_body = None
            if(self._highlight_obj is not None):
                self._highlight_obj.highlight = False
                self._highlight_obj = None
            self.Refresh()

    def _on_move(self, evt: wx.MouseEvent):
        old_hover = self._hover_obj
        mx, my = evt.GetPosition()
        dc = wx.ClientDC(self)
        w, h = self.GetClientSize()

        if(self._pressed_obj is not None):
            if(self._hover_obj is not None):
                self._hover_obj.hover = False
                self._hover_obj = None
            self._pressed_obj.x = mx - self._press_offset[0]
            self._pressed_obj.y = my - self._press_offset[1]
            self._handle_drag(dc, w, h, mx, my)
        else:
            for body in self._bodies:
                res = body.get_mouseover(dc, *self._get_internal_size(w, h), mx, my)
                if(res is not None):
                    if(self._hover_obj is not None):
                        self._hover_obj.hover = False
                    self._hover_obj = res
                    res.hover = True
                    break
            else:
                if(self._hover_obj is not None):
                    self._hover_obj.hover = False
                self._hover_obj = None

        if(self._pressed_obj is not None or old_hover is not self._hover_obj):
            self.Refresh()


def _main():
    from diplomat.utils.colormaps import to_rgba

    app = wx.App()
    dlg = IdSwapDialog(
        None,
        wx.ID_ANY,
        num_outputs=3,
        labels=["Nose 1", "Nose 2", "Nose 3", "Back 1", "Back 2", "Back 3", "Tail 1", "Tail 2", "Tail 3"],
        colors=[tuple(int(v * 255) for v in to_rgba(c)) for c in ["red", "red", "red", "green", "green", "green", "blue", "blue", "blue"]],
        shapes=["circle", "triangle", "square", "circle", "triangle", "square", "circle", "triangle", "square"]
    )
    with dlg as dlg:
        if(dlg.ShowModal() == wx.ID_OK):
            print(dlg.get_proposed_order())
            print([dlg._labels[i] for i in dlg.get_proposed_order()])
        else:
            print("Canceled...")


if(__name__ == "__main__"):
    _main()
