"""
Provides a scrollable image list in wx widgets. Supports displaying multiple images in a row...
"""

from typing import List, Optional
import wx
from diplomat.utils._bit_or import _bit_or


class ScrollImageList(wx.ScrolledCanvas):
    """
    A custom wx widget which is capable of dynamically displaying a list of images with scroll bars. The images can
    be updated without breaking the widget, which will properly resize its scrollbars to accommodate the images...
    """

    SCROLL_RATE = 5

    def __init__(self, parent, img_list: Optional[List[wx.Bitmap]], orientation = wx.VERTICAL, padding = 20,
                 wid=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.DefaultSize, style=_bit_or(wx.HSCROLL, wx.VSCROLL),
                 name="ScrollImageList"):
        """
        Construct a new scrollable image list.

        :param parent: The parent widget.
        :param img_list: A list of wx.Bitmap, the bitmaps to be displayed in the widget
        :param orientation: The direction to layout images in, wx.VERTICAL or wx.HORIZONTAL. Defaults to wx.VERTICAL.
        :param padding: The padding between images. Defaults to 20 pixels.
        :param wid: wx ID of the window, and integer. Defaults to wx.ID_ANY.
        :param pos: WX Position of the control. Defaults to wx.DefaultPosition.
        :param size: WX Size of the control. Defaults to wx.DefaultSize.
        :param style: WX ScrolledCanvas Style. See wx.Control docs for possible options.
                      (Defaults to wx.HSCROLL | wx.VSCROLL).
        :param name: WX internal name of widget.
        """
        super().__init__(parent, wid, pos, size, style | wx.FULL_REPAINT_ON_RESIZE, name)
        if(img_list is None):
            img_list = []

        self._bitmaps = []
        self._mode = wx.VERTICAL
        self._padding = 20
        self._dims = None
        self.image_quality = wx.IMAGE_QUALITY_NEAREST

        self._scroll_extra = 0

        self.set_bitmaps(img_list)
        self.set_orientation(orientation)
        self.set_padding(padding)

        if(size == wx.DefaultSize):
            self.SetInitialSize(wx.Size(*self._dims))
        else:
            self.SetInitialSize(size)
        self.SetSize(wx.Size(*self._dims))
        self.EnableScrolling(True, True)
        self.ShowScrollbars(True, True)

        self.Bind(wx.EVT_SIZE, self._compute_dimensions)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel)

    # Not sure why I duplicate this...
    def SetScrollPageSize(self, orient, pageSize):
        super().SetScrollPageSize(orient, pageSize)

    def OnMouseWheel(self, event: wx.MouseEvent):
        if(event.GetWheelAxis() != wx.MOUSE_WHEEL_VERTICAL):
            return

        is_inv_func = getattr(event, "IsWheelInverted", lambda: False)
        self._scroll_extra += (-1 if(is_inv_func()) else 1) * event.GetWheelRotation()

        x, y = self.CalcUnscrolledPosition(0, 0)
        scale_x, scale_y = self.GetScrollPixelsPerUnit()

        if(scale_x != 0 and scale_y != 0):
            self.Scroll((x // scale_x), (y // scale_y) + (self._scroll_extra // scale_y))
            self._scroll_extra = (abs(self._scroll_extra) % scale_y) * (1 if(self._scroll_extra >= 0) else -1)
        else:
            self._scroll_extra = 0

        event.Skip(False)
        event.StopPropagation()

    def _compute_dimensions(self, event=None):
        """
        Compute the total size that will be taken up by the images. Returns, a tuple of two integers being the width
        and height of the canvas (internal size, so inside the scroll area).
        """
        cw, ch = self.GetClientSize()

        if(len(self._bitmaps) == 0):
            width, height = 100, 100
        elif(self._mode == wx.VERTICAL):
            width = cw
            height = (
                sum(int(bitmap.GetHeight() * (cw / bitmap.GetWidth())) for bitmap in self._bitmaps)
                + self._padding * len(self._bitmaps)
            )
        else:
            height = ch
            width = (
                sum(int(bitmap.GetWidth() * (ch / bitmap.GetHeight())) for bitmap in self._bitmaps)
                + self._padding * len(self._bitmaps)
            )

        self._dims = width, height
        self.SetVirtualSize(width, height)
        self.SetScrollRate(self.SCROLL_RATE, self.SCROLL_RATE)
        self.AdjustScrollbars()
        if(event is None):
            self.SendSizeEvent()
            self.Refresh()

        return width, height

    def OnDraw(self, dc: wx.DC):
        """
        Executed whenever the ScrolledImageList is requested to be redrawn. Redraws all of the images.

        param dc: The wx.DC to draw to.
        """
        width, height = self.GetClientSize()

        if((not width) or (not height)):
            return

        offset = 0

        if(self._mode == wx.VERTICAL):
            for bitmap in self._bitmaps:
                modified_height = bitmap.GetHeight()
                if(bitmap.GetWidth() != width):
                    modified_height = int(bitmap.GetHeight() * (width / bitmap.GetWidth()))

                pos_x, pos_y = self.CalcScrolledPosition(0, offset)
                if(pos_y + modified_height < 0):
                    pass
                elif(pos_y > height):
                    break
                else:
                    self._draw_bitmap(dc, bitmap, 0, offset, width, modified_height)

                offset += modified_height + self._padding
        else:
            for bitmap in self._bitmaps:
                modified_width = bitmap.GetWidth()
                if(bitmap.GetHeight() != height):
                    modified_width = int(bitmap.GetWidth() * (height / bitmap.GetHeight()))

                pos_x, pos_y = self.CalcScrolledPosition(offset, 0)
                if(pos_x + modified_width < 0):
                    pass
                elif(pos_x > width):
                    break
                else:
                    self._draw_bitmap(dc, bitmap, offset, 0, modified_width, height)

                offset += modified_width + self._padding

    def _draw_bitmap(self, dc: wx.DC, bitmap: wx.Bitmap, x: int, y: int, width: int, height: int):
        if(bitmap.GetWidth() != width or bitmap.GetHeight() != height):
            bitmap = wx.Bitmap(bitmap.ConvertToImage().Scale(width, height, self.image_quality))
        dc.DrawBitmap(bitmap, x, y)

    def get_padding(self) -> int:
        """
        Get the padding between images for this scroll image list.

        :returns: An integer, being the padding value used between images.
        """
        return self._padding

    def set_padding(self, value: int):
        """
        Set the padding between images for this scroll image list.

        :param value: An integer, being the padding value to use between images.
        """
        self._padding = int(value)
        self._dims = None
        self._compute_dimensions()

    def get_orientation(self) -> int:
        """
        Get the orientation of the images.

        :returns: wx.VERTICAL or wx.HORIZONTAL.
        """
        return self._mode

    def set_orientation(self, value: int):
        """
        Set the orientation of the images.

        :param value: wx.VERTICAL or wx.HORIZONTAL.
        """
        if((value != wx.VERTICAL) and (value != wx.HORIZONTAL)):
            raise ValueError("Orientation must be wx.VERTICAL or wx.HORIZONTAL!!!")
        self._mode = value
        self._dims = None
        self._compute_dimensions()

    def get_bitmaps(self) -> List[wx.Bitmap]:
        """
        Get the list of bitmaps being shown.

        :returns: A list of wx.Bitmap.
        """
        return self._bitmaps

    def set_bitmaps(self, bitmaps: List[wx.Bitmap]):
        """
        Set the list of bitmaps being shown.

        :param bitmaps: A list of wx.Bitmap.
        """
        if(bitmaps is None):
            bitmaps = []
        self._bitmaps = bitmaps
        self._dims = None
        self._compute_dimensions()


def scroll_image_demo():
    app = wx.App()

    im_list = [wx.Bitmap.FromRGBA(100, 100, 0, 0, 0, 255) for i in range(40)]
    frame = wx.Frame(None, title="Scrolling Image List")
    f_layout = wx.BoxSizer(wx.VERTICAL)

    im_widget = ScrollImageList(frame, im_list, wx.VERTICAL, size=wx.Size(200, 300))
    f_layout.Add(im_widget, 1, wx.EXPAND)

    frame.SetSizerAndFit(f_layout)
    frame.Show()

    app.MainLoop()


if(__name__ == "__main__"):
    scroll_image_demo()
