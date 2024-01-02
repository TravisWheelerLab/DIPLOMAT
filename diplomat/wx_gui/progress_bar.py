"""
An implementation of DIPLOMAT's progress bar API using wx widgets. Allows for displaying the progress of diplomat
processes.
"""

from typing import Iterable
import wx
import time
from datetime import timedelta


class TqdmWxPanel(wx.Panel):
    """
    A WX progress bar which mimics the tqdm interface. Currently supports the update, __iter__, and reset methods...
    """
    # This is the number of nanoseconds that must go by before we allow another update. This is current set to
    # only allow 10 updates a second.
    UPDATE_RATE = 1e8

    def __init__(self, parent, wid=wx.ID_ANY):
        """
        Construct the new progress bar...

        :param parent: The parent wx widget...
        :param wid: The id of this new wx widget...
        """
        super().__init__(parent, wid, pos=wx.DefaultPosition, size=wx.DefaultSize, style=wx.TAB_TRAVERSAL,
                          name="TqdmWxPanel")

        self.__main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.__progress_bar = wx.Gauge(self)
        self.__text = wx.StaticText(self, label="\n?it/s | Time Spent: ?, Time Left: ?",
                                    style=wx.ALIGN_CENTER_HORIZONTAL)

        self.__main_sizer.Add(self.__progress_bar, 0, wx.EXPAND)
        self.__main_sizer.Add(self.__text, 0, wx.ALIGN_CENTER)

        self.SetSizerAndFit(self.__main_sizer)

        # Initialize all progress variables... Also set the start time to now!
        self._total = None
        self._gap_sum = 0
        self._n = 0
        try:
            self._old_time = time.time_ns()
        except AttributeError:
            self._old_time = time.time() * 1e9
        self._start_time = self._old_time
        self._step = 0
        self._speed = 0

        self._pre_txt = ""

        self._closed = False

    def __call__(self, iterable: Iterable = None, total: int = None):
        """
        Set the iterable or total of this progress bar, and return this current progress bar with the changed values.
        (Does not construct a new object). This provides compatibility with the tqdm progress bar api.
        """
        self._iter = iterable
        if(self._iter is not None):
            try:
                self._total = len(self._iter)
            except AttributeError:
                self._total = None
            self.reset(total)
        elif(total is not None):
            self._total = total
            self.reset(total)

        return self

    def __iter__(self):
        """
        Iterate the iterable stored in this progress bar, also displaying its progress...
        """
        total = None
        iterator = self._iter
        if(self._iter is not None):
            if(hasattr(iterator, "__len__")):
                total = len(iterator)
            self.reset(total)
        else:
            raise ValueError("No Iterator!")

        for item in iterator:
            self.update()
            yield item
        return

    def message(self, msg: str):
        """
        Set the message of the progress bar...
        """
        self._pre_txt = msg
        self._display()

    def update(self, amount=1):
        """
        Update the progress bar by amount, and display the change...
        """
        total = self._total

        if(total is not None):
            self._n = min(total - 1, self._n + amount)
        try:
            new_time = time.time_ns()
        except AttributeError:
            new_time = time.time() * 1e9
        self._step = (new_time - self._old_time)
        self._speed = self._step / amount
        self._old_time = new_time
        self._display()

    def reset(self, total = None):
        """
        Reset the progress bar to 0, and set a new total value.
        """
        self._n = 0
        try:
            self._old_time = time.time_ns()
        except AttributeError:
            self._old_time = time.time() * 1e9
        self._start_time = self._old_time
        self._speed = 0
        self._step = 0
        self._gap_sum = 0
        self._total = total
        self._display()

    def GetTextExtent(self, string):
        return self.__text.GetTextExtent(string)

    def _display(self):
        """
        Displays the wx progress bar. Internal, should not be called directly!
        """
        if(self._closed):
            raise ValueError("The progress bar has been closed!")

        total = self._total
        n = self._n

        # We update the gap sum and check if it has been long enough since the last update(or we reached the end), if
        # not just immediately return. This makes performance much faster!!!
        self._gap_sum += self._step
        if(self._gap_sum < self.UPDATE_RATE):
            if((total is not None) and (n != total - 1)):
                return
        # Reset the gap sum, for next redraw...
        self._gap_sum = 0

        if(total is None):
            self.__progress_bar.Pulse()
        else:
            self.__progress_bar.SetRange(total - 1)
            self.__progress_bar.SetValue(n)

        time_spent = timedelta(seconds=int((self._old_time - self._start_time) / 1e9))

        if(self._speed != 0):
            it_sec = f"{1 / (self._speed * 1e-9):.02f}"
            est_time = None if(total is None) else int(((total - n) * ((self._old_time - self._start_time) / (n if(n != 0) else 1))) / 1e9)
            est_time = "?" if (est_time is None) else str(timedelta(seconds=est_time))
        else:
            it_sec = "?"
            est_time = "?"

        self.__text.SetLabelText(f"{self._pre_txt}\n{it_sec}it/s | Time Spent: {time_spent}, Time Left: {est_time}")
        # This sends a resize event, which corrects the StaticText widget and centers it properly...
        self.SendSizeEvent()
        # Vital, this gives control back to wxWidgets, and allows it to update the UI and process any events...
        wx.GetApp().Yield(True)

    def close(self):
        self._closed = True
        self.Enable(False)


if(__name__ == "__main__"):
    # Tests the progress bar by running it on some fake work via clicking a button...
    def run(tqdm: TqdmWxPanel):
        tqdm.message("Running main loop...")
        for i in tqdm(range(int(1e6))):
            pass

    class TestFrm(wx.Frame):
        def __init__(self, parent=None, wid=wx.ID_ANY, title=""):
            super().__init__(parent, wid, title)

            self._sizer = wx.BoxSizer(wx.VERTICAL)
            self._tqdm = TqdmWxPanel(self, wx.ID_ANY)
            self._button = wx.Button(self, label="Run Progress Bar")
            self._sizer.Add(self._tqdm, 1, wx.EXPAND)
            self._sizer.Add(self._button, 0, wx.ALIGN_CENTER)

            self.SetSizerAndFit(self._sizer)
            size: wx.Size = self.GetMinSize()

            self._sizer.SetMinSize(wx.Size(self._tqdm.GetTextExtent("0" * 80).GetWidth(), size.GetHeight()))
            self.SetSize(wx.Size(self._tqdm.GetTextExtent("0" * 80).GetWidth(), size.GetHeight()))

            self.SendSizeEvent()
            self.Bind(wx.EVT_BUTTON, self.on_btn)
            self.Bind(wx.EVT_CLOSE, lambda evt: None)

        def on_btn(self, evt):
            self._button.Enable(False)
            run(self._tqdm)
            self.Destroy()

    app = wx.App()
    frm = TestFrm(None, title="Test...")
    frm.Show()
    app.MainLoop()
