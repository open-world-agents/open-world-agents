import time

from owa.core.listener import Listener
from owa.core.registry import LISTENERS

from .callables import get_active_window


@LISTENERS.register("window")
class WindowListener(Listener):
    """
    Calls callback every second with the active window information.
    """

    def loop(self, stop_event):
        while not stop_event.is_set():
            window = get_active_window()
            self.callback(window)
            time.sleep(1)
