import time

from owa.core.listener import Listener

from .callables import get_active_window


class WindowListener(Listener):
    """
    Periodically monitors and reports the currently active window.

    This listener calls the callback function every second with information
    about the currently active window, including title, position, and handle.

    Examples:
        Monitor active window changes:

        >>> def on_window_change(window):
        ...     if window:
        ...         print(f"Active window: {window.title}")
        >>>
        >>> listener = WindowListener().configure(callback=on_window_change)
        >>> listener.start()
        >>> # ... listener runs in background ...
        >>> listener.stop()
        >>> listener.join()

        Track window focus for automation:

        >>> def track_focus(window):
        ...     if window and "notepad" in window.title.lower():
        ...         print("Notepad is now active!")
        >>>
        >>> listener = WindowListener().configure(callback=track_focus)
        >>> listener.start()
    """

    def loop(self, stop_event):
        while not stop_event.is_set():
            window = get_active_window()
            self.callback(window)
            time.sleep(1)
