from pynput.keyboard import Listener as KeyboardListener
from pynput.mouse import Listener as MouseListener

from owa.env import Listener
from owa.registry import LISTENERS


@LISTENERS.register("keyboard")
class KeyboardListenerWrapper(Listener):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.listener = KeyboardListener(on_press=self.on_press, on_release=self.on_release)

    def on_press(self, key):
        self.callback("keyboard.press", key)

    def on_release(self, key):
        self.callback("keyboard.release", key)


@LISTENERS.register("mouse")
class MouseListenerWrapper(Listener):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.listener = MouseListener(on_move=self.on_move, on_click=self.on_click, on_scroll=self.on_scroll)

    def on_move(self, x, y):
        self.callback("mouse.move", x, y)

    def on_click(self, x, y, button, pressed):
        self.callback("mouse.click", x, y, button, pressed)

    def on_scroll(self, x, y, dx, dy):
        self.callback("mouse.scroll", x, y, dx, dy)
