from owa import Listener
from owa.registry import LISTENERS
from pynput.keyboard import Listener as KeyboardListener
from pynput.mouse import Listener as MouseListener


@LISTENERS.register("keyboard")
class KeyboardListenerWrapper(Listener):
    def on_configure(self):
        self.listener = KeyboardListener(on_press=self.on_press, on_release=self.on_release)
        return True

    def on_press(self, key):
        self.callback("keyboard.press", key)

    def on_release(self, key):
        self.callback("keyboard.release", key)

    def on_activate(self):
        self.listener.start()
        return True

    def on_deactivate(self):
        self.listener.stop()
        return True


@LISTENERS.register("mouse")
class MouseListenerWrapper(Listener):
    def on_configure(self, callback):
        self.callback = callback
        self.listener = MouseListener(on_move=self.on_move, on_click=self.on_click, on_scroll=self.on_scroll)
        return True

    def on_move(self, x, y):
        self.callback("mouse.move", x, y)

    def on_click(self, x, y, button, pressed):
        self.callback("mouse.click", x, y, button, pressed)

    def on_scroll(self, x, y, dx, dy):
        self.callback("mouse.scroll", x, y, dx, dy)

    def on_activate(self):
        self.listener.start()
        return True

    def on_deactivate(self):
        self.listener.stop()
        return True
