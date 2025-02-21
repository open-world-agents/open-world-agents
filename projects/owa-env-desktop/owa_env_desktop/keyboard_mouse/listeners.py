from pynput.keyboard import Listener as KeyboardListener
from pynput.mouse import Button
from pynput.mouse import Listener as MouseListener

from owa.listener import Listener
from owa.registry import LISTENERS

from ..utils import key_to_vk


@LISTENERS.register("keyboard")
class KeyboardListenerWrapper(Listener):
    def configure(self):
        self.listener = KeyboardListener(on_press=self.on_press, on_release=self.on_release)

    def on_press(self, key):
        vk = key_to_vk(key)
        self.callback("keyboard.press", vk)

    def on_release(self, key):
        vk = key_to_vk(key)
        self.callback("keyboard.release", vk)

    def loop(self):
        self.listener.start()

    def cleanup(self): ...  # nothing to clean up

    def stop(self):
        self.listener.stop()


@LISTENERS.register("mouse")
class MouseListenerWrapper(Listener):
    def configure(self):
        self.listener = MouseListener(on_move=self.on_move, on_click=self.on_click, on_scroll=self.on_scroll)

    def on_move(self, x, y):
        self.callback("mouse.move", x, y)

    def on_click(self, x, y, button: Button, pressed):
        self.callback("mouse.click", x, y, button.name, pressed)

    def on_scroll(self, x, y, dx, dy):
        self.callback("mouse.scroll", x, y, dx, dy)

    def loop(self):
        self.listener.start()

    def cleanup(self): ...  # nothing to clean up

    def stop(self):
        self.listener.stop()
