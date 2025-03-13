from pynput.keyboard import Listener as KeyboardListener
from pynput.mouse import Button
from pynput.mouse import Listener as MouseListener

from owa.core.listener import Listener
from owa.core.registry import LISTENERS

from ..msg import KeyboardEvent, MouseEvent
from ..utils import key_to_vk


@LISTENERS.register("keyboard")
class KeyboardListenerWrapper(Listener):
    def on_configure(self):
        self.listener = KeyboardListener(on_press=self.on_press, on_release=self.on_release)

    def on_press(self, key):
        vk = key_to_vk(key)
        self.callback(KeyboardEvent(event_type="press", vk=vk))

    def on_release(self, key):
        vk = key_to_vk(key)
        self.callback(KeyboardEvent(event_type="release", vk=vk))

    def loop(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()


@LISTENERS.register("mouse")
class MouseListenerWrapper(Listener):
    def on_configure(self):
        self.listener = MouseListener(on_move=self.on_move, on_click=self.on_click, on_scroll=self.on_scroll)

    def on_move(self, x, y):
        self.callback(MouseEvent(event_type="move", x=x, y=y))

    def on_click(self, x, y, button: Button, pressed):
        self.callback(MouseEvent(event_type="click", x=x, y=y, button=button.name, pressed=pressed))

    def on_scroll(self, x, y, dx, dy):
        self.callback(MouseEvent(event_type="scroll", x=x, y=y, dx=dx, dy=dy))

    def loop(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()
