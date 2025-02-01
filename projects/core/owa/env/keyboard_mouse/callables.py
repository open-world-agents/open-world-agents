from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Controller as MouseController

from owa.registry import CALLABLES

mouse_controller = MouseController()

CALLABLES.register("mouse.click")(mouse_controller.click)
CALLABLES.register("mouse.move")(mouse_controller.move)
CALLABLES.register("mouse.position")(lambda: mouse_controller.position)
CALLABLES.register("mouse.press")(mouse_controller.press)
CALLABLES.register("mouse.release")(mouse_controller.release)
CALLABLES.register("mouse.scroll")(mouse_controller.scroll)

keyboard_controller = KeyboardController()

CALLABLES.register("keyboard.press")(keyboard_controller.press)
CALLABLES.register("keyboard.release")(keyboard_controller.release)
CALLABLES.register("keyboard.type")(keyboard_controller.type)
