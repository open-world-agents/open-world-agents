from owa.core.registry import CALLABLES, LISTENERS, activate_module
from owa.env.desktop.constants import VK
from owa.env.desktop.msg import KeyboardEvent
import _thread
import threading
import time

DELAY = 0.4  # seconds


def press_key(key):
    time.sleep(DELAY)  # seconds
    print(f"{key=} pressed")
    CALLABLES["keyboard.press"](key)


def release_key(key):
    time.sleep(DELAY)  # seconds
    print(f"{key=} released")
    CALLABLES["keyboard.release"](key)


def on_keyboard_event(keyboard_event: KeyboardEvent):
    if keyboard_event.vk == VK.F10:
        print("Stopping with F10 key")
        _thread.interrupt_main()
    elif keyboard_event.vk == VK.F7 and keyboard_event.event_type == "press":
        threading.Thread(target=press_key, args=(VK.LEFT,), daemon=True).start()
    elif keyboard_event.vk == VK.F7 and keyboard_event.event_type == "release":
        threading.Thread(target=release_key, args=(VK.LEFT,), daemon=True).start()
    elif keyboard_event.vk == VK.F8 and keyboard_event.event_type == "press":
        threading.Thread(target=press_key, args=(VK.RIGHT,), daemon=True).start()
    elif keyboard_event.vk == VK.F8 and keyboard_event.event_type == "release":
        threading.Thread(target=release_key, args=(VK.RIGHT,), daemon=True).start()


activate_module("owa.env.desktop")

keyboard_listener = LISTENERS["keyboard"]().configure(callback=on_keyboard_event)
keyboard_listener.start()


time.sleep(10000)
