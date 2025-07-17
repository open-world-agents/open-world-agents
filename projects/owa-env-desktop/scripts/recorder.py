"""
This script is to verify whether record/replaying keyboard/mouse events works well with pynput library.
"""

import pickle
import sys
import time

import typer
from loguru import logger
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key, KeyCode
from pynput.keyboard import Listener as KeyboardListener
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController
from pynput.mouse import Listener as MouseListener

from owa.env.desktop.utils import vk_to_name

# set logger level
logger.remove()
logger.add(sys.stderr, level="DEBUG")

app = typer.Typer()


@app.command()
def record(file: str = "events.pkl"):
    """
    Record keyboard and mouse events and save them to a file.
    Press ESC to stop recording.
    """
    events = []
    start_time = time.time()

    mouse_listener = None
    keyboard_listener = None

    def on_move(x, y):
        timestamp = time.time() - start_time
        events.append(("move", timestamp, x, y))

    def on_click(x, y, button, pressed):
        timestamp = time.time() - start_time
        events.append(("click", timestamp, x, y, button.name, pressed))

    def on_scroll(x, y, dx, dy):
        timestamp = time.time() - start_time
        events.append(("scroll", timestamp, x, y, dx, dy))

    def on_press(key):
        timestamp = time.time() - start_time
        _key = key.value if hasattr(key, "value") else key
        vk = getattr(_key, "vk", None)
        logger.debug(f"key: {key}, _key: {_key}, vk: {vk}, scancode: {_key._scan}, name: {vk_to_name(vk)}")
        if vk is not None:
            events.append(("key_press", timestamp, vk))

    def on_release(key):
        if key == Key.esc:
            # Stop both listeners
            mouse_listener.stop()
            return False
        timestamp = time.time() - start_time
        _key = key.value if hasattr(key, "value") else key
        vk = getattr(_key, "vk", None)
        logger.debug(f"key: {key}, _key: {_key}, vk: {vk}, scancode: {_key._scan}, name: {vk_to_name(vk)}")
        if vk is not None:
            events.append(("key_release", timestamp, vk))

    with (
        MouseListener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as ml,
        KeyboardListener(on_press=on_press, on_release=on_release) as kl,
    ):
        mouse_listener = ml
        keyboard_listener = kl  # noqa: F841
        typer.echo("Recording... Press ESC to stop.")
        kl.join()

    # Save events to file
    with open(file, "wb") as f:
        pickle.dump(events, f)
    typer.echo(f"Recording saved to {file}")


@app.command()
def replay(file: str = "events.pkl"):
    """
    Replay keyboard and mouse events from a file.
    """
    # Load events from file
    with open(file, "rb") as f:
        events = pickle.load(f)
    typer.echo(f"Replaying events from {file}")

    mouse_controller = MouseController()
    keyboard_controller = KeyboardController()

    start_time = None

    for event in events:
        event_type = event[0]
        timestamp = event[1]
        if start_time is None:
            start_time = time.time()
        else:
            # Sleep until the appropriate time
            time_to_wait = (start_time + timestamp) - time.time()
            if time_to_wait > 0:
                time.sleep(time_to_wait)

        if event_type == "move":
            x = event[2]
            y = event[3]
            mouse_controller.position = (x, y)
        elif event_type == "click":
            x = event[2]
            y = event[3]
            button = Button[event[4]]
            pressed = event[5]
            mouse_controller.position = (x, y)
            if pressed:
                mouse_controller.press(button)
            else:
                mouse_controller.release(button)
        elif event_type == "scroll":
            x = event[2]
            y = event[3]
            dx = event[4]
            dy = event[5]
            mouse_controller.position = (x, y)
            mouse_controller.scroll(dx, dy)
        elif event_type == "key_press":
            vk = event[2]
            keycode = KeyCode.from_vk(vk)
            logger.debug(f"key_press: keycode {keycode}, vk {vk}")
            keyboard_controller.press(keycode)
        elif event_type == "key_release":
            vk = event[2]
            keycode = KeyCode.from_vk(vk)
            logger.debug(f"key_release: keycode {keycode}, vk {vk}")
            keyboard_controller.release(keycode)


if __name__ == "__main__":
    app()
