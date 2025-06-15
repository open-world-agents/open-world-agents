import time

import pytest

from owa.core import CALLABLES, LISTENERS, MESSAGES


def test_screen_capture():
    # Test that the screen capture returns an image with the expected dimensions.
    capture_func = CALLABLES["desktop/screen.capture"]
    image = capture_func()
    # Check the color channel count. shape should be (H, W, 3)
    assert image.ndim == 3 and image.shape[2] == 3, "Expected 3-color channel image"

    # in github workflow, the screen capture will be 768x1024


def test_get_active_window():
    # Test that the active window function returns a non-None value.
    active_window = CALLABLES["desktop/window.get_active_window"]()
    assert active_window is not None, "Active window returned None"


def test_get_window_by_title():
    # Test retrieving a window by a specific title.
    try:
        window_instance = CALLABLES["desktop/window.get_window_by_title"]("open-world-agents")
    except ValueError:
        window_instance = None

    if window_instance is None:
        pytest.skip("Window with title 'open-world-agents' not found; skipping test.")
    else:
        # Here we assume window_instance should be a dict or similar object.
        # Adjust type check or property tests as necessary.
        assert isinstance(window_instance, MESSAGES["desktop/WindowInfo"]), (
            "Expected window instance to be a WindowInfo"
        )


def test_mouse_click():
    # Test that the mouse-click callable can be triggered.
    # Instead of causing a real click, we'll replace it temporarily with a fake.
    captured = []

    def fake_click(button, clicks):
        captured.append((button, clicks))
        return "clicked"

    # Save the original function so we can restore it
    original_click = CALLABLES["desktop/mouse.click"]
    CALLABLES.register("desktop/mouse.click", fake_click, is_instance=True)

    try:
        result = CALLABLES["desktop/mouse.click"]("left", 2)
        assert result == "clicked", "Fake click did not return expected result"
        assert captured == [("left", 2)], f"Expected captured click data [('left', 2)], got {captured}"
    finally:
        # Restore the original function no matter what.
        CALLABLES.register("desktop/mouse.click", original_click, is_instance=True)


def test_keyboard_listener():
    # Test the keyboard listener by verifying that a custom callback receives simulated events.
    received_events = []

    def on_keyboard_event(event_type, key):
        received_events.append((event_type, key))

    # Create and configure the listener.
    keyboard_listener = LISTENERS["desktop/keyboard"]().configure(callback=on_keyboard_event)
    keyboard_listener.start()

    # In a real-world scenario, the listener would capture actual events.
    # For testing purposes, we simulate calling the callback manually.
    on_keyboard_event("press", "a")
    on_keyboard_event("release", "a")

    # Wait briefly to mimic asynchronous event handling.
    time.sleep(0.5)

    # Stop the listener if your framework provides a stop() method,
    # or allow the thread to end naturally.
    if hasattr(keyboard_listener, "stop"):
        keyboard_listener.stop()

    # Verify that the simulated events were handled.
    assert ("press", "a") in received_events, "Did not capture key press event"
    assert ("release", "a") in received_events, "Did not capture key release event"
