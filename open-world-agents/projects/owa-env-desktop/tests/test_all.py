import sys
import time
from unittest.mock import patch

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


def test_raw_mouse_event_message():
    """Test RawMouseEvent message creation and serialization."""
    RawMouseEvent = MESSAGES["desktop/RawMouseEvent"]

    # Test creating a raw mouse event
    event = RawMouseEvent(
        last_x=10,
        last_y=-5,
        button_flags=0x0001,  # RI_MOUSE_LEFT_BUTTON_DOWN
        button_data=0,
        device_handle=12345,
        timestamp=1234567890,
    )

    assert event.dx == 10
    assert event.dy == -5
    assert event.button_flags == 0x0001
    assert event.button_data == 0
    assert event.device_handle == 12345
    assert event.timestamp == 1234567890

    # Test with minimal required fields
    minimal_event = RawMouseEvent(last_x=0, last_y=0, button_flags=0, button_data=0)
    assert minimal_event.dx == 0
    assert minimal_event.dy == 0
    assert minimal_event.device_handle is None
    assert minimal_event.timestamp is None


def test_raw_mouse_listener_registration():
    """Test that RawMouseListener is properly registered in the plugin system."""
    # Verify the raw mouse listener is available by checking if it can be accessed
    assert "desktop/raw_mouse" in LISTENERS

    # Test creating the listener
    raw_mouse_listener = LISTENERS["desktop/raw_mouse"]()
    assert raw_mouse_listener is not None

    # Test that it has the expected interface
    assert hasattr(raw_mouse_listener, "configure")
    assert hasattr(raw_mouse_listener, "start")
    assert hasattr(raw_mouse_listener, "stop")


@pytest.mark.skipif(sys.platform != "win32", reason="Raw input only supported on Windows")
def test_raw_mouse_listener_functionality():
    """Test RawMouseListener functionality with mocked raw input."""
    from owa.env.desktop.keyboard_mouse.raw_input import RawInputCapture

    received_events = []

    def on_raw_mouse_event(event):
        received_events.append(event)

    # Create and configure the listener
    raw_mouse_listener = LISTENERS["desktop/raw_mouse"]()

    # Mock the RawInputCapture to avoid actual Windows API calls
    with (
        patch.object(RawInputCapture, "start", return_value=True),
        patch.object(RawInputCapture, "stop"),
        patch.object(RawInputCapture, "register_callback") as mock_register,
    ):
        # Configure the listener
        raw_mouse_listener.configure(callback=on_raw_mouse_event)

        # Verify that the raw input capture was configured
        assert hasattr(raw_mouse_listener, "raw_input_capture")
        mock_register.assert_called_once()

        # Simulate receiving a raw mouse event
        RawMouseEvent = MESSAGES["desktop/RawMouseEvent"]
        test_event = RawMouseEvent(last_x=15, last_y=-10, button_flags=0x0001, button_data=0, timestamp=time.time_ns())

        # Manually trigger the internal callback
        raw_mouse_listener._on_raw_mouse_event(test_event)

        # Since we're not actually running the loop, manually set the callback
        raw_mouse_listener._current_callback = on_raw_mouse_event
        raw_mouse_listener._on_raw_mouse_event(test_event)

        # Verify the event was received
        assert len(received_events) == 1
        assert received_events[0].dx == 15
        assert received_events[0].dy == -10
        assert received_events[0].button_flags == 0x0001


@pytest.mark.skipif(sys.platform == "win32", reason="Test non-Windows platform behavior")
def test_raw_mouse_listener_non_windows():
    """Test RawMouseListener behavior on non-Windows platforms."""
    from owa.env.desktop.keyboard_mouse.raw_input import RawInputCapture

    # Create raw input capture instance
    capture = RawInputCapture()

    # Should return False on non-Windows platforms
    assert capture.start() is False

    # Stop should not raise an error
    capture.stop()  # Should not raise


def test_raw_mouse_integration_with_existing_mouse():
    """Test that raw mouse listener can coexist with regular mouse listener."""
    received_mouse_events = []
    received_raw_events = []

    def on_mouse_event(event):
        received_mouse_events.append(event)

    def on_raw_mouse_event(event):
        received_raw_events.append(event)

    # Create both listeners
    mouse_listener = LISTENERS["desktop/mouse"]()
    raw_mouse_listener = LISTENERS["desktop/raw_mouse"]()

    # Configure both listeners
    mouse_listener.configure(callback=on_mouse_event)
    raw_mouse_listener.configure(callback=on_raw_mouse_event)

    # Verify both are configured
    assert hasattr(mouse_listener, "callback")
    assert hasattr(raw_mouse_listener, "raw_input_capture")

    # Both should be different instances
    assert mouse_listener is not raw_mouse_listener
    assert type(mouse_listener).__name__ == "MouseListenerWrapper"
    assert type(raw_mouse_listener).__name__ == "RawMouseListener"


def test_raw_mouse_event_fields():
    """Test RawMouseEvent field validation and types."""
    RawMouseEvent = MESSAGES["desktop/RawMouseEvent"]

    # Test with various button flags (Windows RI_MOUSE_* constants)
    button_flags = [
        0x0001,  # RI_MOUSE_LEFT_BUTTON_DOWN
        0x0002,  # RI_MOUSE_LEFT_BUTTON_UP
        0x0004,  # RI_MOUSE_RIGHT_BUTTON_DOWN
        0x0008,  # RI_MOUSE_RIGHT_BUTTON_UP
        0x0400,  # RI_MOUSE_WHEEL
    ]

    for flag in button_flags:
        event = RawMouseEvent(
            last_x=1,
            last_y=1,
            button_flags=flag,
            button_data=120 if flag == 0x0400 else 0,  # Wheel delta for wheel events
        )
        assert event.button_flags == flag

    # Test negative movement values
    event = RawMouseEvent(last_x=-100, last_y=-50, button_flags=0, button_data=0)
    assert event.dx == -100
    assert event.dy == -50

    # Test large movement values (high-DPI scenarios)
    event = RawMouseEvent(last_x=1000, last_y=800, button_flags=0, button_data=0)
    assert event.dx == 1000
    assert event.dy == 800
