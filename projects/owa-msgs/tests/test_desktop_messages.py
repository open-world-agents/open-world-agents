"""
Tests for desktop message definitions.
"""

import io

import pytest
from pydantic import ValidationError

from owa.msgs.desktop.keyboard import KeyboardEvent, KeyboardState
from owa.msgs.desktop.mouse import MouseEvent, MouseState
from owa.msgs.desktop.window import WindowInfo


class TestKeyboardMessages:
    """Test cases for keyboard message types."""

    def test_keyboard_event_creation(self):
        """Test creating KeyboardEvent instances."""
        # Valid press event
        event = KeyboardEvent(event_type="press", vk=65)
        assert event.event_type == "press"
        assert event.vk == 65
        assert event.timestamp is None
        assert event._type == "desktop/KeyboardEvent"

        # Valid release event with timestamp
        event = KeyboardEvent(event_type="release", vk=27, timestamp=1234567890)
        assert event.event_type == "release"
        assert event.vk == 27
        assert event.timestamp == 1234567890

    def test_keyboard_event_validation(self):
        """Test KeyboardEvent validation."""
        # Invalid event_type
        with pytest.raises(ValidationError):
            KeyboardEvent(event_type="invalid", vk=65)

        # Missing required fields
        with pytest.raises(ValidationError):
            KeyboardEvent(event_type="press")  # Missing vk

    def test_keyboard_event_serialization(self):
        """Test KeyboardEvent serialization."""
        event = KeyboardEvent(event_type="press", vk=65, timestamp=1234567890)

        buffer = io.BytesIO()
        event.serialize(buffer)

        serialized = buffer.getvalue().decode("utf-8")
        assert '"event_type":"press"' in serialized
        assert '"vk":65' in serialized
        assert '"timestamp":1234567890' in serialized

    def test_keyboard_event_deserialization(self):
        """Test KeyboardEvent deserialization."""
        json_data = '{"event_type":"press","vk":65,"timestamp":1234567890}'
        buffer = io.BytesIO(json_data.encode("utf-8"))

        event = KeyboardEvent.deserialize(buffer)
        assert event.event_type == "press"
        assert event.vk == 65
        assert event.timestamp == 1234567890

    def test_keyboard_state_creation(self):
        """Test creating KeyboardState instances."""
        # Empty state
        state = KeyboardState(buttons=set())
        assert state.buttons == set()
        assert state.timestamp is None
        assert state._type == "desktop/KeyboardState"

        # State with pressed keys
        state = KeyboardState(buttons={65, 66, 67}, timestamp=1234567890)
        assert state.buttons == {65, 66, 67}
        assert state.timestamp == 1234567890

    def test_keyboard_state_validation(self):
        """Test KeyboardState validation."""
        # Valid range (0-255)
        state = KeyboardState(buttons={0, 255})
        assert 0 in state.buttons
        assert 255 in state.buttons

        # Invalid range - should be caught by UInt8 annotation
        with pytest.raises(ValidationError):
            KeyboardState(buttons={-1})

        with pytest.raises(ValidationError):
            KeyboardState(buttons={256})


class TestMouseMessages:
    """Test cases for mouse message types."""

    def test_mouse_event_creation(self):
        """Test creating MouseEvent instances."""
        # Move event
        event = MouseEvent(event_type="move", x=100, y=200)
        assert event.event_type == "move"
        assert event.x == 100
        assert event.y == 200
        assert event.button is None
        assert event._type == "desktop/MouseEvent"

        # Click event
        event = MouseEvent(event_type="click", x=100, y=200, button="left", pressed=True, timestamp=1234567890)
        assert event.event_type == "click"
        assert event.button == "left"
        assert event.pressed is True
        assert event.timestamp == 1234567890

        # Scroll event
        event = MouseEvent(event_type="scroll", x=100, y=200, dx=5, dy=-3)
        assert event.event_type == "scroll"
        assert event.dx == 5
        assert event.dy == -3

    def test_mouse_event_validation(self):
        """Test MouseEvent validation."""
        # Invalid event_type
        with pytest.raises(ValidationError):
            MouseEvent(event_type="invalid", x=100, y=200)

        # Invalid button
        with pytest.raises(ValidationError):
            MouseEvent(event_type="click", x=100, y=200, button="invalid")

        # Missing required fields
        with pytest.raises(ValidationError):
            MouseEvent(event_type="move")  # Missing x, y

    def test_mouse_event_serialization(self):
        """Test MouseEvent serialization."""
        event = MouseEvent(event_type="click", x=100, y=200, button="left", pressed=True)

        buffer = io.BytesIO()
        event.serialize(buffer)

        serialized = buffer.getvalue().decode("utf-8")
        assert '"event_type":"click"' in serialized
        assert '"x":100' in serialized
        assert '"y":200' in serialized
        assert '"button":"left"' in serialized
        assert '"pressed":true' in serialized

    def test_mouse_state_creation(self):
        """Test creating MouseState instances."""
        # Empty state
        state = MouseState(x=100, y=200, buttons=set())
        assert state.x == 100
        assert state.y == 200
        assert state.buttons == set()
        assert state._type == "desktop/MouseState"

        # State with pressed buttons
        state = MouseState(x=100, y=200, buttons={"left", "right"}, timestamp=1234567890)
        assert state.buttons == {"left", "right"}
        assert state.timestamp == 1234567890

    def test_mouse_state_validation(self):
        """Test MouseState validation."""
        # Valid buttons
        state = MouseState(x=100, y=200, buttons={"left", "middle", "right"})
        assert "left" in state.buttons

        # Invalid button in set
        with pytest.raises(ValidationError):
            MouseState(x=100, y=200, buttons={"invalid"})


class TestWindowMessages:
    """Test cases for window message types."""

    def test_window_info_creation(self):
        """Test creating WindowInfo instances."""
        window = WindowInfo(title="Test Window", rect=(100, 200, 500, 600), hWnd=12345)

        assert window.title == "Test Window"
        assert window.rect == (100, 200, 500, 600)
        assert window.hWnd == 12345
        assert window._type == "desktop/WindowInfo"

    def test_window_info_properties(self):
        """Test WindowInfo computed properties."""
        window = WindowInfo(title="Test", rect=(100, 200, 500, 600), hWnd=1)

        assert window.width == 400  # 500 - 100
        assert window.height == 400  # 600 - 200

    def test_window_info_validation(self):
        """Test WindowInfo validation."""
        # Missing required fields
        with pytest.raises(ValidationError):
            WindowInfo(title="Test", rect=(100, 200, 500, 600))  # Missing hWnd

        with pytest.raises(ValidationError):
            WindowInfo(hWnd=1, rect=(100, 200, 500, 600))  # Missing title

    def test_window_info_serialization(self):
        """Test WindowInfo serialization."""
        window = WindowInfo(title="Test Window", rect=(100, 200, 500, 600), hWnd=12345)

        buffer = io.BytesIO()
        window.serialize(buffer)

        serialized = buffer.getvalue().decode("utf-8")
        assert '"title":"Test Window"' in serialized
        assert '"hWnd":12345' in serialized


class TestMessageSchemas:
    """Test message schema generation."""

    def test_keyboard_event_schema(self):
        """Test KeyboardEvent schema generation."""
        schema = KeyboardEvent.get_schema()
        assert schema is not None
        assert "properties" in schema
        assert "event_type" in schema["properties"]
        assert "vk" in schema["properties"]

    def test_mouse_event_schema(self):
        """Test MouseEvent schema generation."""
        schema = MouseEvent.get_schema()
        assert schema is not None
        assert "properties" in schema
        assert "event_type" in schema["properties"]
        assert "x" in schema["properties"]
        assert "y" in schema["properties"]
