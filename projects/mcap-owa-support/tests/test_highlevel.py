"""
Tests for mcap-owa-support package functionality.

This module tests only the MCAP format support features:
- OWAMcapReader and OWAMcapWriter
- Message serialization/deserialization
- MCAP file format handling

These tests focus on the mcap-owa-support package's own functionality
and use mock message types instead of importing from other packages.
"""

import tempfile
import warnings

import pytest
from owa.core.message import OWAMessage

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter


# Mock message types for testing (instead of importing from other packages)
class MockKeyboardEvent(OWAMessage):
    """Mock keyboard event for testing MCAP functionality."""

    _type = "test.msg.KeyboardEvent"
    event_type: str
    vk: int
    timestamp: int = 0


class MockMouseEvent(OWAMessage):
    """Mock mouse event for testing MCAP functionality."""

    _type = "test.msg.MouseEvent"
    event_type: str
    button: str
    x: int
    y: int
    timestamp: int = 0


@pytest.fixture
def temp_mcap_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = tmpdir + "/output.mcap"
        yield file_path


def test_write_and_read_messages(temp_mcap_file):
    """Test writing and reading multiple messages."""
    file_path = temp_mcap_file
    topic = "/chatter"

    # Suppress warnings only for mock message creation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        event = MockKeyboardEvent(event_type="press", vk=1)

    with OWAMcapWriter(file_path) as writer:
        for i in range(0, 10):
            publish_time = i
            writer.write_message(topic, event, log_time=publish_time)

    # Suppress warnings only for reading mock messages and version compatibility
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Failed to import module for schema 'test.msg.*", UserWarning)
        warnings.filterwarnings("ignore", "Reader version.*may not be compatible with writer version.*", UserWarning)

        with OWAMcapReader(file_path) as reader:
            messages = list(reader.iter_messages())
            assert len(messages) == 10
            for i, msg in enumerate(messages):
                assert msg.topic == topic
                assert msg.decoded.event_type == "press"
                assert msg.decoded.vk == 1
                assert msg.timestamp == i


def test_mcap_message_object(temp_mcap_file):
    """Test the new McapMessage object interface."""
    file_path = temp_mcap_file
    topic = "/keyboard"

    # Suppress warnings only for mock message creation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        event = MockKeyboardEvent(event_type="press", vk=65)

    with OWAMcapWriter(file_path) as writer:
        writer.write_message(topic, event, log_time=1000)

    # Suppress warnings only for reading mock messages and version compatibility
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Failed to import module for schema 'test.msg.*", UserWarning)
        warnings.filterwarnings("ignore", "Reader version.*may not be compatible with writer version.*", UserWarning)

        with OWAMcapReader(file_path) as reader:
            messages = list(reader.iter_messages())
            assert len(messages) == 1

            msg = messages[0]
            # Test all properties
            assert msg.topic == topic
            assert msg.timestamp == 1000
            assert isinstance(msg.message, bytes)
            assert msg.message_type == "test.msg.KeyboardEvent"

            # Test lazy decoded property
            decoded = msg.decoded
            assert decoded.event_type == "press"
            assert decoded.vk == 65

            # Test that decoded is cached (same object)
            assert msg.decoded is decoded


def test_schema_based_filtering(temp_mcap_file):
    """Test filtering messages by schema name."""
    file_path = temp_mcap_file

    # Suppress warnings only for mock message creation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        keyboard_event = MockKeyboardEvent(event_type="press", vk=65)
        keyboard_event2 = MockKeyboardEvent(event_type="release", vk=65)
        mouse_event = MockMouseEvent(event_type="click", button="left", x=100, y=200)

    with OWAMcapWriter(file_path) as writer:
        # Write different message types
        writer.write_message("/keyboard", keyboard_event, log_time=1000)
        writer.write_message("/keyboard", keyboard_event2, log_time=2000)
        writer.write_message("/mouse", mouse_event, log_time=3000)

    # Suppress warnings only for reading mock messages and version compatibility
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Failed to import module for schema 'test.msg.*", UserWarning)
        warnings.filterwarnings("ignore", "Reader version.*may not be compatible with writer version.*", UserWarning)

        with OWAMcapReader(file_path) as reader:
            # Filter by schema name
            keyboard_messages = [msg for msg in reader.iter_messages() if msg.message_type == "test.msg.KeyboardEvent"]

            mouse_messages = [msg for msg in reader.iter_messages() if msg.message_type == "test.msg.MouseEvent"]

            assert len(keyboard_messages) == 2
            assert keyboard_messages[0].decoded.event_type == "press"
            assert keyboard_messages[1].decoded.event_type == "release"

            assert len(mouse_messages) == 1
            assert mouse_messages[0].decoded.event_type == "click"
            assert mouse_messages[0].decoded.button == "left"


def test_multiple_message_types(temp_mcap_file):
    """Test writing and reading multiple different message types."""
    file_path = temp_mcap_file

    # Suppress warnings only for mock message creation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        keyboard_event = MockKeyboardEvent(event_type="press", vk=65, timestamp=1000)
        mouse_event = MockMouseEvent(event_type="move", button="none", x=150, y=250, timestamp=2000)

    with OWAMcapWriter(file_path) as writer:
        # Write keyboard events
        writer.write_message("/keyboard", keyboard_event, log_time=1000)
        # Write mouse events
        writer.write_message("/mouse", mouse_event, log_time=2000)

    # Suppress warnings only for reading mock messages and version compatibility
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Failed to import module for schema 'test.msg.*", UserWarning)
        warnings.filterwarnings("ignore", "Reader version.*may not be compatible with writer version.*", UserWarning)

        with OWAMcapReader(file_path) as reader:
            messages = list(reader.iter_messages())
            assert len(messages) == 2

            # Check first message (keyboard)
            keyboard_msg = messages[0]
            assert keyboard_msg.topic == "/keyboard"
            assert keyboard_msg.message_type == "test.msg.KeyboardEvent"
            assert keyboard_msg.decoded.event_type == "press"
            assert keyboard_msg.decoded.vk == 65

            # Check second message (mouse)
            mouse_msg = messages[1]
            assert mouse_msg.topic == "/mouse"
            assert mouse_msg.message_type == "test.msg.MouseEvent"
            assert mouse_msg.decoded.event_type == "move"
            assert mouse_msg.decoded.x == 150
            assert mouse_msg.decoded.y == 250
