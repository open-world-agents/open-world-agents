"""
Tests for McapMessage generic type functionality.

This module tests the new generic type support for McapMessage[T],
ensuring type safety, backward compatibility, and proper IDE support.
"""

import warnings

import pytest

from mcap_owa.highlevel import McapMessage, OWAMcapReader, OWAMcapWriter
from owa.core.message import OWAMessage
from owa.core.utils.tempfile import NamedTemporaryFile


# Mock message types for testing generic functionality
class MockKeyboardEvent(OWAMessage):
    """Mock keyboard event for testing generic types."""

    _type = "test/KeyboardEvent"
    event_type: str
    vk: int
    timestamp: int = 0


class MockMouseEvent(OWAMessage):
    """Mock mouse event for testing generic types."""

    _type = "test/MouseEvent"
    event_type: str
    button: str
    x: int
    y: int
    timestamp: int = 0


@pytest.fixture
def temp_mcap_file():
    """Create a temporary MCAP file for testing."""
    with NamedTemporaryFile(suffix=".mcap") as f:
        yield f.name


class TestGenericTypes:
    """Test cases for McapMessage generic type functionality."""

    def test_backward_compatibility(self, temp_mcap_file):
        """Test that existing code without generic types still works."""
        keyboard_event = MockKeyboardEvent(event_type="press", vk=65, timestamp=1000)

        # Write message
        with OWAMcapWriter(temp_mcap_file) as writer:
            writer.write_message(keyboard_event, topic="/keyboard", timestamp=1000)

        # Read message without generic type annotation (backward compatibility)
        # Suppress warnings for mock messages not in registry
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Domain-based message.*not found in registry.*", UserWarning)
            with OWAMcapReader(temp_mcap_file, decode_args={"return_dict_on_failure": True}) as reader:
                for msg in reader.iter_messages():
                    decoded = msg.decoded
                    assert decoded.event_type == "press"
                    assert decoded.vk == 65
                    break

    def test_generic_type_annotations(self, temp_mcap_file):
        """Test that generic type annotations work correctly."""
        keyboard_event = MockKeyboardEvent(event_type="press", vk=65, timestamp=1000)

        # Write message
        with OWAMcapWriter(temp_mcap_file) as writer:
            writer.write_message(keyboard_event, topic="/keyboard", timestamp=1000)

        # Read with type annotations
        # Suppress warnings for mock messages not in registry
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Domain-based message.*not found in registry.*", UserWarning)
            with OWAMcapReader(temp_mcap_file, decode_args={"return_dict_on_failure": True}) as reader:
                for msg in reader.iter_messages():
                    # Type-safe access
                    typed_msg: McapMessage[MockKeyboardEvent] = msg
                    decoded = typed_msg.decoded
                    assert decoded.event_type == "press"
                    assert decoded.vk == 65
                    break
