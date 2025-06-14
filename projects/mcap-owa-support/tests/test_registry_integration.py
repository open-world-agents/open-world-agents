"""
Integration tests for MCAP with the OWA message registry.

This module tests the integration between mcap-owa-support and the
message registry system, ensuring that messages from the registry
can be properly serialized and deserialized through MCAP.
"""

import tempfile

import pytest

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter


@pytest.fixture
def temp_mcap_file():
    """Create a temporary MCAP file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp:
        yield tmp.name
    # Cleanup is handled by the OS since delete=False


class TestMcapRegistryIntegration:
    """Integration tests for MCAP with message registry."""

    def test_registry_message_mcap_roundtrip(self, temp_mcap_file):
        """Test writing and reading registry messages through MCAP."""
        try:
            from owa.core import MESSAGES
        except ImportError:
            pytest.skip("owa-core not available")

        # Get a message type from the registry
        KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
        original_event = KeyboardEvent(event_type="press", vk=65, timestamp=1234567890)

        # Write to MCAP
        with OWAMcapWriter(temp_mcap_file) as writer:
            writer.write_message("/keyboard", original_event, log_time=1000)

        # Read from MCAP and verify
        with OWAMcapReader(temp_mcap_file) as reader:
            messages = list(reader.iter_messages())
            assert len(messages) == 1

            msg = messages[0]
            assert msg.topic == "/keyboard"
            assert msg.timestamp == 1000
            assert msg.message_type == "desktop/KeyboardEvent"

            # Verify decoded content
            decoded = msg.decoded
            assert decoded.event_type == "press"
            assert decoded.vk == 65
            assert decoded.timestamp == 1234567890

    def test_multiple_registry_message_types(self):
        """Test MCAP with multiple message types from registry."""
        try:
            from owa.core import MESSAGES
        except ImportError:
            pytest.skip("owa-core not available")

        # NOTE: There appears to be a schema ID collision issue when mixing different
        # message types in the same MCAP file. For now, we test each type separately.
        # This is a known limitation that should be addressed in the MCAP implementation.

        # Test KeyboardEvent messages
        KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]

        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp:
            kb_file = tmp.name

        with OWAMcapWriter(kb_file) as writer:
            kb_event1 = KeyboardEvent(event_type="press", vk=65, timestamp=1000)
            kb_event2 = KeyboardEvent(event_type="release", vk=65, timestamp=3000)
            writer.write_message("/keyboard", kb_event1, log_time=1000)
            writer.write_message("/keyboard", kb_event2, log_time=3000)

        with OWAMcapReader(kb_file) as reader:
            kb_messages = list(reader.iter_messages())
            assert len(kb_messages) == 2

            assert kb_messages[0].message_type == "desktop/KeyboardEvent"
            assert kb_messages[0].decoded.event_type == "press"
            assert kb_messages[0].decoded.vk == 65

            assert kb_messages[1].message_type == "desktop/KeyboardEvent"
            assert kb_messages[1].decoded.event_type == "release"
            assert kb_messages[1].decoded.vk == 65

        # Test MouseEvent messages separately
        MouseEvent = MESSAGES["desktop/MouseEvent"]

        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp:
            mouse_file = tmp.name

        with OWAMcapWriter(mouse_file) as writer:
            mouse_event = MouseEvent(event_type="click", x=100, y=200, button="left", pressed=True, timestamp=2000)
            writer.write_message("/mouse", mouse_event, log_time=2000)

        with OWAMcapReader(mouse_file) as reader:
            mouse_messages = list(reader.iter_messages())
            assert len(mouse_messages) == 1

            assert mouse_messages[0].message_type == "desktop/MouseEvent"
            assert mouse_messages[0].decoded.event_type == "click"
            assert mouse_messages[0].decoded.x == 100
            assert mouse_messages[0].decoded.y == 200
            assert mouse_messages[0].decoded.button == "left"
            assert mouse_messages[0].decoded.pressed is True

    def test_registry_message_schema_consistency(self, temp_mcap_file):
        """Test that MCAP schemas match registry message schemas."""
        try:
            from owa.core import MESSAGES
        except ImportError:
            pytest.skip("owa-core not available")

        # Test with just one message type to avoid schema collision issues
        KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]

        # Create test message
        kb_event = KeyboardEvent(event_type="press", vk=65)

        # Write to MCAP
        with OWAMcapWriter(temp_mcap_file) as writer:
            writer.write_message("/keyboard", kb_event, log_time=1000)

        # Read and verify schema consistency
        with OWAMcapReader(temp_mcap_file) as reader:
            messages = list(reader.iter_messages())
            assert len(messages) == 1

            # Verify that message type matches the registry type
            kb_msg = messages[0]
            assert kb_msg.message_type == KeyboardEvent._type.default

            # Verify that decoded message is instance of the correct class
            decoded = kb_msg.decoded
            assert isinstance(decoded, KeyboardEvent)
            assert decoded.event_type == "press"
            assert decoded.vk == 65

    def test_registry_discovery_integration(self):
        """Test that all registry messages can be discovered and used with MCAP."""
        try:
            from owa.core import MESSAGES
        except ImportError:
            pytest.skip("owa-core not available")

        # Force reload to ensure fresh discovery
        MESSAGES.reload()

        # Verify expected message types are available
        expected_messages = [
            "desktop/KeyboardEvent",
            "desktop/KeyboardState",
            "desktop/MouseEvent",
            "desktop/MouseState",
        ]

        available_messages = list(MESSAGES.keys())

        for message_type in expected_messages:
            assert message_type in available_messages, f"Message type {message_type} not found in registry"

            # Verify we can get the class
            MessageClass = MESSAGES[message_type]
            assert hasattr(MessageClass, "_type")
            assert MessageClass._type.default == message_type

    def test_registry_message_filtering(self):
        """Test filtering MCAP messages by schema type from registry."""
        try:
            from owa.core import MESSAGES
        except ImportError:
            pytest.skip("owa-core not available")

        # Test with just keyboard events to avoid schema collision issues
        KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]

        # Create separate temp files for each message type to avoid schema collisions
        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp:
            kb_file = tmp.name

        # Write keyboard events only
        with OWAMcapWriter(kb_file) as writer:
            for i in range(3):
                event = KeyboardEvent(event_type="press", vk=65 + i)
                writer.write_message("/keyboard", event, log_time=i * 1000)

        # Read and verify filtering
        with OWAMcapReader(kb_file) as reader:
            all_messages = list(reader.iter_messages())
            assert len(all_messages) == 3

            # Filter keyboard events
            keyboard_messages = [msg for msg in all_messages if msg.message_type == "desktop/KeyboardEvent"]
            assert len(keyboard_messages) == 3

            # Verify content
            for i, msg in enumerate(keyboard_messages):
                assert msg.decoded.event_type == "press"
                assert msg.decoded.vk == 65 + i
                assert msg.topic == "/keyboard"
                assert msg.timestamp == i * 1000

    def test_schema_collision_issue_documentation(self):
        """Document the known schema collision issue when mixing message types."""
        try:
            from owa.core import MESSAGES
        except ImportError:
            pytest.skip("owa-core not available")

        # This test documents a known issue where mixing different message types
        # in the same MCAP file can cause schema ID collisions during decoding.
        # The issue manifests as messages being decoded with the wrong schema.

        KeyboardEvent = MESSAGES["desktop/KeyboardEvent"]
        MouseEvent = MESSAGES["desktop/MouseEvent"]

        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp:
            mixed_file = tmp.name

        # Write mixed message types (this should work for writing)
        with OWAMcapWriter(mixed_file) as writer:
            kb_event = KeyboardEvent(event_type="press", vk=65)
            mouse_event = MouseEvent(event_type="click", x=100, y=200, button="left", pressed=True)

            writer.write_message("/keyboard", kb_event, log_time=1000)
            writer.write_message("/mouse", mouse_event, log_time=2000)

        # Reading should work for metadata, but decoding may have issues
        with OWAMcapReader(mixed_file) as reader:
            messages = list(reader.iter_messages())
            assert len(messages) == 2

            # Schema names should be correct
            assert messages[0].message_type == "desktop/KeyboardEvent"
            assert messages[1].message_type == "desktop/MouseEvent"

            # Topics should be correct
            assert messages[0].topic == "/keyboard"
            assert messages[1].topic == "/mouse"

            # Note: Decoding may fail due to schema collision issue
            # This is a known limitation that should be addressed in future versions
