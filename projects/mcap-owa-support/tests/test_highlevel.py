import tempfile

import pytest
from owa.env.desktop.msg import KeyboardEvent

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter


@pytest.fixture
def temp_mcap_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = tmpdir + "/output.mcap"
        yield file_path


def test_write_and_read_messages(temp_mcap_file):
    file_path = temp_mcap_file
    topic = "/chatter"
    event = KeyboardEvent(event_type="press", vk=1)

    with OWAMcapWriter(file_path) as writer:
        for i in range(0, 10):
            publish_time = i
            writer.write_message(topic, event, log_time=publish_time)

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
    event = KeyboardEvent(event_type="press", vk=65)

    with OWAMcapWriter(file_path) as writer:
        writer.write_message(topic, event, log_time=1000)

    with OWAMcapReader(file_path) as reader:
        messages = list(reader.iter_messages())
        assert len(messages) == 1

        msg = messages[0]
        # Test all properties
        assert msg.topic == topic
        assert msg.timestamp == 1000
        assert isinstance(msg.message, bytes)
        assert msg.message_type == "owa.env.desktop.msg.KeyboardEvent"

        # Test lazy decoded property
        decoded = msg.decoded
        assert decoded.event_type == "press"
        assert decoded.vk == 65

        # Test that decoded is cached (same object)
        assert msg.decoded is decoded


def test_schema_based_filtering(temp_mcap_file):
    """Test filtering messages by schema name."""
    file_path = temp_mcap_file

    with OWAMcapWriter(file_path) as writer:
        # Write different message types
        keyboard_event = KeyboardEvent(event_type="press", vk=65)
        writer.write_message("/keyboard", keyboard_event, log_time=1000)

        # Write another keyboard event
        keyboard_event2 = KeyboardEvent(event_type="release", vk=65)
        writer.write_message("/keyboard", keyboard_event2, log_time=2000)

    with OWAMcapReader(file_path) as reader:
        # Filter by schema name
        keyboard_messages = [
            msg for msg in reader.iter_messages() if msg.message_type == "owa.env.desktop.msg.KeyboardEvent"
        ]

        assert len(keyboard_messages) == 2
        assert keyboard_messages[0].decoded.event_type == "press"
        assert keyboard_messages[1].decoded.event_type == "release"
