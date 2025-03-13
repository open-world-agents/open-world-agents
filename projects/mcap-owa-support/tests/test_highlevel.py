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
        messages = list(reader.iter_decoded_messages())
        assert len(messages) == 10
        for i, (_topic, timestamp, msg) in enumerate(messages):
            assert _topic == topic
            assert msg.event_type == "press"
            assert msg.vk == 1
            assert timestamp == i
