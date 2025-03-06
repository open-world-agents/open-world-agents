import tempfile

import pytest
from owa_env_desktop.msg import KeyboardEvent

from mcap_owa.highlevel import Reader, Writer


@pytest.fixture
def temp_mcap_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = tmpdir + "/output.mcap"
        yield file_path


def test_write_and_read_messages(temp_mcap_file):
    file_path = temp_mcap_file
    topic = "/chatter"
    event = KeyboardEvent(event_type="press", vk=1)

    with Writer(file_path) as writer:
        for i in range(0, 10):
            publish_time = i
            writer.write_message(topic, event, publish_time=publish_time)

    with Reader(file_path) as reader:
        messages = list(reader.iter_decoded_messages())
        assert len(messages) == 10
        for i, (schema, channel, message, decoded) in enumerate(messages):
            assert channel.topic == topic
            assert decoded.event_type == "press"
            assert decoded.vk == 1
            assert message.publish_time == i
