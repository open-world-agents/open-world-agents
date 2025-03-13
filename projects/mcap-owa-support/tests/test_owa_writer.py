from io import BytesIO

from mcap.reader import make_reader
from owa.core.message import OWAMessage

from mcap_owa.decoder import DecoderFactory
from mcap_owa.writer import Writer as OWAWriter


class String(OWAMessage):
    _type = "std_msgs/String"
    data: str


def read_owa_messages(stream: BytesIO):
    reader = make_reader(stream, decoder_factories=[DecoderFactory()])
    return reader.iter_decoded_messages()


def test_write_messages():
    output = BytesIO()
    writer = OWAWriter(output=output)
    for i in range(0, 10):
        writer.write_message("/chatter", String(data=f"string message {i}"), i)
    writer.finish()

    output.seek(0)
    for index, msg in enumerate(read_owa_messages(output)):
        assert msg.channel.topic == "/chatter"
        assert msg.decoded_message.data == f"string message {index}"
        assert msg.message.log_time == index
