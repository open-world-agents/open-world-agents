import warnings
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
    # Suppress warnings only for mock message creation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        string_messages = [String(data=f"string message {i}") for i in range(0, 10)]

    output = BytesIO()
    writer = OWAWriter(output=output)
    for i, string_msg in enumerate(string_messages):
        writer.write_message("/chatter", string_msg, i)
    writer.finish()

    # Suppress warnings only for reading mock messages
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Domain-based message 'std_msgs/String' not found in registry.*", UserWarning
        )

        output.seek(0)
        for index, msg in enumerate(read_owa_messages(output)):
            assert msg.channel.topic == "/chatter"
            assert msg.decoded_message.data == f"string message {index}"
            assert msg.message.log_time == index
