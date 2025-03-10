from io import BytesIO

from mcap.reader import make_reader
from mcap_owa.decoder import DecoderFactory


def read_owa_messages(stream: BytesIO):
    reader = make_reader(stream, decoder_factories=[DecoderFactory()])
    return reader.iter_decoded_messages()


if __name__ == "__main__":
    with open("test.mcap", "rb") as f:
        for schema, channel, msg, decoded_msg in read_owa_messages(BytesIO(f.read())):
            print(schema.name, channel.topic, msg, decoded_msg)
            print(type(msg), type(decoded_msg))
