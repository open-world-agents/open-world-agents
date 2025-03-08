import functools
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, TypeAlias

from mcap.reader import McapReader, make_reader
from mcap.records import Channel, Message, Schema

from mcap_owa.decoder import DecoderFactory

PathType: TypeAlias = str | Path

"""
from io import BytesIO

from mcap.reader import make_reader
from owa.message import OWAMessage

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

"""


class Reader:
    def __init__(self, file_path: PathType):
        self.file_path = file_path
        self._file = open(file_path, "rb")
        self.reader = make_reader(self._file, decoder_factories=[DecoderFactory()])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._file.close()

    @functools.wraps(McapReader.iter_messages)
    def iter_messages(
        self,
        topics: Optional[Iterable[str]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        log_time_order: bool = True,
        reverse: bool = False,
    ) -> Iterator[Tuple[Optional[Schema], Channel, Message]]:
        """iterates through the messages in an MCAP.

        :param topics: if not None, only messages from these topics will be returned.
        :param start_time: an integer nanosecond timestamp. if provided, messages logged before this
            timestamp are not included.
        :param end_time: an integer nanosecond timestamp. if provided, messages logged at or after
            this timestamp are not included.
        :param log_time_order: if True, messages will be yielded in ascending log time order. If
            False, messages will be yielded in the order they appear in the MCAP file.
        :param reverse: if both ``log_time_order`` and ``reverse`` are True, messages will be
            yielded in descending log time order.
        """
        return self.reader.iter_messages(
            topics=topics,
            start_time=start_time,
            end_time=end_time,
            log_time_order=log_time_order,
            reverse=reverse,
        )

    @functools.wraps(McapReader.iter_decoded_messages)
    def iter_decoded_messages(self, *args, **kwargs):
        return self.reader.iter_decoded_messages(*args, **kwargs)
