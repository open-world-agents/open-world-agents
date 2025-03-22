import functools
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Tuple, TypeAlias

import semantic_version
from mcap.reader import McapReader, make_reader

from mcap_owa.decoder import DecoderFactory

from .. import __version__

PathType: TypeAlias = str | Path


class OWAMcapReader:
    def __init__(self, file_path: PathType, deserialize_to_objects: bool = False):
        self.file_path = file_path
        self._file = open(file_path, "rb")
        self.reader: McapReader = make_reader(
            self._file, decoder_factories=[DecoderFactory(deserialize_to_objects=deserialize_to_objects)]
        )
        self.__finished = False

        # Check profile of mcap file
        header = self.reader.get_header()
        assert header.profile == "owa"

        # Check version compatibility
        libversion = header.library
        m = re.match(r"mcap-owa-support (?P<version>[\d.]+); mcap (?P<mcap_version>[\d.]+)", libversion)

        # assert by semantic versioning: patch version change is backward compatible
        if not semantic_version.match(f"~{m['version']}", __version__):
            warnings.warn(f"Reader version {__version__} is not compatible with writer version {m['version']}")

    def finish(self):
        if not self.__finished:
            self.__finished = True
            self._file.close()

    @functools.cached_property
    def topics(self):
        summary = self.reader.get_summary()
        self._topics = {}

        for channel_id, channel in summary.channels.items():
            self._topics[channel.topic] = (channel, summary.schemas[channel.schema_id])
        return list(self._topics.keys())

    @functools.cached_property
    def start_time(self):
        summary = self.reader.get_summary()
        return summary.statistics.message_start_time

    @functools.cached_property
    def end_time(self):
        summary = self.reader.get_summary()
        return summary.statistics.message_end_time

    @functools.cached_property
    def duration(self):
        return self.end_time - self.start_time

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()

    def iter_messages(
        self,
        topics: Optional[Iterable[str]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        log_time_order: bool = True,
        reverse: bool = False,
    ) -> Iterator[Tuple[str, int, bytes]]:
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
        for schema, channel, message in self.reader.iter_messages(
            topics=topics,
            start_time=start_time,
            end_time=end_time,
            log_time_order=log_time_order,
            reverse=reverse,
        ):
            return channel.topic, message.log_time, message.data

    def iter_decoded_messages(
        self,
        topics: Optional[Iterable[str]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        log_time_order: bool = True,
        reverse: bool = False,
    ) -> Iterator[Tuple[str, int, Any]]:
        """iterates through messages in an MCAP, decoding their contents.

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
        for schema, channel, message, decoded in self.reader.iter_decoded_messages(
            topics=topics,
            start_time=start_time,
            end_time=end_time,
            log_time_order=log_time_order,
            reverse=reverse,
        ):
            yield channel.topic, message.log_time, decoded
