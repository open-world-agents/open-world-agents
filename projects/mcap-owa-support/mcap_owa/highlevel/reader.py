import functools
import io
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, TypeAlias, Union

import requests
from mcap.reader import McapReader, make_reader
from mcap.records import Channel, Message, Schema
from packaging import version
from packaging.specifiers import SpecifierSet

from mcap_owa.decode_utils import get_decode_function

from .. import __version__

PathType: TypeAlias = Union[str, Path]


@dataclass
class McapMessage:
    """
    A wrapper around MCAP message data that provides lazy evaluation of high-level properties.

    This class wraps the low-level (schema, channel, message) tuple from the MCAP reader
    and provides convenient access to topic, timestamp, raw data, schema name, and decoded content.
    """

    _schema: Schema
    _channel: Channel
    _message: Message

    @property
    def topic(self) -> str:
        """Get the topic name."""
        return self._channel.topic

    @property
    def timestamp(self) -> int:
        """Get the log timestamp in nanoseconds."""
        return self._message.log_time

    @property
    def message(self) -> bytes:
        """Get the raw message data."""
        return self._message.data

    @property
    def message_type(self) -> str:
        """Get the message type."""
        return self._schema.name

    @functools.cached_property
    def decoded(self) -> Any:
        """
        Get the decoded message content. This is lazily evaluated.

        :return: Decoded message content
        """
        # Use automatic decode function generation
        decode_fn = get_decode_function(self.message_type)
        if decode_fn is not None:
            return decode_fn(self._message.data)
        else:
            raise ValueError(f"Could not generate decode function for message type '{self.message_type}'")

    def __repr__(self) -> str:
        return f"McapMessage(topic={self.topic}, timestamp={self.timestamp}, message_type={self.message_type})"


class OWAMcapReader:
    """
    Reader for OWA MCAP files that supports both local file paths and network URLs.

    This class provides functionality to read, parse, and iterate through messages
    in an MCAP file, with support for both local filesystem paths and remote HTTP/HTTPS URLs.
    """

    def __init__(self, file_path: PathType):
        """
        Initialize an OWA MCAP reader.

        :param file_path: Path to the MCAP file. Can be either:
                          - A local file path (string or Path object)
                          - A network URL (http:// or https:// protocol)
                          e.g., "https://huggingface.co/datasets/.../example.mcap"
        """
        self.file_path = file_path

        # Check if the path is a URL or local file
        if isinstance(file_path, str) and (file_path.startswith("http://") or file_path.startswith("https://")):
            # Handle network path (URL)
            response = requests.get(file_path, stream=True)
            response.raise_for_status()  # Raise exception for HTTP errors
            self._file = io.BytesIO(response.content)
            self._is_network_path = True
        else:
            # Handle local file path
            self._file = open(file_path, "rb")
            self._is_network_path = False

        self.reader: McapReader = make_reader(self._file)
        self.__finished = False

        # Check profile of mcap file
        header = self.reader.get_header()
        assert header.profile == "owa", f"MCAP file is not an OWA file (profile={header.profile})"

        # Check version compatibility
        libversion = header.library
        m = re.match(
            r"mcap-owa-support (?P<version>[\d\.]+[a-zA-Z0-9\-]*)\s*;\s*mcap (?P<mcap_version>[\d\.]+[a-zA-Z0-9\-]*)",
            libversion,
        )

        # Use packaging.version instead of semantic_version for PEP 440 compliance
        self._file_version_str = m["version"] if m else "unknown"
        self._mcap_version_str = m["mcap_version"] if m else "unknown"
        current_version = version.Version(__version__)

        # ~=X.Y.Z is equivalent to >=X.Y.Z, ==X.Y.* (allows patch-level changes)
        if m:
            specifier = SpecifierSet(f"~={self._file_version_str}")
            if current_version not in specifier:
                warnings.warn(
                    f"Reader version {__version__} may not be compatible with writer version {self._file_version_str}"
                )

    def finish(self):
        """Close the file and release resources."""
        if not self.__finished:
            self.__finished = True
            self._file.close()

    @functools.cached_property
    def topics(self):
        """
        Get a list of topics in the MCAP file.

        :return: List of topic names
        """
        summary = self.reader.get_summary()
        self._topics = {}

        for channel_id, channel in summary.channels.items():
            self._topics[channel.topic] = (channel, summary.schemas[channel.schema_id])
        return list(self._topics.keys())

    @functools.cached_property
    def start_time(self):
        """Get the start time of the MCAP file in nanoseconds."""
        summary = self.reader.get_summary()
        return summary.statistics.message_start_time

    @functools.cached_property
    def end_time(self):
        """Get the end time of the MCAP file in nanoseconds."""
        summary = self.reader.get_summary()
        return summary.statistics.message_end_time

    @functools.cached_property
    def duration(self):
        """Get the duration of the MCAP file in nanoseconds."""
        return self.end_time - self.start_time

    @property
    def file_version(self) -> str:
        """Get the mcap-owa-support version that created this file."""
        return self._file_version_str

    @property
    def mcap_version(self) -> str:
        """Get the mcap library version that created this file."""
        return self._mcap_version_str

    @property
    def schemas(self) -> Dict[int, Any]:
        """Get all schemas in the MCAP file."""
        return self.reader.get_summary().schemas

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
    ) -> Iterator[McapMessage]:
        """Iterates through the messages in an MCAP, returning rich message objects.

        :param topics: if not None, only messages from these topics will be returned.
        :param start_time: an integer nanosecond timestamp. if provided, messages logged before this
            timestamp are not included.
        :param end_time: an integer nanosecond timestamp. if provided, messages logged at or after
            this timestamp are not included.
        :param log_time_order: if True, messages will be yielded in ascending log time order. If
            False, messages will be yielded in the order they appear in the MCAP file.
        :param reverse: if both ``log_time_order`` and ``reverse`` are True, messages will be
            yielded in descending log time order.
        :returns: Iterator yielding McapMessage objects with lazy-evaluated properties
        """
        for schema, channel, message in self.reader.iter_messages(
            topics=topics,
            start_time=start_time,
            end_time=end_time,
            log_time_order=log_time_order,
            reverse=reverse,
        ):
            yield McapMessage(schema, channel, message)
