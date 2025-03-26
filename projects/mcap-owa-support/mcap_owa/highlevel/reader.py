import functools
import io
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Tuple, TypeAlias, Union

import requests
import semantic_version
from mcap.reader import McapReader, make_reader

from mcap_owa.decoder import DecoderFactory

from .. import __version__

PathType: TypeAlias = Union[str, Path]


class OWAMcapReader:
    """
    Reader for OWA MCAP files that supports both local file paths and network URLs.

    This class provides functionality to read, parse, and iterate through messages
    in an MCAP file, with support for both local filesystem paths and remote HTTP/HTTPS URLs.
    """

    def __init__(self, file_path: PathType, deserialize_to_objects: bool = False):
        """
        Initialize an OWA MCAP reader.

        :param file_path: Path to the MCAP file. Can be either:
                          - A local file path (string or Path object)
                          - A network URL (http:// or https:// protocol)
                          e.g., "https://huggingface.co/datasets/.../example.mcap"
        :param deserialize_to_objects: If True, messages will be deserialized into
                                      objects, otherwise they will be kept as raw data.
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
            warnings.warn(f"Reader version {__version__} may not be compatible with writer version {m['version']}")

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
        """Iterates through the messages in an MCAP.

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
            yield channel.topic, message.log_time, message.data

    def iter_decoded_messages(
        self,
        topics: Optional[Iterable[str]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        log_time_order: bool = True,
        reverse: bool = False,
    ) -> Iterator[Tuple[str, int, Any]]:
        """Iterates through messages in an MCAP, decoding their contents.

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
