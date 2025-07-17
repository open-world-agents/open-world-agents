import time
from io import BufferedWriter, BytesIO
from pathlib import Path
from typing import IO, Any, Dict, Optional, Union

import mcap
import orjson
from mcap.well_known import SchemaEncoding
from mcap.writer import CompressionType
from mcap.writer import Writer as McapWriter

from . import __version__


def _library_identifier():
    mcap_version = getattr(mcap, "__version__", "<=0.0.10")
    return f"mcap-owa-support {__version__}; mcap {mcap_version}"


class Writer:
    """
    A writer for MCAP streams that writes OWAMessage objects.


    Example:
    ```python
    from pathlib import Path

    from owa.core.message import OWAMessage

    from mcap_owa.writer import Writer as OWAWriter


    class String(OWAMessage):
        _type = "std_msgs/String"
        data: str


    def main():
        output_file = Path("output.mcap")
        stream = output_file.open("wb")
        with OWAWriter(stream) as writer:
            topic = "/chatter"
            event = String(data="string message")
            for i in range(0, 10):
                publish_time = i
                writer.write_message(topic, event, publish_time=publish_time)


    if __name__ == "__main__":
        main()

    """

    def __init__(
        self,
        output: Union[str, IO[Any], BufferedWriter, Path],
        chunk_size: int = 1024 * 1024,
        compression: CompressionType = CompressionType.ZSTD,
        enable_crcs: bool = True,
    ):
        if isinstance(output, Path):
            output = output.as_posix()

        self.__writer = McapWriter(
            output=output,
            chunk_size=chunk_size,
            compression=compression,
            enable_crcs=enable_crcs,
        )
        self.__schema_ids: Dict[str, int] = {}
        self.__channel_ids: Dict[str, int] = {}
        self.__writer.start(profile="owa", library=_library_identifier())
        self.__finished = False

    def finish(self):
        """
        Finishes writing to the MCAP stream. This must be called before the stream is closed.
        """
        if not self.__finished:
            self.__writer.finish()
            self.__finished = True

    def write_message(
        self,
        topic: str,
        message: Any,
        log_time: Optional[int] = None,
        publish_time: Optional[int] = None,
        sequence: int = 0,
    ):
        """
        Writes a message to the MCAP stream, automatically registering schemas and channels as
        needed.

        :param topic: The topic of the message.
        :param message: The message to write.
        :param log_time: The time at which the message was logged as a nanosecond UNIX timestamp.
            Will default to the current time if not specified.
        :param publish_time: The time at which the message was published as a nanosecond UNIX
            timestamp. Will default to ``log_time`` if not specified.
        :param sequence: An optional sequence number.
        """
        if message._type not in self.__schema_ids:
            if hasattr(message, "get_schema"):
                schema_data = orjson.dumps(message.get_schema())
            else:
                # Fallback to class name as schema
                schema_data = message.__class__.__name__.encode()
            schema_id = self.__writer.register_schema(
                name=message._type,
                data=schema_data,
                encoding=SchemaEncoding.JSONSchema,
            )
            self.__schema_ids[message._type] = schema_id
        schema_id = self.__schema_ids[message._type]

        if topic not in self.__channel_ids:
            channel_id = self.__writer.register_channel(
                topic=topic,
                message_encoding="json",
                schema_id=schema_id,
            )
            self.__channel_ids[topic] = channel_id
        channel_id = self.__channel_ids[topic]

        # Type check for message
        scheme_for_channel = self.__writer.__channels[channel_id].schema_id
        assert scheme_for_channel == schema_id, (
            f"Schema ID Mismatch Error:\n"
            f"-------------------------\n"
            f"Channel (ID: {channel_id}):\n"
            f"  Schema ID:   {scheme_for_channel}\n"
            f"  Schema Name: {self.__writer.__schemas[scheme_for_channel].name}\n\n"
            f"Message (Type: {message._type}):\n"
            f"  Schema ID:   {schema_id}\n"
            f"  Schema Name: {message._type}\n\n"
            f"Error: Each channel must use a consistent message type.\n"
            f"Solution: Ensure you're publishing the correct message type on this channel."
        )

        buffer = BytesIO()
        message.serialize(buffer)
        if log_time is None:
            log_time = time.time_ns()
        if publish_time is None:
            publish_time = log_time

        self.__writer.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=publish_time,
            sequence=sequence,
            data=buffer.getvalue(),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_: Any, exc_type_: Any, tb_: Any):
        self.finish()
