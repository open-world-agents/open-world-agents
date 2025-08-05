from typing import Dict, Optional, Union

from owa.core import OWAMessage

from ..writer import Writer as _Writer
from .mcap_msg import McapMessage


class OWAMcapWriter(_Writer):
    """
    A high-level interface for writing OWA messages to MCAP files.
    """

    def write_message(
        self, message: Union[McapMessage, OWAMessage], topic: Optional[str] = None, timestamp: Optional[int] = None
    ):
        """
        Write a message to the MCAP stream.

        Args:
            message: Message to write, either as McapMessage or OWAMessage
            topic: Optional topic name, required if message is OWAMessage
            timestamp: Optional timestamp, required if message is OWAMessage
        """
        if isinstance(message, McapMessage):
            if topic is not None and message.topic != topic:
                raise ValueError(f"Topic mismatch: {message.topic} != {topic}")
            if timestamp is not None and message.timestamp != timestamp:
                raise ValueError(f"Timestamp mismatch: {message.timestamp} != {timestamp}")
            topic = message.topic
            timestamp = message.timestamp
            message = message.decoded

        if topic is None:
            raise ValueError("Topic is required when message is OWAMessage")

        super().write_message(topic=topic, message=message, log_time=timestamp)

    def write_metadata(self, name: str, data: Dict[str, str]):
        """
        Write metadata to the MCAP stream.

        Args:
            name: Name/key for the metadata
            data: Metadata as Dict[str, str]
        """
        self._writer.add_metadata(name=name, data=data)
