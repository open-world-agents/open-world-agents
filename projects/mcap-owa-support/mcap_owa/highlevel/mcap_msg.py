import functools
from typing import Any

from mcap.records import Channel, Message, Schema
from pydantic import BaseModel

from mcap_owa.decode_utils import get_decode_function


class McapMessage(BaseModel):
    """
    A wrapper around MCAP message data that provides lazy evaluation of high-level properties.

    This class stores the 4 core fields (topic, timestamp, message, message_type) directly
    and provides convenient access to decoded content.
    """

    topic: str
    timestamp: int
    message: bytes
    message_type: str

    # Non-serialized decode configuration
    model_config = {"extra": "forbid"}

    def __init__(self, *, decode_args: dict = {}, **data):
        super().__init__(**data)
        # Store decode parameters as private attributes (not serialized)
        self._decode_args = {"return_dict": False, "return_dict_on_failure": False, **decode_args}

    @classmethod
    def from_mcap_primitives(
        cls,
        schema: Schema,
        channel: Channel,
        message: Message,
        *,
        decode_args: dict = None,
    ) -> "McapMessage":
        """
        Create a McapMessage from MCAP primitive objects.

        Args:
            schema: MCAP Schema object
            channel: MCAP Channel object
            message: MCAP Message object
            decode_args: Optional dictionary of decode arguments (return_dict, return_dict_on_failure)

        Returns:
            McapMessage instance
        """
        return cls(
            topic=channel.topic,
            timestamp=message.log_time,
            message=message.data,
            message_type=schema.name,
            decode_args=decode_args,
        )

    @functools.cached_property
    def decoded(self) -> Any:
        """
        Get the decoded message content. This is lazily evaluated.

        :return: Decoded message content
        """
        # Use automatic decode function generation
        decode_fn = get_decode_function(self.message_type, **self._decode_args)
        if decode_fn is not None:
            return decode_fn(self.message)
        else:
            raise ValueError(f"Could not generate decode function for message type '{self.message_type}'")

    def __repr__(self) -> str:
        return f"McapMessage(topic={self.topic}, timestamp={self.timestamp}, message_type={self.message_type})"
