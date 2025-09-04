import io
import warnings
from typing import Any, Callable, TypeAlias

import orjson
from easydict import EasyDict

try:
    from owa.core import MESSAGES
except ImportError:
    # Fallback if owa.core is not available
    MESSAGES = None

# Type alias for decode functions
DecodeFunction: TypeAlias = Callable[[bytes], Any]


def dict_decoder(message_data: bytes) -> Any:
    return EasyDict(orjson.loads(message_data))


def create_message_decoder(message_type: str, fallback: bool = False) -> DecodeFunction:
    """
    Create a decode function for a specific OWA message type/schema name.

    This function attempts to create a decoder for the specified message type using
    the domain-based format (domain/MessageType) via the MESSAGES registry. If the
    message type is not found and fallback is enabled, it falls back to dictionary
    decoding with EasyDict.

    :param message_type: The message type or schema name (e.g., "desktop/KeyboardEvent")
    :param fallback: Whether to fall back to dictionary decoding on failure
    :return: DecodeFunction that can decode messages of this type
    :raises ValueError: If message type is unsupported and fallback is disabled
    """
    cls = None

    # Try to find message class in registry
    if MESSAGES:
        try:
            cls = MESSAGES[message_type]
        except KeyError:
            pass  # Fall through to fallback handling

    if cls is None:
        if fallback:
            if "/" in message_type:
                warnings.warn(
                    f"Domain-based message '{message_type}' not found in registry. Falling back to dictionary decoding."
                )
            else:
                warnings.warn(
                    f"Message type '{message_type}' not found in registry (expected domain-based format 'domain/MessageType'). "
                    f"Falling back to dictionary decoding."
                )
            return dict_decoder

        raise ValueError(f"Unsupported message type: {message_type}")

    def decoder(message_data: bytes) -> Any:
        try:
            buffer = io.BytesIO(message_data)
            return cls.deserialize(buffer)
        except Exception as e:
            if fallback:
                warnings.warn(
                    f"Failed to decode message of type {message_type}: {e}. Falling back to dictionary decoding."
                )
                return dict_decoder(message_data)
            raise e

    return decoder


def get_decode_function(
    message_type: str, *, return_dict: bool = False, return_dict_on_failure: bool = False
) -> DecodeFunction:
    """
    Convenience function to get a decode function using the global cache.

    :param message_type: The message type or schema name
    :return: DecodeFunction that can decode messages of this type, or None if unsupported
    """
    if return_dict:
        return dict_decoder
    else:
        return create_message_decoder(message_type, fallback=return_dict_on_failure)
