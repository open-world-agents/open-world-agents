import importlib
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

    This internal function attempts to create a decoder for the specified message type by:
    1. First trying the new domain-based format (domain/MessageType) via MESSAGES registry
    2. Then trying the old module-based format (module.path.ClassName) via importlib
    3. Finally falling back to dictionary decoding with EasyDict

    :param message_type: The message type or schema name (e.g., "desktop/KeyboardEvent" or "owa.env.desktop.msg.KeyboardState")
    :return: DecodeFunction that can decode messages of this type, or None if unsupported
    """
    cls = None

    # Try new domain-based format first
    if MESSAGES and "/" in message_type:
        try:
            cls = MESSAGES[message_type]
        except KeyError:
            pass  # Fall through to old format or dictionary decoding

    # Try old module-based format for backward compatibility
    if cls is None and "." in message_type:
        try:
            module, class_name = message_type.rsplit(".", 1)  # e.g. "owa.env.desktop.msg.KeyboardState"
            mod = importlib.import_module(module)
            cls = getattr(mod, class_name)
        except (ValueError, ImportError, AttributeError):
            pass  # Fall through to dictionary decoding

    if cls is None:
        if fallback:
            if "/" in message_type:
                warnings.warn(
                    f"Domain-based message '{message_type}' not found in registry. Falling back to dictionary decoding."
                )
            else:
                warnings.warn(
                    f"Failed to import module for schema '{message_type}'. Falling back to dictionary decoding."
                )
            return dict_decoder

        raise ValueError(f"Unsupported message type: {message_type}")

    def decoder(message_data: bytes) -> Any:
        buffer = io.BytesIO(message_data)
        return cls.deserialize(buffer)

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
