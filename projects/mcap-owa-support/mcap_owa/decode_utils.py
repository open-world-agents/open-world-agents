import importlib
import io
import warnings
from typing import Any, Callable, Dict, Optional, TypeAlias

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


def _create_message_decoder(message_type: str) -> Optional[DecodeFunction]:
    """
    Create a decode function for a specific OWA message type/schema name.

    This internal function attempts to create a decoder for the specified message type by:
    1. First trying the new domain-based format (domain/MessageType) via MESSAGES registry
    2. Then trying the old module-based format (module.path.ClassName) via importlib
    3. Finally falling back to dictionary decoding with EasyDict

    :param message_type: The message type or schema name (e.g., "desktop/KeyboardEvent" or "owa.env.desktop.msg.KeyboardState")
    :return: DecodeFunction that can decode messages of this type, or None if unsupported
    """
    if not message_type:
        return None

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

    if cls is not None:
        # Successfully found message class
        def decoder(message_data: bytes) -> Any:
            buffer = io.BytesIO(message_data)
            return cls.deserialize(buffer)

        return decoder
    else:
        # Fall back to dictionary decoding
        if "/" in message_type:
            warnings.warn(
                f"Domain-based message '{message_type}' not found in registry. Falling back to dictionary decoding."
            )
        else:
            warnings.warn(f"Failed to import module for schema '{message_type}'. Falling back to dictionary decoding.")

        return dict_decoder


class DecodeCache:
    """
    Cache for decode functions to avoid regenerating them for the same message types.
    """

    def __init__(self):
        self._cache: Dict[str, DecodeFunction] = {}

    def get_decode_function(self, message_type: str) -> Optional[DecodeFunction]:
        """
        Get a decode function for the given message type, using cache if available.

        :param message_type: The message type or schema name
        :return: DecodeFunction that can decode messages of this type, or None if unsupported
        """
        if message_type not in self._cache:
            decode_fn = _create_message_decoder(message_type)
            if decode_fn is not None:
                self._cache[message_type] = decode_fn
            else:
                return None

        return self._cache[message_type]

    def clear(self):
        """Clear the decode function cache."""
        self._cache.clear()


# Global cache instance for convenience
_global_decode_cache = DecodeCache()


def get_decode_function(message_type: str) -> Optional[DecodeFunction]:
    """
    Convenience function to get a decode function using the global cache.

    :param message_type: The message type or schema name
    :return: DecodeFunction that can decode messages of this type, or None if unsupported
    """
    return _global_decode_cache.get_decode_function(message_type)
