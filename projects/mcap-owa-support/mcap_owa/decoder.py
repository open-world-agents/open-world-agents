import importlib
import io
import warnings
from typing import Any, Dict, Optional

import orjson
from easydict import EasyDict
from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.records import Schema
from mcap.well_known import MessageEncoding, SchemaEncoding

try:
    from owa.core import MESSAGES
except ImportError:
    # Fallback if owa.core is not available
    MESSAGES = None


class DecoderFactory(McapDecoderFactory):
    def __init__(self):
        """Initialize the decoder factory."""
        self._decoders: Dict[int, Any] = {}

    def decoder_for(self, message_encoding: str, schema: Optional[Schema]):
        if message_encoding != MessageEncoding.JSON or schema is None or schema.encoding != SchemaEncoding.JSONSchema:
            return None

        # Decoder that attempts to load and use the actual message class
        def object_decoder(message_data: bytes) -> Any:
            if schema.id not in self._decoders:
                cls = None

                # Try new domain-based format first (OEP-0006)
                if MESSAGES and "/" in schema.name:
                    try:
                        cls = MESSAGES[schema.name]
                    except KeyError:
                        pass  # Fall through to old format or dictionary decoding

                # Try old module-based format for backward compatibility
                if cls is None and "." in schema.name:
                    try:
                        module, class_name = schema.name.rsplit(".", 1)  # e.g. "owa.env.desktop.msg.KeyboardState"
                        mod = importlib.import_module(module)
                        cls = getattr(mod, class_name)
                    except (ValueError, ImportError, AttributeError):
                        pass  # Fall through to dictionary decoding

                if cls is not None:
                    # Successfully found message class
                    def decoder(message_data: bytes) -> Any:
                        buffer = io.BytesIO(message_data)
                        return cls.deserialize(buffer)

                    self._decoders[schema.id] = decoder
                else:
                    # Fall back to dictionary decoding
                    if "/" in schema.name:
                        warnings.warn(
                            f"Domain-based message '{schema.name}' not found in registry. "
                            f"Falling back to dictionary decoding."
                        )
                    else:
                        warnings.warn(
                            f"Failed to import module for schema '{schema.name}'. Falling back to dictionary decoding."
                        )
                    self._decoders[schema.id] = lambda data: EasyDict(orjson.loads(data))
                    return self._decoders[schema.id](message_data)

            return self._decoders[schema.id](message_data)

        return object_decoder
