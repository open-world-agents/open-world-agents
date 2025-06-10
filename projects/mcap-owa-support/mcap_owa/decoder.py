import importlib
import io
import warnings
from typing import Any, Dict, Optional

import orjson
from easydict import EasyDict
from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.records import Schema
from mcap.well_known import MessageEncoding, SchemaEncoding


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
                try:
                    module, class_name = schema.name.rsplit(".", 1)  # e.g. "owa.env.desktop.msg.KeyboardState"
                except ValueError:
                    # Fall back to dictionary decoding on invalid schema name
                    warnings.warn(f"Invalid schema name {schema.name}. Falling back to dictionary decoding.")
                    self._decoders[schema.id] = lambda data: EasyDict(orjson.loads(data))
                    return self._decoders[schema.id](message_data)

                try:
                    mod = importlib.import_module(module)
                    cls = getattr(mod, class_name)

                    def decoder(message_data: bytes) -> Any:
                        buffer = io.BytesIO(message_data)
                        return cls.deserialize(buffer)

                    self._decoders[schema.id] = decoder
                except ImportError:
                    # Fall back to dictionary decoding on import error
                    warnings.warn(
                        f"Failed to import module {module} for schema {schema.name}. Falling back to dictionary decoding."
                    )
                    self._decoders[schema.id] = lambda data: EasyDict(orjson.loads(data))
                    return self._decoders[schema.id](message_data)
                except AttributeError as e:
                    raise RuntimeError(f"Error accessing class {class_name} in module {module}: {e}")
                except Exception as e:
                    raise RuntimeError(f"Error deserializing message: {e}")

            return self._decoders[schema.id](message_data)

        return object_decoder
