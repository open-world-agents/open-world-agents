import importlib
import io
from typing import Any, Optional

import orjson
from easydict import EasyDict
from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.records import Schema
from mcap.well_known import MessageEncoding, SchemaEncoding


# Copied from: https://github.com/foxglove/mcap/blob/6c9ce28b227b379164b2e61e0fd02f365c5442d9/python/mcap/tests/test_reader.py#L152
class DecoderFactory(McapDecoderFactory):
    def __init__(self):
        self._decoders: dict[int, Any] = {}

    def decoder_for(self, message_encoding: str, schema: Optional[Schema]):
        if message_encoding != MessageEncoding.JSON or schema is None or schema.encoding != SchemaEncoding.JSONSchema:
            return None

        def decoder(message_data: bytes) -> Any:
            if schema.id not in self._decoders:
                module, class_name = schema.name.rsplit(".", 1)  # e.g. "owa.env.desktop.msg.KeyboardState"
                try:
                    mod = importlib.import_module(module)
                    cls = getattr(mod, class_name)

                    def decoder(message_data: bytes) -> Any:
                        buffer = io.BytesIO(message_data)
                        return cls.deserialize(buffer)

                    self._decoders[schema.id] = decoder
                except ImportError as e:
                    raise RuntimeError(f"Error importing module {module}: {e}")
                except AttributeError as e:
                    raise RuntimeError(f"Error accessing class {class_name} in module {module}: {e}")
                except Exception as e:
                    raise RuntimeError(f"Error deserializing message: {e}")
            return self._decoders[schema.id](message_data)

        return decoder
