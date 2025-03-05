from typing import Any, Optional

import orjson
from easydict import EasyDict
from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.records import Schema
from mcap.well_known import MessageEncoding, SchemaEncoding


# Copied from: https://github.com/foxglove/mcap/blob/6c9ce28b227b379164b2e61e0fd02f365c5442d9/python/mcap/tests/test_reader.py#L152
class DecoderFactory(McapDecoderFactory):
    def __init__(self):
        # TODO: implement decoders for OWA
        # self._decoders: Dict[int, Any] = {}
        pass

    def decoder_for(self, message_encoding: str, schema: Optional[Schema]):
        if message_encoding != MessageEncoding.JSON or schema is None or schema.encoding != SchemaEncoding.JSONSchema:
            return None

        def decoder(message_data: bytes) -> Any:
            return EasyDict(orjson.loads(message_data))

        return decoder
