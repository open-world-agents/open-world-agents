from typing import Optional

from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.records import Schema
from mcap.well_known import MessageEncoding, SchemaEncoding

from .decode_utils import dict_decoder, get_decode_function


class DecoderFactory(McapDecoderFactory):
    def __init__(self):
        """Initialize the decoder factory."""

    def decoder_for(self, message_encoding: str, schema: Optional[Schema]):
        if message_encoding != MessageEncoding.JSON or schema is None or schema.encoding != SchemaEncoding.JSONSchema:
            return None

        return get_decode_function(schema.name)


__all__ = ["dict_decoder", "get_decode_function", "DecoderFactory"]
