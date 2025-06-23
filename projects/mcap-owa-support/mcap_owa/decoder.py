from typing import Dict, Optional

from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.records import Schema
from mcap.well_known import MessageEncoding, SchemaEncoding

from .decode_utils import DecodeCache, DecodeFunction, get_decode_function


class DecoderFactory(McapDecoderFactory):
    def __init__(self):
        """Initialize the decoder factory."""
        self._decoders: Dict[int, DecodeFunction] = {}
        self._decode_cache = DecodeCache()

    def decoder_for(self, message_encoding: str, schema: Optional[Schema]):
        if message_encoding != MessageEncoding.JSON or schema is None or schema.encoding != SchemaEncoding.JSONSchema:
            return None

        decode_fn = get_decode_function(schema.name)
        if decode_fn is None:
            # This should not happen as _create_message_decoder always returns something
            # but we handle it gracefully just in case
            raise ValueError(f"Could not generate decode function for schema '{schema.name}'")

        return decode_fn
