from typing import Any, Dict, Optional

from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.records import Schema
from mcap.well_known import MessageEncoding, SchemaEncoding

from .decode_utils import DecodeCache, DecodeFunction


class DecoderFactory(McapDecoderFactory):
    def __init__(self):
        """Initialize the decoder factory."""
        self._decoders: Dict[int, DecodeFunction] = {}
        self._decode_cache = DecodeCache()

    def decoder_for(self, message_encoding: str, schema: Optional[Schema]):
        if message_encoding != MessageEncoding.JSON or schema is None or schema.encoding != SchemaEncoding.JSONSchema:
            return None

        # Decoder that uses the detached decode function generation logic
        def object_decoder(message_data: bytes) -> Any:
            if schema.id not in self._decoders:
                # Use the detached decode function generator
                decode_fn = self._decode_cache.get_decode_function(schema.name)
                if decode_fn is not None:
                    self._decoders[schema.id] = decode_fn
                else:
                    # This should not happen as generate_decode_function always returns something
                    # but we handle it gracefully just in case
                    raise ValueError(f"Could not generate decode function for schema '{schema.name}'")

            return self._decoders[schema.id](message_data)

        return object_decoder
