from typing import Optional

from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.records import Schema
from mcap.well_known import MessageEncoding, SchemaEncoding

from .decode_utils import dict_decoder, get_decode_function


class DecoderFactory(McapDecoderFactory):
    def __init__(self, *, decode_as_dict: bool = False):
        """Initialize the decoder factory."""
        self._decode_as_dict = decode_as_dict

    def decoder_for(self, message_encoding: str, schema: Optional[Schema]):
        if message_encoding != MessageEncoding.JSON or schema is None or schema.encoding != SchemaEncoding.JSONSchema:
            return None

        if not self._decode_as_dict:
            decode_fn = get_decode_function(schema.name)
        else:
            decode_fn = dict_decoder

        if decode_fn is None:
            # This should not happen as _create_message_decoder always returns something
            # but we handle it gracefully just in case
            raise ValueError(f"Could not generate decode function for schema '{schema.name}'")

        return decode_fn
