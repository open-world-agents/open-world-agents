from typing import Optional

from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.records import Schema
from mcap.well_known import MessageEncoding, SchemaEncoding

from .decode_utils import dict_decoder, get_decode_function


class DecoderFactory(McapDecoderFactory):
    def __init__(self, *, decode_args: dict = {}):
        """Initialize the decoder factory.

        :param decode_args: Dictionary of decode arguments (return_dict, return_dict_on_failure)
        """
        self.decode_args = {"return_dict": False, "return_dict_on_failure": False, **decode_args}

    def decoder_for(self, message_encoding: str, schema: Optional[Schema]):
        if message_encoding != MessageEncoding.JSON or schema is None or schema.encoding != SchemaEncoding.JSONSchema:
            return None

        return get_decode_function(schema.name, **self.decode_args)


__all__ = ["DecoderFactory", "dict_decoder", "get_decode_function"]
