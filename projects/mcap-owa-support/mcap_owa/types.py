"""Type definitions for mcap_owa package."""

from typing import TypedDict


class DecodeArgs(TypedDict, total=False):
    """
    Type definition for decode_args parameter used throughout mcap_owa.

    Attributes:
        return_dict: If True, always decode messages as dictionaries (EasyDict objects)
                    instead of typed OWA message objects. Use this when you want simple
                    dictionary access to message fields.
        return_dict_on_failure: If True, fall back to dictionary decoding when typed
                               decoding fails. Use this for robust reading of files that
                               may contain unknown or malformed message types.
    """

    return_dict: bool
    return_dict_on_failure: bool
