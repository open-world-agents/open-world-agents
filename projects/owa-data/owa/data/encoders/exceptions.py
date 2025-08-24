"""Exception classes for event encoders."""


class EventEncodingError(Exception):
    """Base exception for event encoding/decoding failures."""

    pass


# ENCODING EXCEPTIONS
class UnsupportedInputError(EventEncodingError):
    """Raised when input is valid but encoder does not support it."""

    pass


class InvalidInputError(EventEncodingError):
    """Raised when input is invalid."""

    pass


# DECODING EXCEPTIONS
class UnsupportedTokenError(EventEncodingError):
    """Raised when token is valid but decoder does not support it."""

    pass


class InvalidTokenError(EventEncodingError):
    """Raised when token is invalid."""

    pass
