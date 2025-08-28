"""Exception classes for event encoders."""


class EventEncoderError(Exception):
    """Base exception for event encoding/decoding failures."""

    pass


# ENCODING EXCEPTIONS
class UnsupportedInputError(EventEncoderError):
    """Raised when input is valid but encoder does not support it."""

    pass


class InvalidInputError(EventEncoderError):
    """Raised when input is invalid."""

    pass


# DECODING EXCEPTIONS
class UnsupportedTokenError(EventEncoderError):
    """Raised when token is valid but decoder does not support it."""

    pass


class InvalidTokenError(EventEncoderError):
    """Raised when token is invalid."""

    pass
