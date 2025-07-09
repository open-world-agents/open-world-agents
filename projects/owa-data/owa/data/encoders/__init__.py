from .base_encoder import BaseEventEncoder
from .hierarchical_event_encoder import HierarchicalEventEncoder, HierarchicalEventEncoderConfig
from .json_event_encoder import JSONEventEncoder, JSONEventEncoderConfig


def create_encoder(encoder_type: str) -> BaseEventEncoder:
    """Create an encoder instance based on the specified type."""

    encoder_type = encoder_type.lower()

    if encoder_type == "hierarchical":
        return HierarchicalEventEncoder()
    elif encoder_type == "json":
        return JSONEventEncoder()
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}.")


__all__ = [
    "BaseEventEncoder",
    "create_encoder",
    "JSONEventEncoder",
    "HierarchicalEventEncoder",
    "HierarchicalEventEncoderConfig",
    "JSONEventEncoderConfig",
]
