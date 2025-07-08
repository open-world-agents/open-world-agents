"""
OWA Dataset utilities for encoder creation.

This module provides utility functions for creating event encoders
used by dataset transforms.
"""

from owa.data.encoders import BaseEventEncoder


def create_encoder(encoder_type: str) -> BaseEventEncoder:
    """Create an encoder instance based on the specified type."""
    from owa.data.encoders import HierarchicalEventEncoder, JSONEventEncoder

    encoder_type = encoder_type.lower()

    if encoder_type == "hierarchical":
        return HierarchicalEventEncoder()
    elif encoder_type == "json":
        return JSONEventEncoder()
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}.")
