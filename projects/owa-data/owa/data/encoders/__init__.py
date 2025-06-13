"""
Event Encoders for OWA Data Pipeline.

This module provides various event encoding strategies for converting raw OWA events
into formats suitable for different types of VLA (Vision-Language-Action) model training.

Available Encoders:
- JSONEventEncoder: JSON string-based encoder with <EVENT_START>/<EVENT_END> tokens
- FlatEventEncoder: Flat token-based encoder inspired by example.py
- HierarchicalEventEncoder: Hierarchical compositional token encoder

Each encoder follows a common interface for encode/decode operations while
implementing different tokenization strategies optimized for specific use cases.
"""

from .base_encoder import BaseEventEncoder
from .flat_event_encoder import FlatEventEncoder, FlatEventEncoderConfig
from .hierarchical_event_encoder import HierarchicalEventEncoder, HierarchicalEventEncoderConfig
from .json_event_encoder import JSONEventEncoder

__all__ = [
    "BaseEventEncoder",
    "JSONEventEncoder",
    "FlatEventEncoder",
    "FlatEventEncoderConfig",
    "HierarchicalEventEncoder",
    "HierarchicalEventEncoderConfig",
]
