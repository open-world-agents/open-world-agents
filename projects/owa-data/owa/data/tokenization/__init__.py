"""
Event tokenization module for OWA data pipeline.

Clean separation of concerns:
- ImageTokenConfig, EventTokenizationContext: Immutable data containers (frozen dataclass)
- expand_tokenizer_for_events, prepare_model_for_events: Side-effect functions (call once)
- tokenize_event, decode_episode, ...: Pure functions (no side effects)
"""

from .config import ImageTokenConfig, get_image_config
from .context import EventTokenizationContext
from .functions import (
    TokenizedEvent,
    decode_episode,
    decode_event,
    tokenize_episode,
    tokenize_event,
    tokenize_event_dataset,
)
from .preparation import expand_tokenizer_for_events, prepare_model_for_events

__all__ = [
    # Context
    "EventTokenizationContext",
    # Config
    "ImageTokenConfig",
    # Types
    "TokenizedEvent",
    "decode_episode",
    "decode_event",
    # Preparation (side-effect functions)
    "expand_tokenizer_for_events",
    "get_image_config",
    "prepare_model_for_events",
    "tokenize_episode",
    # Tokenization (pure functions)
    "tokenize_event",
    "tokenize_event_dataset",
]
