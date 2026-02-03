"""
Event tokenization module for OWA data pipeline.

Clean separation of concerns:
- ImageTokenConfig, EventTokenizationContext: Immutable data containers (frozen dataclass)
- expand_tokenizer_for_events, prepare_model_for_events: Side-effect functions (call once)
- tokenize_event, decode_episode, ...: Pure functions (no side effects)
"""

from .config import ImageTokenConfig
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
    # Config
    "ImageTokenConfig",
    # Context
    "EventTokenizationContext",
    # Types
    "TokenizedEvent",
    # Preparation (side-effect functions)
    "expand_tokenizer_for_events",
    "prepare_model_for_events",
    # Tokenization (pure functions)
    "tokenize_event",
    "decode_event",
    "tokenize_episode",
    "decode_episode",
    "tokenize_event_dataset",
]
