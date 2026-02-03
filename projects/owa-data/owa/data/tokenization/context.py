"""
Immutable context container for event tokenization.
"""

from dataclasses import dataclass

from transformers.tokenization_utils import PreTrainedTokenizer

from owa.data.encoders import create_encoder
from owa.data.encoders.base_encoder import BaseEventEncoder

from .config import ImageTokenConfig


@dataclass(frozen=True)
class EventTokenizationContext:
    """Immutable container for event tokenization dependencies.

    Design: frozen=True ensures thread-safety and prevents accidental state changes.
    The context bundles encoder, tokenizer, and image_config for pure tokenization functions.

    Safety: Validates that tokenizer has required event tokens on creation.
    Raises ValueError if expand_tokenizer_for_events() was not called first.

    Example:
        expand_tokenizer_for_events(tokenizer, encoder, image_config)
        ctx = EventTokenizationContext(encoder, tokenizer, image_config)
        result = tokenize_event(ctx, mcap_msg)
    """

    encoder: BaseEventEncoder
    tokenizer: PreTrainedTokenizer
    image_config: ImageTokenConfig

    def __post_init__(self):
        # Validate tokenizer has all required event tokens
        required = self.encoder.get_vocab() | self.image_config.vocab_tokens
        existing = set(self.tokenizer.get_vocab().keys())
        missing = required - existing
        if missing:
            examples = list(missing)[:3]
            raise ValueError(
                f"Tokenizer missing {len(missing)} event tokens (e.g., {examples}). "
                f"Call expand_tokenizer_for_events(tokenizer, encoder, image_config) first."
            )

    @classmethod
    def create(
        cls,
        *,
        encoder_type: str,
        tokenizer: PreTrainedTokenizer,
        image_config: ImageTokenConfig,
    ) -> "EventTokenizationContext":
        """Create context with a new encoder instance (convenience factory)."""
        encoder = create_encoder(
            encoder_type,
            fake_image_placeholder=image_config.fake_placeholder,
        )
        return cls(encoder=encoder, tokenizer=tokenizer, image_config=image_config)

    @property
    def encoder_type(self) -> str:
        return type(self.encoder).__name__
