"""
Preparation functions for event tokenization.

These functions have side effects (mutating tokenizer/model) and should be
called once during setup, before creating the EventTokenizationContext.
"""

from loguru import logger
from transformers.tokenization_utils import PreTrainedTokenizer

from owa.data.encoders.base_encoder import BaseEventEncoder

from .config import ImageTokenConfig


def expand_tokenizer_for_events(
    tokenizer: PreTrainedTokenizer,
    encoder: BaseEventEncoder,
    image_config: ImageTokenConfig,
    *,
    skip_if_exists: bool = True,
) -> int:
    """Expand tokenizer vocabulary with event tokens. Mutates tokenizer in-place."""
    vocab = encoder.get_vocab() | image_config.vocab_tokens
    existing_vocab = tokenizer.get_vocab()

    if skip_if_exists and all(tok in existing_vocab for tok in vocab):
        logger.info("Tokenizer already has all event tokens, skipping expansion")
        return 0

    # Sort for deterministic ordering (set is unordered)
    num_added = tokenizer.add_tokens(sorted(vocab))
    logger.info(f"Added {num_added} event tokens to tokenizer")
    return num_added


def prepare_model_for_events(
    tokenizer: PreTrainedTokenizer,
    encoder: BaseEventEncoder,
    image_config: ImageTokenConfig,
    model=None,
    *,
    apply_semantic_init: bool = True,
) -> None:
    """Prepare tokenizer and optionally model for event tokenization.

    Expands tokenizer vocab, resizes model embeddings, and applies semantic init.
    """
    from owa.data.semantic_init import apply_semantic_initialization

    num_added = expand_tokenizer_for_events(tokenizer, encoder, image_config)

    if model is not None and num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized model embeddings to {len(tokenizer)} tokens")

        if apply_semantic_init:
            # Determine encoder type from encoder class name
            encoder_type = type(encoder).__name__.lower()
            if "factorized" in encoder_type:
                encoder_type = "factorized"
            elif "hierarchical" in encoder_type:
                encoder_type = "hierarchical"
            else:
                encoder_type = "factorized"  # default

            apply_semantic_initialization(tokenizer, model, encoder_type)
            logger.info(f"Applied semantic initialization for {encoder_type} encoder")
