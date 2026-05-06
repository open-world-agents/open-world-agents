#!/usr/bin/env python3
"""
Test suite for the tokenization module.

Tests the refactored tokenization module with:
- ImageTokenConfig (frozen dataclass)
- EventTokenizationContext (frozen dataclass)
- Preparation functions (side-effect functions)
- Tokenization functions (pure functions)
"""

from dataclasses import FrozenInstanceError

import numpy as np
import orjson
import pytest
from transformers import AutoTokenizer

from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.data.encoders import create_encoder
from owa.data.tokenization import (
    EventTokenizationContext,
    ImageTokenConfig,
    decode_episode,
    expand_tokenizer_for_events,
    tokenize_episode,
    tokenize_event,
)


@pytest.fixture
def image_config():
    """Create a test ImageTokenConfig."""
    return ImageTokenConfig(
        prefix="<img>",
        token="<IMG>",
        length=4,  # Short for testing
        suffix="</img>",
    )


@pytest.fixture
def encoder():
    """Create a FactorizedEventEncoder for testing."""
    return create_encoder("factorized")


@pytest.fixture
def tokenizer():
    """Create a minimal tokenizer for testing."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def prepared_tokenizer(tokenizer, encoder, image_config):
    """Create a tokenizer with event vocab already added."""
    expand_tokenizer_for_events(tokenizer, encoder, image_config)
    return tokenizer


@pytest.fixture
def ctx(encoder, prepared_tokenizer, image_config):
    """Create an EventTokenizationContext for testing."""
    return EventTokenizationContext(
        encoder=encoder,
        tokenizer=prepared_tokenizer,
        image_config=image_config,
    )


class TestImageTokenConfig:
    """Tests for ImageTokenConfig frozen dataclass."""

    def test_creation(self, image_config):
        """Test basic creation."""
        assert image_config.prefix == "<img>"
        assert image_config.token == "<IMG>"
        assert image_config.length == 4
        assert image_config.suffix == "</img>"

    def test_pattern_property(self, image_config):
        """Test pattern generation."""
        expected = "<img><IMG><IMG><IMG><IMG></img>"
        assert image_config.pattern == expected

    def test_vocab_tokens_property(self, image_config):
        """Test vocab_tokens returns correct set."""
        expected = {"<img>", "<IMG>", "</img>"}
        assert image_config.vocab_tokens == expected

    def test_frozen(self, image_config):
        """Test that config is immutable."""
        with pytest.raises(FrozenInstanceError):
            image_config.prefix = "<new>"

    def test_default_fake_placeholder(self, image_config):
        """Test default fake_placeholder value."""
        assert image_config.fake_placeholder == "<fake_image_placeholder>"


class TestEventTokenizationContext:
    """Tests for EventTokenizationContext frozen dataclass."""

    def test_creation(self, ctx):
        """Test basic creation."""
        assert ctx.encoder is not None
        assert ctx.tokenizer is not None
        assert ctx.image_config is not None

    def test_frozen(self, ctx):
        """Test that context is immutable."""
        with pytest.raises(FrozenInstanceError):
            ctx.encoder = None

    def test_encoder_type_property(self, ctx):
        """Test encoder_type property."""
        assert "Factorized" in ctx.encoder_type

    def test_create_factory(self, prepared_tokenizer, image_config):
        """Test create factory method."""
        ctx = EventTokenizationContext.create(
            encoder_type="factorized",
            tokenizer=prepared_tokenizer,
            image_config=image_config,
        )
        assert ctx.encoder is not None
        assert "Factorized" in ctx.encoder_type

    def test_missing_tokens_raises_error(self, tokenizer, encoder, image_config):
        """Test that creating context without expanding tokenizer raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            EventTokenizationContext(encoder=encoder, tokenizer=tokenizer, image_config=image_config)
        assert "missing" in str(exc_info.value).lower()
        assert "expand_tokenizer_for_events" in str(exc_info.value)


class TestPreparationFunctions:
    """Tests for preparation functions."""

    def test_expand_tokenizer_for_events(self, tokenizer, encoder, image_config):
        """Test tokenizer expansion."""
        original_size = len(tokenizer)
        num_added = expand_tokenizer_for_events(tokenizer, encoder, image_config)

        assert num_added > 0
        assert len(tokenizer) == original_size + num_added

        # Verify tokens are in vocab
        vocab = tokenizer.get_vocab()
        assert "<EVENT_START>" in vocab
        assert "<img>" in vocab

    def test_expand_tokenizer_skip_if_exists(self, prepared_tokenizer, encoder, image_config):
        """Test that expansion is skipped if tokens already exist."""
        num_added = expand_tokenizer_for_events(prepared_tokenizer, encoder, image_config, skip_if_exists=True)
        assert num_added == 0


class TestTokenizationFunctions:
    """Tests for pure tokenization functions."""

    def test_tokenize_event_mouse(self, ctx):
        """Test tokenizing a mouse event."""
        data = {"last_x": 10, "last_y": -5, "button_flags": 1, "button_data": 0}
        msg = McapMessage(
            topic="mouse/raw",
            timestamp=1000000000,
            message=orjson.dumps(data),
            message_type="desktop/RawMouseEvent",
        )

        result = tokenize_event(ctx, msg)

        assert isinstance(result, dict)
        assert "text" in result
        assert "images" in result
        assert "token_ids" in result
        assert "total_token_count" in result
        assert "<EVENT_START>" in result["text"]
        assert "<MOUSE>" in result["text"]
        assert len(result["token_ids"]) > 0

    def test_tokenize_event_keyboard(self, ctx):
        """Test tokenizing a keyboard event."""
        data = {"event_type": "press", "vk": 65}
        msg = McapMessage(
            topic="keyboard",
            timestamp=2000000000,
            message=orjson.dumps(data),
            message_type="desktop/KeyboardEvent",
        )

        result = tokenize_event(ctx, msg)

        assert "<KEYBOARD>" in result["text"]
        assert len(result["token_ids"]) > 0

    def test_tokenize_event_return_array(self, ctx):
        """Test tokenize_event with return_dict=False."""
        data = {"event_type": "press", "vk": 65}
        msg = McapMessage(
            topic="keyboard",
            timestamp=2000000000,
            message=orjson.dumps(data),
            message_type="desktop/KeyboardEvent",
        )

        result = tokenize_event(ctx, msg, return_dict=False)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_tokenize_episode(self, ctx):
        """Test tokenizing multiple events."""
        messages = [
            McapMessage(
                topic="keyboard",
                timestamp=1000000000 + i * 100000000,
                message=orjson.dumps({"event_type": "press", "vk": 65 + i}),
                message_type="desktop/KeyboardEvent",
            )
            for i in range(3)
        ]

        results = list(tokenize_episode(ctx, iter(messages)))

        assert len(results) == 3
        for result in results:
            assert "<KEYBOARD>" in result["text"]

    def test_decode_episode(self, ctx):
        """Test decoding tokenized events back to messages."""
        # Create and tokenize events
        original_messages = [
            McapMessage(
                topic="keyboard",
                timestamp=1000000000,
                message=orjson.dumps({"event_type": "press", "vk": 65}),
                message_type="desktop/KeyboardEvent",
            ),
            McapMessage(
                topic="keyboard",
                timestamp=2000000000,
                message=orjson.dumps({"event_type": "release", "vk": 65}),
                message_type="desktop/KeyboardEvent",
            ),
        ]

        # Tokenize to text
        tokenized = [tokenize_event(ctx, msg) for msg in original_messages]
        combined_text = "".join(t["text"] for t in tokenized)

        # Decode back
        decoded = list(decode_episode(ctx, combined_text, adjust_timestamp=False))

        assert len(decoded) == 2
        for decoded_msg in decoded:
            assert decoded_msg.topic == "keyboard"
