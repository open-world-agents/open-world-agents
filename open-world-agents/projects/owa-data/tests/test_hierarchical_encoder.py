#!/usr/bin/env python3
"""
Test for hierarchical event encoder - focuses on essential functionality and configuration sanity.
"""

import json

import pytest

from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.data.encoders.hierarchical_event_encoder import HierarchicalEventEncoder, HierarchicalEventEncoderConfig


class TestHierarchicalEncoderConfig:
    """Test configuration sanity and unified semantics."""

    def test_default_config(self):
        """Test default configuration has correct unified semantics."""
        config = HierarchicalEventEncoderConfig()

        # Both should represent half-ranges
        assert config.max_timestamp_range_ns == 8_000_000_000  # ±8 seconds
        assert config.max_mouse_delta == (1000, 1000)  # ±1000 x, ±1000 y

        # Old attribute should not exist
        assert not hasattr(config, "timestamp_range_ns")
        assert not hasattr(config, "max_mouse_delta_x")
        assert not hasattr(config, "max_mouse_delta_y")

    def test_custom_config(self):
        """Test custom configuration with different half-ranges."""
        config = HierarchicalEventEncoderConfig(
            max_timestamp_range_ns=3_000_000_000,  # ±3 seconds
            max_mouse_delta=(500, 800),  # ±500 x, ±800 y
        )

        assert config.max_timestamp_range_ns == 3_000_000_000
        assert config.max_mouse_delta == (500, 800)

    def test_encoder_creation(self):
        """Test encoder can be created with various configurations."""
        # Default config
        encoder1 = HierarchicalEventEncoder()
        assert encoder1 is not None

        # Custom config
        config = HierarchicalEventEncoderConfig(max_mouse_delta=(200, 300))
        encoder2 = HierarchicalEventEncoder(config)
        assert encoder2 is not None


class TestHierarchicalEncoderOperation:
    """Test essential encoding/decoding operations."""

    @pytest.fixture
    def encoder(self):
        """Create encoder with default config."""
        return HierarchicalEventEncoder()

    def test_mouse_encoding_roundtrip(self, encoder):
        """Test mouse event encoding/decoding preserves data."""
        # Create test mouse message
        original_data = {"last_x": 500, "last_y": -750, "button_flags": 1, "button_data": 120}
        mouse_msg = McapMessage(
            topic="mouse/raw",
            timestamp=1_000_000_000,
            message=json.dumps(original_data).encode("utf-8"),
            message_type="desktop/RawMouseEvent",
        )

        # Encode
        encoded, images = encoder.encode(mouse_msg)
        assert encoded.startswith("<EVENT_START>")
        assert encoded.endswith("<EVENT_END>")
        assert "<MOUSE>" in encoded

        # Decode
        decoded_msg = encoder.decode(encoded, images)
        decoded_data = json.loads(decoded_msg.message.decode("utf-8"))

        # Verify preservation of key fields
        assert decoded_data["last_x"] == original_data["last_x"]
        assert decoded_data["last_y"] == original_data["last_y"]
        assert decoded_data["button_flags"] == original_data["button_flags"]

    def test_keyboard_encoding_roundtrip(self, encoder):
        """Test keyboard event encoding/decoding preserves data."""
        # Create test keyboard message
        original_data = {"event_type": "press", "vk": 65}
        kb_msg = McapMessage(
            topic="keyboard",
            timestamp=2_000_000_000,
            message=json.dumps(original_data).encode("utf-8"),
            message_type="desktop/KeyboardEvent",
        )

        # Encode
        encoded, images = encoder.encode(kb_msg)
        assert encoded.startswith("<EVENT_START>")
        assert encoded.endswith("<EVENT_END>")
        assert "<KEYBOARD>" in encoded

        # Decode
        decoded_msg = encoder.decode(encoded, images)
        decoded_data = json.loads(decoded_msg.message.decode("utf-8"))

        # Verify preservation
        assert decoded_data["event_type"] == original_data["event_type"]
        assert decoded_data["vk"] == original_data["vk"]

    def test_boundary_values(self, encoder):
        """Test encoding/decoding at configuration boundaries."""
        config = encoder.config
        max_x, max_y = config.max_mouse_delta

        # Test boundary mouse values
        boundary_data = {"last_x": max_x, "last_y": -max_y, "button_flags": 0, "button_data": 0}
        mouse_msg = McapMessage(
            topic="mouse/raw",
            timestamp=0,
            message=json.dumps(boundary_data).encode("utf-8"),
            message_type="desktop/RawMouseEvent",
        )

        # Should handle boundary values without error
        encoded, images = encoder.encode(mouse_msg)
        decoded_msg = encoder.decode(encoded, images)
        decoded_data = json.loads(decoded_msg.message.decode("utf-8"))

        # Boundary values should be preserved exactly
        assert decoded_data["last_x"] == max_x
        assert decoded_data["last_y"] == -max_y

    def test_vocab_generation(self, encoder):
        """Test vocabulary generation works."""
        vocab = encoder.get_vocab()
        assert isinstance(vocab, set)
        assert len(vocab) > 0

        # Should contain essential tokens
        assert "<EVENT_START>" in vocab
        assert "<EVENT_END>" in vocab
        assert "<TIMESTAMP>" in vocab
        assert "<KEYBOARD>" in vocab
        assert "<MOUSE>" in vocab

    def test_different_mouse_ranges(self):
        """Test encoder works with different mouse ranges."""
        config = HierarchicalEventEncoderConfig(max_mouse_delta=(200, 400))
        encoder = HierarchicalEventEncoder(config)

        # Test with values within the smaller range
        test_data = {"last_x": 150, "last_y": -300, "button_flags": 0, "button_data": 0}
        mouse_msg = McapMessage(
            topic="mouse/raw",
            timestamp=0,
            message=json.dumps(test_data).encode("utf-8"),
            message_type="desktop/RawMouseEvent",
        )

        # Should work without error
        encoded, images = encoder.encode(mouse_msg)
        decoded_msg = encoder.decode(encoded, images)
        decoded_data = json.loads(decoded_msg.message.decode("utf-8"))

        # Values should be preserved
        assert decoded_data["last_x"] == test_data["last_x"]
        assert decoded_data["last_y"] == test_data["last_y"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
