#!/usr/bin/env python3
"""
Unified test suite for both HierarchicalEventEncoder and FactorizedEventEncoder.

This test suite runs the same core tests on both encoders to ensure they both
work correctly and maintain compatibility.
"""

import orjson
import pytest

from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.data.encoders.exceptions import InvalidInputError, InvalidTokenError
from owa.data.encoders.factorized_event_encoder import FactorizedEventEncoder
from owa.data.encoders.hierarchical_event_encoder import HierarchicalEventEncoder
from owa.msgs.desktop.screen import ScreenCaptured


@pytest.fixture(params=["hierarchical", "factorized"])
def encoder(request):
    """Parametrized fixture that provides both encoder types."""
    if request.param == "hierarchical":
        return HierarchicalEventEncoder()
    elif request.param == "factorized":
        return FactorizedEventEncoder()
    else:
        raise ValueError(f"Unknown encoder type: {request.param}")


@pytest.fixture
def encoder_type(encoder):
    """Get the encoder type name for the current test."""
    if isinstance(encoder, HierarchicalEventEncoder):
        return "hierarchical"
    elif isinstance(encoder, FactorizedEventEncoder):
        return "factorized"
    else:
        return "unknown"


class TestBothEncoders:
    """Test both encoders with the same test cases."""

    def test_basic_mouse_fidelity(self, encoder, encoder_type, subtests):
        """Basic mouse events should round-trip without data loss on both encoders."""
        test_cases = [
            # (movement_x, movement_y, button_flags, button_data, description)
            (0, 0, 0, 0, "no movement, no buttons"),
            (1, 1, 0, 0, "minimal movement"),
            (7, -3, 1, 0, "small movement, left button"),
            (-13, 9, 2, 0, "small negative movement, left button up"),
            (47, -29, 4, 0, "medium movement, right button down"),
            (-53, 31, 8, 0, "medium negative movement, right button up"),
        ]

        for dx, dy, flags, data, desc in test_cases:
            with subtests.test(encoder=encoder_type, dx=dx, dy=dy, flags=flags, data=data, desc=desc):
                original = {"last_x": dx, "last_y": dy, "button_flags": flags, "button_data": data}
                msg = McapMessage(
                    topic="mouse/raw",
                    timestamp=1000000000,
                    message=orjson.dumps(original),
                    message_type="desktop/RawMouseEvent",
                )

                # Round trip
                encoded, images = encoder.encode(msg)
                decoded = encoder.decode(encoded, images)
                result = orjson.loads(decoded.message)

                # Check fidelity
                assert result["last_x"] == dx, (
                    f"[{encoder_type}] X movement not preserved in {desc}: expected {dx}, got {result['last_x']}"
                )
                assert result["last_y"] == dy, (
                    f"[{encoder_type}] Y movement not preserved in {desc}: expected {dy}, got {result['last_y']}"
                )
                assert result["button_flags"] == flags, f"[{encoder_type}] Button flags lost in {desc}"
                assert result["button_data"] == data, f"[{encoder_type}] Button data lost in {desc}"

    def test_basic_keyboard_fidelity(self, encoder, encoder_type, subtests):
        """Basic keyboard events should round-trip without data loss on both encoders."""
        test_cases = [
            ("press", 65),  # A key
            ("release", 65),  # A key
            ("press", 13),  # Enter
            ("release", 13),  # Enter
            ("press", 32),  # Space
            ("release", 32),  # Space
        ]

        for event_type, vk in test_cases:
            with subtests.test(encoder=encoder_type, event_type=event_type, vk=vk):
                original = {"event_type": event_type, "vk": vk}
                msg = McapMessage(
                    topic="keyboard",
                    timestamp=2000000000,
                    message=orjson.dumps(original),
                    message_type="desktop/KeyboardEvent",
                )

                # Round trip
                encoded, images = encoder.encode(msg)
                decoded = encoder.decode(encoded, images)
                result = orjson.loads(decoded.message)

                # Check perfect fidelity
                assert result["event_type"] == event_type, f"[{encoder_type}] Event type lost for {event_type} {vk}"
                assert result["vk"] == vk, f"[{encoder_type}] VK code lost for {event_type} {vk}"

    def test_screen_fidelity(self, encoder, encoder_type):
        """Screen events should preserve structure on both encoders."""
        original = {
            "utc_ns": 3000000000,
            "source_shape": [1920, 1080],
            "shape": [1920, 1080],
            "media_ref": {"uri": "test.png"},
        }

        msg = McapMessage(
            topic="screen",
            timestamp=3000000000,
            message=orjson.dumps(original),
            message_type="desktop/ScreenCaptured",
        )

        # Encode and check image object preservation
        encoded, images = encoder.encode(msg)
        assert len(images) == 1, f"[{encoder_type}] Expected 1 image, got {len(images)}"
        assert isinstance(images[0], ScreenCaptured), f"[{encoder_type}] Expected ScreenCaptured object"
        assert images[0].utc_ns == 3000000000, f"[{encoder_type}] utc_ns not preserved"

        # Decode and check message timestamp preservation
        decoded = encoder.decode(encoded, images)
        assert decoded.timestamp == 3000000000, f"[{encoder_type}] timestamp not preserved"
        assert decoded.topic == "screen", f"[{encoder_type}] topic not preserved"

    def test_zero_values(self, encoder, encoder_type):
        """Zero values should be handled correctly by both encoders."""
        data = {"last_x": 0, "last_y": 0, "button_flags": 0, "button_data": 0}
        msg = McapMessage(
            topic="mouse/raw",
            timestamp=0,
            message=orjson.dumps(data),
            message_type="desktop/RawMouseEvent",
        )

        encoded, images = encoder.encode(msg)
        decoded = encoder.decode(encoded, images)
        result = orjson.loads(decoded.message)

        assert result["last_x"] == 0, f"[{encoder_type}] Zero X not preserved"
        assert result["last_y"] == 0, f"[{encoder_type}] Zero Y not preserved"
        assert result["button_flags"] == 0, f"[{encoder_type}] Zero button_flags not preserved"
        assert result["button_data"] == 0, f"[{encoder_type}] Zero button_data not preserved"

    def test_basic_functionality(self, encoder, encoder_type):
        """Basic encoder functionality should work for both encoders."""
        # Can create vocab
        vocab = encoder.get_vocab()
        assert len(vocab) > 0, f"[{encoder_type}] Empty vocabulary"

        # Contains expected common tokens
        assert "<EVENT_START>" in vocab, f"[{encoder_type}] Missing <EVENT_START>"
        assert "<EVENT_END>" in vocab, f"[{encoder_type}] Missing <EVENT_END>"
        assert "<MOUSE>" in vocab, f"[{encoder_type}] Missing <MOUSE>"
        assert "<KEYBOARD>" in vocab, f"[{encoder_type}] Missing <KEYBOARD>"

    def test_token_efficiency(self, encoder, encoder_type):
        """Both encoders should produce reasonable token counts."""
        # Test mouse event
        data = {"last_x": 10, "last_y": -5, "button_flags": 1, "button_data": 0}
        msg = McapMessage(
            topic="mouse/raw",
            timestamp=1000000000,
            message=orjson.dumps(data),
            message_type="desktop/RawMouseEvent",
        )

        encoded, _ = encoder.encode(msg)
        token_count = encoded.count("<")

        # Both encoders should be reasonably efficient
        assert token_count <= 30, f"[{encoder_type}] Too many tokens for mouse event: {token_count}"
        assert token_count >= 5, f"[{encoder_type}] Too few tokens for mouse event: {token_count}"

        # Test keyboard event
        data = {"event_type": "press", "vk": 65}
        msg = McapMessage(
            topic="keyboard",
            timestamp=2000000000,
            message=orjson.dumps(data),
            message_type="desktop/KeyboardEvent",
        )

        encoded, _ = encoder.encode(msg)
        token_count = encoded.count("<")

        assert token_count <= 20, f"[{encoder_type}] Too many tokens for keyboard event: {token_count}"
        assert token_count >= 5, f"[{encoder_type}] Too few tokens for keyboard event: {token_count}"

    def test_encoder_specific_tokens(self, encoder, encoder_type):
        """Test that each encoder uses its specific token vocabulary."""
        # Test keyboard event to see encoder-specific tokens
        original = {"event_type": "press", "vk": 65}  # A key
        msg = McapMessage(
            topic="keyboard",
            timestamp=2000000000,
            message=orjson.dumps(original),
            message_type="desktop/KeyboardEvent",
        )

        encoded, images = encoder.encode(msg)

        if encoder_type == "factorized":
            # Factorized should use VK tokens
            assert "<VK_65>" in encoded, f"Factorized encoder should use <VK_65> token: {encoded}"
        elif encoder_type == "hierarchical":
            # Hierarchical should use numeric tokens
            assert "<65>" in encoded, f"Hierarchical encoder should use <65> token: {encoded}"

        # Test mouse event with negative movement
        original = {"last_x": -5, "last_y": 3, "button_flags": 0, "button_data": 0}
        msg = McapMessage(
            topic="mouse/raw",
            timestamp=1000000000,
            message=orjson.dumps(original),
            message_type="desktop/RawMouseEvent",
        )

        encoded, images = encoder.encode(msg)

        if encoder_type == "factorized":
            # Factorized should use SIGN tokens
            assert "<SIGN_MINUS>" in encoded, f"Factorized encoder should use <SIGN_MINUS> token: {encoded}"
            assert "<SIGN_PLUS>" in encoded, f"Factorized encoder should use <SIGN_PLUS> token: {encoded}"

    def test_mouse_validation(self, encoder, encoder_type, subtests):
        """Both encoders should validate input ranges and warn for invalid delta values."""
        min_delta, max_delta = encoder.config.mouse_delta_range

        # Test boundary values (should work)
        valid_cases = [
            (max_delta, max_delta),  # At positive boundary
            (min_delta, min_delta),  # At negative boundary
            (0, 0),  # Zero
        ]

        for dx, dy in valid_cases:
            with subtests.test(encoder=encoder_type, case="valid", dx=dx, dy=dy):
                data = {"last_x": dx, "last_y": dy, "button_flags": 0, "button_data": 0}
                msg = McapMessage(
                    topic="mouse/raw",
                    timestamp=1000000000,
                    message=orjson.dumps(data),
                    message_type="desktop/RawMouseEvent",
                )

                # Should work without errors
                encoded, images = encoder.encode(msg)
                decoded = encoder.decode(encoded, images)
                result = orjson.loads(decoded.message)
                assert result["last_x"] == dx, f"[{encoder_type}] X value not preserved"
                assert result["last_y"] == dy, f"[{encoder_type}] Y value not preserved"

        # Test invalid values (should issue warnings and clamp values)
        invalid_cases = [
            (max_delta + 1, 0),  # X too large
            (min_delta - 1, 0),  # X too small
        ]

        for dx, dy in invalid_cases:
            with subtests.test(encoder=encoder_type, case="invalid", dx=dx, dy=dy):
                data = {"last_x": dx, "last_y": dy, "button_flags": 0, "button_data": 0}
                msg = McapMessage(
                    topic="mouse/raw",
                    timestamp=1000000000,
                    message=orjson.dumps(data),
                    message_type="desktop/RawMouseEvent",
                )

                # Should issue warning and work (with clamping)
                with pytest.warns(UserWarning, match=r"Mouse delta value .* is outside valid range"):
                    encoded, images = encoder.encode(msg)
                    decoded = encoder.decode(encoded, images)
                    result = orjson.loads(decoded.message)
                    # Values should be clamped to valid range
                    assert min_delta <= result["last_x"] <= max_delta, f"[{encoder_type}] X not clamped properly"
                    assert min_delta <= result["last_y"] <= max_delta, f"[{encoder_type}] Y not clamped properly"

    def test_invalid_token_errors(self, encoder, encoder_type):
        """Both encoders should handle invalid token formats appropriately."""
        # Missing EVENT_START token
        with pytest.raises(InvalidTokenError, match="Missing EVENT_START or EVENT_END tokens"):
            encoder.decode("<KEYBOARD><0><0><0><press><65><EVENT_END>")

        # Missing EVENT_END token
        with pytest.raises(InvalidTokenError, match="Missing EVENT_START or EVENT_END tokens"):
            encoder.decode("<EVENT_START><KEYBOARD><0><0><0><press><65>")

        # Token sequence too short
        with pytest.raises(InvalidTokenError, match="Token sequence too short"):
            encoder.decode("<EVENT_START><KEYBOARD><EVENT_END>")

    def test_unsupported_input_errors(self, encoder, encoder_type):
        """Both encoders should handle unsupported message inputs appropriately."""
        # Unsupported topic
        msg = McapMessage(
            topic="unsupported_topic",
            timestamp=1000000000,
            message_type="test/Message",
            message=b'{"test": "data"}',
        )
        with pytest.raises(InvalidInputError, match="Failed to decode message"):
            encoder.encode(msg)

    def test_extreme_timestamps(self, encoder, encoder_type, subtests):
        """Both encoders should handle extreme timestamp values gracefully."""
        extreme_timestamps = [
            0,  # Minimum
            9223372036854775807,  # Maximum int64
            1000000000000000000,  # Very large
        ]

        for ts in extreme_timestamps:
            with subtests.test(encoder=encoder_type, timestamp=ts):
                data = {"last_x": 10, "last_y": 10, "button_flags": 0, "button_data": 0}
                msg = McapMessage(
                    topic="mouse/raw",
                    timestamp=ts,
                    message=orjson.dumps(data),
                    message_type="desktop/RawMouseEvent",
                )

                # Should handle gracefully (may quantize large timestamps)
                encoded, images = encoder.encode(msg)
                decoded = encoder.decode(encoded, images)

                # Should produce valid timestamp
                assert isinstance(decoded.timestamp, int), f"[{encoder_type}] Invalid timestamp type"
                assert decoded.timestamp >= 0, f"[{encoder_type}] Negative timestamp"

    def test_extreme_keyboard_values(self, encoder, encoder_type, subtests):
        """Both encoders should handle extreme keyboard values gracefully."""
        extreme_cases = [
            ("press", 0),  # Minimum VK
            ("release", 255),  # Maximum standard VK
            ("press", 65535),  # Very large VK
        ]

        for event_type, vk in extreme_cases:
            with subtests.test(encoder=encoder_type, event_type=event_type, vk=vk):
                data = {"event_type": event_type, "vk": vk}
                msg = McapMessage(
                    topic="keyboard",
                    timestamp=4000000000 + vk,
                    message=orjson.dumps(data),
                    message_type="desktop/KeyboardEvent",
                )

                # Should handle gracefully
                encoded, images = encoder.encode(msg)
                decoded = encoder.decode(encoded, images)
                result = orjson.loads(decoded.message)

                # Should produce valid results
                assert result["event_type"] == event_type, f"[{encoder_type}] Event type not preserved"
                assert isinstance(result["vk"], int), f"[{encoder_type}] VK not integer"


class TestEncoderComparison:
    """Compare the two encoders directly."""

    def test_vocab_size_comparison(self):
        """Compare vocabulary sizes between encoders."""
        h_encoder = HierarchicalEventEncoder()
        f_encoder = FactorizedEventEncoder()

        h_vocab = h_encoder.get_vocab()
        f_vocab = f_encoder.get_vocab()

        print(f"Hierarchical vocab size: {len(h_vocab)}")
        print(f"Factorized vocab size: {len(f_vocab)}")

        # Both should have reasonable vocab sizes
        assert len(h_vocab) > 100, "Hierarchical vocab too small"
        assert len(f_vocab) > 100, "Factorized vocab too small"
        assert len(h_vocab) < 1000, "Hierarchical vocab too large"
        assert len(f_vocab) < 1000, "Factorized vocab too large"

    def test_encoding_differences(self):
        """Test that the two encoders produce different but valid encodings."""
        original = {"event_type": "press", "vk": 65}
        msg = McapMessage(
            topic="keyboard",
            timestamp=2000000000,
            message=orjson.dumps(original),
            message_type="desktop/KeyboardEvent",
        )

        h_encoder = HierarchicalEventEncoder()
        f_encoder = FactorizedEventEncoder()

        h_encoded, h_images = h_encoder.encode(msg)
        f_encoded, f_images = f_encoder.encode(msg)

        # Encodings should be different (different token vocabularies)
        assert h_encoded != f_encoded, "Encoders should produce different token sequences"

        # But both should decode to the same result
        h_decoded = h_encoder.decode(h_encoded, h_images)
        f_decoded = f_encoder.decode(f_encoded, f_images)

        h_result = orjson.loads(h_decoded.message)
        f_result = orjson.loads(f_decoded.message)

        assert h_result == f_result, "Both encoders should decode to the same result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
