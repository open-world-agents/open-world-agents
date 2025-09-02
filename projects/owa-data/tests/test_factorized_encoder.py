#!/usr/bin/env python3
"""
Test suite for FactorizedEventEncoder.

TESTING PRIORITIES:
==================
1. FIDELITY: Encode-decode yields same/similar result (no data loss)
2. EFFICIENCY: Token count is reasonable (not bloated)
3. NO EDGE CASES: Works well even with edge cases - users don't need to worry

COVERAGE MATRIX:
===============
Event Type    | Encoding | Decoding | Fidelity | Efficiency | Edge Cases | Exhaustive
------------- | -------- | -------- | -------- | ---------- | ---------- | ----------
Mouse Move    |    ✓     |    ✓     |    ✓     |     ✓      |     ✓      |  ✓ (1,681)
Mouse Buttons |    ✓     |    ✓     |    ✓     |     ✓      |     ✓      |     -
Mouse Wheel   |    ✓     |    ✓     |    ✓     |     ✓      |     ✓      |     -
Keyboard      |    ✓     |    ✓     |    ✓     |     ✓      |     ✓      |     -
Screen        |    ✓     |    ✓     |    ✓     |     ✓      |     ✓      |     -
Timestamps    |    ✓     |    ✓     |    ✓     |     -      |     ✓      |   ✓ (160)
"""

import orjson
import pytest

from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.data.encoders.factorized_event_encoder import FactorizedEventEncoder, FactorizedEventEncoderConfig
from owa.msgs.desktop.screen import ScreenCaptured

# =============================================================================
# 1. FIDELITY TESTS: Encode-decode must preserve data
# =============================================================================


class TestFidelity:
    """Test that encoding-decoding preserves data without loss."""

    @pytest.fixture
    def encoder(self):
        return FactorizedEventEncoder()

    def test_mouse_fidelity(self, encoder, subtests):
        """Mouse events should round-trip without data loss."""
        test_cases = [
            # (movement_x, movement_y, button_flags, button_data, description)
            (0, 0, 0, 0, "no movement, no buttons"),
            (1, 1, 0, 0, "minimal movement"),
            (7, -3, 1, 0, "small movement, left button"),
            (-13, 9, 2, 0, "small negative movement, left button up"),
            (47, -29, 4, 0, "medium movement, right button down"),
            (-53, 31, 8, 0, "medium negative movement, right button up"),
            (97, 203, 0x10, 0, "large movement, middle button down"),
            (-103, -197, 0x20, 0, "large negative movement, middle button up"),
            (211, 307, 0x400, 120, "large movement, wheel forward"),
            (-193, -289, 0x400, -120, "large negative movement, wheel backward"),
            (149, -157, 0x401, 240, "movement, left button + wheel forward"),
            (-143, 163, 0x402, -240, "negative movement, left up + wheel backward"),
            (317, 409, 0x404, 360, "movement, right down + wheel forward"),
            (503, 797, 0x410, 480, "large movement, middle down + wheel forward"),
            (251, -247, 0x800, 0, "movement, horizontal wheel"),
            (-259, 241, 0xFFF, -120, "negative movement, all flags + wheel backward"),
        ]

        for dx, dy, flags, data, desc in test_cases:
            with subtests.test(dx=dx, dy=dy, flags=flags, data=data, desc=desc):
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

                # Check fidelity - all test values are within range so should be preserved exactly
                assert result["last_x"] == dx, (
                    f"X movement not preserved in {desc}: expected {dx}, got {result['last_x']}"
                )
                assert result["last_y"] == dy, (
                    f"Y movement not preserved in {desc}: expected {dy}, got {result['last_y']}"
                )
                assert result["button_flags"] == flags, f"Button flags lost in {desc}"

                # Button data is quantized by 120, so check expected value
                expected_data = (data // 120) * 120 if flags & 0x400 else data
                assert result["button_data"] == expected_data, f"Button data lost in {desc}"

    def test_keyboard_fidelity(self, encoder, subtests):
        """Keyboard events should round-trip without data loss."""
        test_cases = [
            ("press", 65),
            ("release", 65),  # A key
            ("press", 90),
            ("release", 90),  # Z key
            ("press", 48),
            ("release", 48),  # 0 key
            ("press", 13),
            ("release", 13),  # Enter
            ("press", 27),
            ("release", 27),  # Escape
            ("press", 32),
            ("release", 32),  # Space
            ("press", 16),
            ("release", 16),  # Shift
            ("press", 112),
            ("release", 112),  # F1
            ("press", 37),
            ("release", 37),  # Left arrow
        ]

        for event_type, vk in test_cases:
            with subtests.test(event_type=event_type, vk=vk):
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
                assert result["event_type"] == event_type, f"Event type lost for {event_type} {vk}"
                assert result["vk"] == vk, f"VK code lost for {event_type} {vk}"

    def test_mouse_small_values_exhaustive(self, encoder, subtests):
        """Exhaustive test for small mouse movements (0-20 range) - must be exact."""
        for dx in range(-20, 21):  # -20 to 20
            for dy in range(-20, 21):  # -20 to 20
                with subtests.test(dx=dx, dy=dy):
                    original = {"last_x": dx, "last_y": dy, "button_flags": 0, "button_data": 0}
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

                    # Small values must be preserved exactly (no quantization loss allowed)
                    assert result["last_x"] == dx, f"X value changed: {dx} -> {result['last_x']}"
                    assert result["last_y"] == dy, f"Y value changed: {dy} -> {result['last_y']}"

    def test_timestamp_exhaustive(self, encoder, subtests):
        """Exhaustive test for timestamp encoding/decoding within reasonable range."""
        # Test range: 0 to 10 seconds in nanoseconds (factorized has smaller range than hierarchical)
        test_range_ns = 10 * 10**9  # 10 seconds total range

        # Test every 100ms within the range (100 values)
        step_ns = 100_000_000  # 100ms in nanoseconds

        for ts in range(0, test_range_ns, step_ns):
            with subtests.test(timestamp=ts):
                data = {"last_x": 1, "last_y": 1, "button_flags": 0, "button_data": 0}
                msg = McapMessage(
                    topic="mouse/raw",
                    timestamp=ts,
                    message=orjson.dumps(data),
                    message_type="desktop/RawMouseEvent",
                )

                # Round trip
                encoded, images = encoder.encode(msg)
                decoded = encoder.decode(encoded, images)

                # Within the quantization range, timestamps should be preserved exactly
                if ts < test_range_ns:
                    assert decoded.timestamp == ts, f"Timestamp changed: {ts} -> {decoded.timestamp}"

    def test_screen_fidelity(self, encoder, subtests):
        """Screen events should preserve structure and handle utc_ns vs timestamp correctly."""
        test_cases = [
            # (utc_ns, timestamp, source_shape, shape, media_ref, description)
            (3000000000, 3000000000, [1920, 1080], [1920, 1080], {"uri": "test1.png"}, "same utc_ns and timestamp"),
            (4000000000, 5000000000, [1280, 720], [1280, 720], {"uri": "test2.png"}, "different utc_ns and timestamp"),
        ]

        for utc_ns, timestamp, source_shape, shape, media_ref, desc in test_cases:
            with subtests.test(utc_ns=utc_ns, timestamp=timestamp, desc=desc):
                original = {
                    "utc_ns": utc_ns,
                    "source_shape": source_shape,
                    "shape": shape,
                    "media_ref": media_ref,
                }

                msg = McapMessage(
                    topic="screen",
                    timestamp=timestamp,
                    message=orjson.dumps(original),
                    message_type="desktop/ScreenCaptured",
                )

                # Encode and check image object preservation
                encoded, images = encoder.encode(msg)
                assert len(images) == 1
                assert isinstance(images[0], ScreenCaptured)
                assert images[0].utc_ns == utc_ns, f"utc_ns not preserved for {desc}"
                assert list(images[0].source_shape) == source_shape, f"source_shape not preserved for {desc}"
                assert list(images[0].shape) == shape, f"shape not preserved for {desc}"
                assert images[0].media_ref.uri == media_ref["uri"], f"media_ref not preserved for {desc}"

                # Decode and check message timestamp preservation
                decoded = encoder.decode(encoded, images)
                assert decoded.timestamp == timestamp, f"timestamp not preserved for {desc}"
                assert decoded.topic == "screen"
                assert decoded.message_type == "desktop/ScreenCaptured"

    def test_factorized_specific_tokens(self, encoder, subtests):
        """Test factorized-specific tokens like VK_*, MB_*, SIGN_* are used correctly."""
        # Test that factorized encoder uses VK tokens for keyboard events
        original = {"event_type": "press", "vk": 65}  # A key
        msg = McapMessage(
            topic="keyboard",
            timestamp=2000000000,
            message=orjson.dumps(original),
            message_type="desktop/KeyboardEvent",
        )

        encoded, images = encoder.encode(msg)

        # Should contain VK_65 token for A key
        assert "<VK_65>" in encoded, f"Expected <VK_65> token in encoded string: {encoded}"

        # Test that factorized encoder uses MB tokens for mouse buttons
        original = {"last_x": 0, "last_y": 0, "button_flags": 1, "button_data": 0}  # Left button
        msg = McapMessage(
            topic="mouse/raw",
            timestamp=1000000000,
            message=orjson.dumps(original),
            message_type="desktop/RawMouseEvent",
        )

        encoded, images = encoder.encode(msg)

        # Should contain MB tokens for button flags
        assert any(f"<MB_{i}>" in encoded for i in range(16)), f"Expected MB tokens in encoded string: {encoded}"

        # Test that factorized encoder uses SIGN tokens for negative values
        original = {"last_x": -5, "last_y": 3, "button_flags": 0, "button_data": 0}
        msg = McapMessage(
            topic="mouse/raw",
            timestamp=1000000000,
            message=orjson.dumps(original),
            message_type="desktop/RawMouseEvent",
        )

        encoded, images = encoder.encode(msg)

        # Should contain SIGN tokens for signed values
        assert "<SIGN_MINUS>" in encoded, f"Expected <SIGN_MINUS> token in encoded string: {encoded}"
        assert "<SIGN_PLUS>" in encoded, f"Expected <SIGN_PLUS> token in encoded string: {encoded}"


# =============================================================================
# 2. EFFICIENCY TESTS: Token count should be reasonable
# =============================================================================


class TestEfficiency:
    """Test that token count is reasonable, not bloated."""

    @pytest.fixture
    def encoder(self):
        return FactorizedEventEncoder()

    def test_mouse_token_count(self, encoder):
        """Mouse events should use reasonable number of tokens."""
        data = {"last_x": 100, "last_y": -50, "button_flags": 0x401, "button_data": 120}
        msg = McapMessage(
            topic="mouse/raw",
            timestamp=1000000000,
            message=orjson.dumps(data),
            message_type="desktop/RawMouseEvent",
        )

        encoded, _ = encoder.encode(msg)
        token_count = encoded.count("<")  # Count tokens

        # Factorized encoder should be more efficient than hierarchical
        # Expected structure: EVENT_START + timestamp + MOUSE + movement + flags + wheel + EVENT_END
        assert token_count <= 25, f"Token count too high for mouse event: {token_count} tokens"
        assert token_count >= 10, f"Token count too low for mouse event: {token_count} tokens"

    def test_keyboard_token_count(self, encoder):
        """Keyboard events should use reasonable number of tokens."""
        data = {"event_type": "press", "vk": 65}
        msg = McapMessage(
            topic="keyboard",
            timestamp=2000000000,
            message=orjson.dumps(data),
            message_type="desktop/KeyboardEvent",
        )

        encoded, _ = encoder.encode(msg)
        token_count = encoded.count("<")

        # Expected structure: EVENT_START + timestamp + KEYBOARD + VK_* + action + EVENT_END
        assert token_count <= 15, f"Token count too high for keyboard event: {token_count} tokens"
        assert token_count >= 6, f"Token count too low for keyboard event: {token_count} tokens"

    def test_screen_token_count(self, encoder):
        """Screen events should use reasonable number of tokens."""
        data = {
            "utc_ns": 3000000000,
            "source_shape": [1920, 1080],
            "shape": [1920, 1080],
            "media_ref": {"uri": "test.png"},
        }
        msg = McapMessage(
            topic="screen",
            timestamp=3000000000,
            message=orjson.dumps(data),
            message_type="desktop/ScreenCaptured",
        )

        encoded, _ = encoder.encode(msg)
        token_count = encoded.count("<")

        # Expected structure: EVENT_START + SCREEN + timestamp + fake_image_placeholder + EVENT_END
        assert token_count <= 10, f"Token count too high for screen event: {token_count} tokens"
        assert token_count >= 5, f"Token count too low for screen event: {token_count} tokens"


# =============================================================================
# 3. EDGE CASE TESTS: Works well even with edge cases
# =============================================================================


class TestEdgeCases:
    """Test that encoder handles edge cases well - users don't need to worry."""

    @pytest.fixture
    def encoder(self):
        return FactorizedEventEncoder()

    def test_extreme_mouse_values(self, encoder, subtests):
        """Extreme mouse values should be handled gracefully."""
        # Test cases with extreme mouse deltas
        extreme_cases = [
            {"last_x": 1000, "last_y": 0, "button_flags": 0, "button_data": 0},
            {"last_x": -1000, "last_y": 0, "button_flags": 0, "button_data": 0},
            {"last_x": 0, "last_y": 1000, "button_flags": 0, "button_data": 0},
            {"last_x": 0, "last_y": -1000, "button_flags": 0, "button_data": 0},
            {"last_x": 10000, "last_y": 10000, "button_flags": 0, "button_data": 0},
            {"last_x": -50000, "last_y": 50000, "button_flags": 0, "button_data": 0},
        ]

        for i, data in enumerate(extreme_cases):
            with subtests.test(case="extreme_delta", index=i, data=data):
                msg = McapMessage(
                    topic="mouse/raw",
                    timestamp=1000000000 + i,
                    message=orjson.dumps(data),
                    message_type="desktop/RawMouseEvent",
                )

                # Should handle gracefully (may quantize or clamp extreme values)
                encoded, images = encoder.encode(msg)
                decoded = encoder.decode(encoded, images)
                result = orjson.loads(decoded.message)

                # Should produce valid results
                assert isinstance(result["last_x"], int)
                assert isinstance(result["last_y"], int)

    def test_extreme_timestamps(self, encoder, subtests):
        """Extreme timestamp values should be handled gracefully."""
        extreme_timestamps = [
            0,  # Minimum
            9223372036854775807,  # Maximum int64
            1000000000000000000,  # Very large
        ]

        for ts in extreme_timestamps:
            with subtests.test(timestamp=ts):
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
                assert isinstance(decoded.timestamp, int)
                assert decoded.timestamp >= 0

    def test_zero_values(self, encoder):
        """Zero values should be handled correctly."""
        # Zero mouse movement
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

        assert result["last_x"] == 0
        assert result["last_y"] == 0
        assert result["button_flags"] == 0
        assert result["button_data"] == 0

    def test_basic_functionality(self, encoder):
        """Basic encoder functionality should work."""
        # Can create vocab
        vocab = encoder.get_vocab()
        assert len(vocab) > 0

        # Contains expected factorized tokens
        assert "<EVENT_START>" in vocab
        assert "<EVENT_END>" in vocab
        assert "<MOUSE>" in vocab
        assert "<KEYBOARD>" in vocab
        assert "<SIGN_PLUS>" in vocab
        assert "<SIGN_MINUS>" in vocab
        assert "<VK_65>" in vocab  # A key
        assert "<MB_0>" in vocab  # Mouse button token


# =============================================================================
# 4. MISCELLANEOUS TESTS: Configuration and other utilities
# =============================================================================


class TestMiscellaneous:
    """Test configuration validation and other utilities."""

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration should work
        encoder = FactorizedEventEncoder()
        assert encoder.config is not None

        # Test custom configuration
        config = FactorizedEventEncoderConfig()
        encoder_custom = FactorizedEventEncoder(config)
        assert encoder_custom.config is not None

    def test_vocab_structure(self):
        """Test that factorized encoder has expected vocabulary structure."""
        encoder = FactorizedEventEncoder()
        vocab = encoder.get_vocab()

        # Should have numeric tokens 0-9
        for i in range(10):
            assert f"<{i}>" in vocab, f"Missing numeric token <{i}>"

        # Should have sign tokens
        assert "<SIGN_PLUS>" in vocab
        assert "<SIGN_MINUS>" in vocab

        # Should have VK tokens for common keys
        common_vks = [65, 66, 67, 13, 27, 32]  # A, B, C, Enter, Esc, Space
        for vk in common_vks:
            assert f"<VK_{vk}>" in vocab, f"Missing VK token <VK_{vk}>"

        # Should have MB tokens
        for i in range(16):
            assert f"<MB_{i}>" in vocab, f"Missing MB token <MB_{i}>"

        # Should have action tokens
        assert "<press>" in vocab
        assert "<release>" in vocab


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
