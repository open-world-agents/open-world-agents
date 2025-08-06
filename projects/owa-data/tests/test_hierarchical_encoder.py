#!/usr/bin/env python3
"""
TESTING PRIORITIES:
==================
1. FIDELITY: Encode-decode yields same/similar result (no data loss)
2. EFFICIENCY: Token count is reasonable (not bloated)
3. NO EDGE CASES: Works well even with edge cases - users don't need to worry
"""

import json

import pytest

from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.data.encoders.hierarchical_event_encoder import HierarchicalEventEncoder

# =============================================================================
# 1. FIDELITY TESTS: Encode-decode must preserve data
# =============================================================================


class TestFidelity:
    """Test that encoding-decoding preserves data without loss."""

    @pytest.fixture
    def encoder(self):
        return HierarchicalEventEncoder()

    def test_mouse_fidelity(self, encoder):
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
            original = {"last_x": dx, "last_y": dy, "button_flags": flags, "button_data": data}
            msg = McapMessage(
                topic="mouse/raw",
                timestamp=1000000000,
                message=json.dumps(original).encode("utf-8"),
                message_type="desktop/RawMouseEvent",
            )

            # Round trip
            encoded, images = encoder.encode(msg)
            decoded = encoder.decode(encoded, images)
            result = json.loads(decoded.message.decode("utf-8"))

            # Check fidelity - values within range must be exact, out-of-range values get clamped
            max_x, max_y = encoder.config.max_mouse_delta
            expected_x = max(-max_x, min(max_x, dx))  # Clamp to range
            expected_y = max(-max_y, min(max_y, dy))  # Clamp to range

            assert result["last_x"] == expected_x, (
                f"X movement incorrect in {desc}: expected {expected_x}, got {result['last_x']}"
            )
            assert result["last_y"] == expected_y, (
                f"Y movement incorrect in {desc}: expected {expected_y}, got {result['last_y']}"
            )
            assert result["button_flags"] == flags, f"Button flags lost in {desc}"

            # Button data is quantized by 120, so check expected value
            expected_data = (data // 120) * 120 if flags & 0x400 else data
            assert result["button_data"] == expected_data, f"Button data lost in {desc}"

    def test_keyboard_fidelity(self, encoder):
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
            original = {"event_type": event_type, "vk": vk}
            msg = McapMessage(
                topic="keyboard",
                timestamp=2000000000,
                message=json.dumps(original).encode("utf-8"),
                message_type="desktop/KeyboardEvent",
            )

            # Round trip
            encoded, images = encoder.encode(msg)
            decoded = encoder.decode(encoded, images)
            result = json.loads(decoded.message.decode("utf-8"))

            # Check perfect fidelity
            assert result["event_type"] == event_type, f"Event type lost for {event_type} {vk}"
            assert result["vk"] == vk, f"VK code lost for {event_type} {vk}"

    def test_mouse_small_values_exhaustive(self, encoder):
        """Exhaustive test for small mouse movements (0-20 range) - must be exact."""
        for dx in range(-20, 21):  # -20 to 20
            for dy in range(-20, 21):  # -20 to 20
                original = {"last_x": dx, "last_y": dy, "button_flags": 0, "button_data": 0}
                msg = McapMessage(
                    topic="mouse/raw",
                    timestamp=1000000000,
                    message=json.dumps(original).encode("utf-8"),
                    message_type="desktop/RawMouseEvent",
                )

                # Round trip
                encoded, images = encoder.encode(msg)
                decoded = encoder.decode(encoded, images)
                result = json.loads(decoded.message.decode("utf-8"))

                # Small values must be preserved exactly (no quantization loss allowed)
                assert result["last_x"] == dx, f"X value changed: {dx} -> {result['last_x']}"
                assert result["last_y"] == dy, f"Y value changed: {dy} -> {result['last_y']}"

    def test_screen_fidelity(self, encoder):
        """Screen events should preserve structure."""
        from owa.msgs.desktop.screen import ScreenCaptured

        original = {
            "utc_ns": 3000000000,
            "source_shape": [1920, 1080],
            "shape": [1920, 1080],
            "media_ref": {"uri": "test.png"},
        }

        msg = McapMessage(
            topic="screen",
            timestamp=3000000000,
            message=json.dumps(original).encode("utf-8"),
            message_type="desktop/ScreenCaptured",
        )

        # Encode (should produce image object)
        encoded, images = encoder.encode(msg)

        # Check that we get a ScreenCaptured object back
        assert len(images) == 1
        assert isinstance(images[0], ScreenCaptured)
        assert images[0].utc_ns == 3000000000


# =============================================================================
# 2. EFFICIENCY TESTS: Token count should be reasonable
# =============================================================================


class TestEfficiency:
    """Test that token count is reasonable, not bloated."""

    @pytest.fixture
    def encoder(self):
        return HierarchicalEventEncoder()

    def test_mouse_token_count(self, encoder):
        """Mouse events should use exact expected number of tokens."""
        data = {"last_x": 100, "last_y": -50, "button_flags": 0x401, "button_data": 120}
        msg = McapMessage(
            topic="mouse/raw",
            timestamp=1000000000,
            message=json.dumps(data).encode("utf-8"),
            message_type="desktop/RawMouseEvent",
        )

        encoded, _ = encoder.encode(msg)
        token_count = encoded.count("<")  # Count tokens

        # Expected: EVENT_START + TIMESTAMP(4) + MOUSE + movement(6) + flags(3) + wheel(1) + EVENT_END = 17
        assert token_count == 17, f"Expected exactly 17 tokens for mouse event, got {token_count}"

    def test_keyboard_token_count(self, encoder):
        """Keyboard events should use exact expected number of tokens."""
        data = {"event_type": "press", "vk": 65}
        msg = McapMessage(
            topic="keyboard",
            timestamp=2000000000,
            message=json.dumps(data).encode("utf-8"),
            message_type="desktop/KeyboardEvent",
        )

        encoded, _ = encoder.encode(msg)
        token_count = encoded.count("<")

        # Expected: EVENT_START + TIMESTAMP(4) + KEYBOARD + vk + event_type + EVENT_END = 9
        assert token_count == 9, f"Expected exactly 9 tokens for keyboard event, got {token_count}"

    def test_screen_token_count(self, encoder):
        """Screen events should use exact expected number of tokens."""
        data = {
            "utc_ns": 3000000000,
            "source_shape": [1920, 1080],
            "shape": [1920, 1080],
            "media_ref": {"uri": "test.png"},
        }
        msg = McapMessage(
            topic="screen",
            timestamp=3000000000,
            message=json.dumps(data).encode("utf-8"),
            message_type="desktop/ScreenCaptured",
        )

        encoded, _ = encoder.encode(msg)
        token_count = encoded.count("<")

        # Expected: EVENT_START + TIMESTAMP(4) + image_prefix + image + image_suffix + EVENT_END = 10
        assert token_count == 10, f"Expected exactly 10 tokens for screen event, got {token_count}"


# =============================================================================
# 3. EDGE CASE TESTS: Works well even with edge cases
# =============================================================================


class TestEdgeCases:
    """Test that encoder handles edge cases well - users don't need to worry."""

    @pytest.fixture
    def encoder(self):
        return HierarchicalEventEncoder()

    def test_extreme_mouse_values(self, encoder):
        """Extreme mouse values should be handled gracefully."""
        edge_cases = [
            # Extreme movements
            {"last_x": 10000, "last_y": 10000, "button_flags": 0, "button_data": 0},
            {"last_x": -10000, "last_y": -10000, "button_flags": 0, "button_data": 0},
            {"last_x": 50000, "last_y": -50000, "button_flags": 0, "button_data": 0},
            {"last_x": -100000, "last_y": 100000, "button_flags": 0, "button_data": 0},
            # Extreme button data
            {"last_x": 0, "last_y": 0, "button_flags": 0x400, "button_data": 32767},
            {"last_x": 0, "last_y": 0, "button_flags": 0x400, "button_data": -32768},
            {"last_x": 0, "last_y": 0, "button_flags": 0x400, "button_data": 100000},
            {"last_x": 0, "last_y": 0, "button_flags": 0x400, "button_data": -100000},
            # Maximum flag values
            {"last_x": 0, "last_y": 0, "button_flags": 0xFFF, "button_data": 0},
            {"last_x": 0, "last_y": 0, "button_flags": 0xFFFF, "button_data": 0},  # Beyond 3-digit hex
            # Combined extremes
            {"last_x": 99999, "last_y": -99999, "button_flags": 0xFFF, "button_data": 32767},
            {"last_x": -99999, "last_y": 99999, "button_flags": 0xFFF, "button_data": -32768},
        ]

        for i, data in enumerate(edge_cases):
            msg = McapMessage(
                topic="mouse/raw",
                timestamp=1000000000 + i,
                message=json.dumps(data).encode("utf-8"),
                message_type="desktop/RawMouseEvent",
            )

            # Should handle gracefully without crashing
            encoded, images = encoder.encode(msg)
            decoded = encoder.decode(encoded, images)
            result = json.loads(decoded.message.decode("utf-8"))

            # Should produce valid results (may be clamped/quantized)
            assert isinstance(result["last_x"], int)
            assert isinstance(result["last_y"], int)
            assert isinstance(result["button_flags"], int)
            assert isinstance(result["button_data"], int)

    def test_extreme_timestamps(self, encoder):
        """Extreme timestamp values should be handled gracefully."""
        extreme_timestamps = [
            0,  # Minimum
            9223372036854775807,  # Maximum int64
            1000000000000000000,  # Very large
        ]

        for ts in extreme_timestamps:
            data = {"last_x": 10, "last_y": 10, "button_flags": 0, "button_data": 0}
            msg = McapMessage(
                topic="mouse/raw",
                timestamp=ts,
                message=json.dumps(data).encode("utf-8"),
                message_type="desktop/RawMouseEvent",
            )

            # Should handle gracefully (may quantize large timestamps)
            encoded, images = encoder.encode(msg)
            decoded = encoder.decode(encoded, images)

            # Should produce valid timestamp
            assert isinstance(decoded.timestamp, int)
            assert decoded.timestamp >= 0

    def test_extreme_keyboard_values(self, encoder):
        """Extreme keyboard values should be handled gracefully."""
        extreme_cases = [
            # Extreme VK codes
            ("press", 0),  # Minimum VK
            ("release", 0),  # Minimum VK
            ("press", 255),  # Maximum standard VK
            ("release", 255),  # Maximum standard VK
            ("press", 65535),  # Very large VK
            ("release", 65535),  # Very large VK
            # All event types with extreme VKs
            ("press", 1000),
            ("release", 1000),
        ]

        for event_type, vk in extreme_cases:
            data = {"event_type": event_type, "vk": vk}
            msg = McapMessage(
                topic="keyboard",
                timestamp=4000000000 + vk,
                message=json.dumps(data).encode("utf-8"),
                message_type="desktop/KeyboardEvent",
            )

            # Should handle gracefully
            encoded, images = encoder.encode(msg)
            decoded = encoder.decode(encoded, images)
            result = json.loads(decoded.message.decode("utf-8"))

            # Should produce valid results
            assert result["event_type"] == event_type
            assert isinstance(result["vk"], int)

    def test_zero_values(self, encoder):
        """Zero values should be handled correctly."""
        # Zero mouse movement
        data = {"last_x": 0, "last_y": 0, "button_flags": 0, "button_data": 0}
        msg = McapMessage(
            topic="mouse/raw",
            timestamp=0,
            message=json.dumps(data).encode("utf-8"),
            message_type="desktop/RawMouseEvent",
        )

        encoded, images = encoder.encode(msg)
        decoded = encoder.decode(encoded, images)
        result = json.loads(decoded.message.decode("utf-8"))

        assert result["last_x"] == 0
        assert result["last_y"] == 0
        assert result["button_flags"] == 0
        assert result["button_data"] == 0

    def test_basic_functionality(self, encoder):
        """Basic encoder functionality should work."""
        # Can create vocab
        vocab = encoder.get_vocab()
        assert len(vocab) > 0

        # Contains expected tokens
        assert "<EVENT_START>" in vocab
        assert "<EVENT_END>" in vocab
        assert "<MOUSE>" in vocab
        assert "<KEYBOARD>" in vocab


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
