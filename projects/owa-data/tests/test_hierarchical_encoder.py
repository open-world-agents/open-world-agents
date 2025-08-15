#!/usr/bin/env python3
"""
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

ACTUAL TEST BREAKDOWN:
=====================
FIDELITY (5): mouse_fidelity, keyboard_fidelity, mouse_small_values_exhaustive, timestamp_exhaustive, screen_fidelity, test_mouse_validation
EFFICIENCY (3): mouse_token_count, keyboard_token_count, screen_token_count
EDGE CASES (6): extreme_mouse_values, extreme_timestamps, extreme_keyboard_values, extreme_screen_values, zero_values, basic_functionality
"""

import orjson
import pytest

from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.data.encoders.hierarchical_event_encoder import HierarchicalEventEncoder
from owa.msgs.desktop.screen import ScreenCaptured

# =============================================================================
# 1. FIDELITY TESTS: Encode-decode must preserve data
# =============================================================================


class TestFidelity:
    """Test that encoding-decoding preserves data without loss."""

    @pytest.fixture
    def encoder(self):
        return HierarchicalEventEncoder()

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
        # Test range: 0 to 16 seconds in nanoseconds (covers typical event intervals)
        test_range_ns = 16 * 10**9  # 16 seconds total range

        # Test every 100ms within the range (160 values)
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

    def test_mouse_validation(self, encoder, subtests):
        """Mouse encoder should validate input ranges and warn for invalid delta values."""
        min_delta, max_delta = encoder.config.mouse_delta_range

        # Test boundary values (should work)
        valid_cases = [
            (max_delta, max_delta),  # At positive boundary
            (min_delta, min_delta),  # At negative boundary
            (0, 0),  # Zero
            (max_delta, min_delta),  # Mixed boundaries
        ]

        for dx, dy in valid_cases:
            with subtests.test(case="valid", dx=dx, dy=dy):
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
                assert result["last_x"] == dx
                assert result["last_y"] == dy

        # Test invalid values (should issue warnings and clamp values)
        invalid_cases = [
            (max_delta + 1, 0),  # X too large
            (min_delta - 1, 0),  # X too small
            (0, max_delta + 1),  # Y too large
            (0, min_delta - 1),  # Y too small
        ]

        for dx, dy in invalid_cases:
            with subtests.test(case="invalid", dx=dx, dy=dy):
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
                    assert min_delta <= result["last_x"] <= max_delta
                    assert min_delta <= result["last_y"] <= max_delta


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
            message=orjson.dumps(data),
            message_type="desktop/RawMouseEvent",
        )

        encoded, _ = encoder.encode(msg)
        token_count = encoded.count("<")  # Count tokens

        # Expected: EVENT_START + timestamp(3) + MOUSE + movement(8) + flags(3) + wheel(2) + EVENT_END = 19
        assert token_count == 19, f"Expected exactly 19 tokens for mouse event, got {token_count}"

    def test_keyboard_token_count(self, encoder):
        """Keyboard events should use exact expected number of tokens."""
        data = {"event_type": "press", "vk": 65}
        msg = McapMessage(
            topic="keyboard",
            timestamp=2000000000,
            message=orjson.dumps(data),
            message_type="desktop/KeyboardEvent",
        )

        encoded, _ = encoder.encode(msg)
        token_count = encoded.count("<")

        # Expected: EVENT_START + timestamp(3) + KEYBOARD + vk + event_type + EVENT_END = 8
        assert token_count == 8, f"Expected exactly 8 tokens for keyboard event, got {token_count}"

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
            message=orjson.dumps(data),
            message_type="desktop/ScreenCaptured",
        )

        encoded, _ = encoder.encode(msg)
        token_count = encoded.count("<")

        # Expected: EVENT_START + timestamp(3) + fake_image_placeholder + EVENT_END = 6
        # (simplified: no prefix/suffix in encoder, just single <fake_image_placeholder> token)
        assert token_count == 6, f"Expected exactly 6 tokens for screen event, got {token_count}"


# =============================================================================
# 3. EDGE CASE TESTS: Works well even with edge cases
# =============================================================================


class TestEdgeCases:
    """Test that encoder handles edge cases well - users don't need to worry."""

    @pytest.fixture
    def encoder(self):
        return HierarchicalEventEncoder()

    def test_extreme_mouse_values(self, encoder, subtests):
        """Extreme mouse values should issue warnings for deltas and raise errors for button_flags."""
        min_delta, max_delta = encoder.config.mouse_delta_range

        # Test cases with out-of-range mouse deltas (should warn and clamp)
        invalid_delta_cases = [
            {"last_x": max_delta + 1, "last_y": 0, "button_flags": 0, "button_data": 0},
            {"last_x": min_delta - 1, "last_y": 0, "button_flags": 0, "button_data": 0},
            {"last_x": 0, "last_y": max_delta + 1, "button_flags": 0, "button_data": 0},
            {"last_x": 0, "last_y": min_delta - 1, "button_flags": 0, "button_data": 0},
            {"last_x": 10000, "last_y": 10000, "button_flags": 0, "button_data": 0},
            {"last_x": -50000, "last_y": 50000, "button_flags": 0, "button_data": 0},
        ]

        for i, data in enumerate(invalid_delta_cases):
            with subtests.test(case="invalid_delta", index=i, data=data):
                msg = McapMessage(
                    topic="mouse/raw",
                    timestamp=1000000000 + i,
                    message=orjson.dumps(data),
                    message_type="desktop/RawMouseEvent",
                )

                # Should warn for out-of-range mouse deltas and work (with clamping)
                with pytest.warns(UserWarning, match=r"Mouse delta value .* is outside valid range"):
                    encoded, images = encoder.encode(msg)
                    decoded = encoder.decode(encoded, images)
                    result = orjson.loads(decoded.message)
                    # Values should be clamped to valid range
                    assert min_delta <= result["last_x"] <= max_delta
                    assert min_delta <= result["last_y"] <= max_delta

        # Test cases with invalid button_flags (should still raise ValueError)
        invalid_button_cases = [
            {"last_x": 0, "last_y": 0, "button_flags": 0xFFFF, "button_data": 0},  # 4-digit hex
            {"last_x": 0, "last_y": 0, "button_flags": 0x1000, "button_data": 0},  # Just above 3-digit hex
            {"last_x": 0, "last_y": 0, "button_flags": 0x12345, "button_data": 0},  # 5-digit hex
            {"last_x": 0, "last_y": 0, "button_flags": 65536, "button_data": 0},  # Large decimal value
        ]

        for i, data in enumerate(invalid_button_cases):
            with subtests.test(case="invalid_button", index=i, data=data):
                msg = McapMessage(
                    topic="mouse/raw",
                    timestamp=1000000000 + i + 100,
                    message=orjson.dumps(data),
                    message_type="desktop/RawMouseEvent",
                )

                # Should raise ValueError for invalid button_flags
                with pytest.raises(
                    ValueError, match=r"Mouse button_flags value .* is outside valid 3-digit hex range"
                ):
                    encoder.encode(msg)

        # Get valid scroll range from encoder config
        min_scroll, max_scroll = encoder.config.mouse_scroll_range

        # Test cases with extreme scroll values (should raise ValueError)
        invalid_scroll_cases = [
            {"last_x": 0, "last_y": 0, "button_flags": 0x400, "button_data": (max_scroll + 1) * 120},  # Just above max
            {"last_x": 0, "last_y": 0, "button_flags": 0x400, "button_data": (min_scroll - 1) * 120},  # Just below min
            {"last_x": 0, "last_y": 0, "button_flags": 0x400, "button_data": 32767},  # Very large value
        ]

        for i, data in enumerate(invalid_scroll_cases):
            with subtests.test(case="invalid_scroll", index=i, data=data):
                msg = McapMessage(
                    topic="mouse/raw",
                    timestamp=2000000000 + i,
                    message=orjson.dumps(data),
                    message_type="desktop/RawMouseEvent",
                )

                # Should raise ValueError for invalid scroll values
                with pytest.raises(ValueError, match=r"Mouse scroll value .* is outside valid range"):
                    encoder.encode(msg)

        # Test cases with valid scroll values and flags (should work)
        min_delta_mouse, max_delta_mouse = encoder.config.mouse_delta_range
        valid_cases = [
            {"last_x": 0, "last_y": 0, "button_flags": 0x400, "button_data": max_scroll * 120},  # Max valid scroll
            {"last_x": 0, "last_y": 0, "button_flags": 0x400, "button_data": min_scroll * 120},  # Min valid scroll
            {"last_x": 0, "last_y": 0, "button_flags": 0xFFF, "button_data": 0},  # Max valid 3-digit hex
            {
                "last_x": max_delta_mouse,
                "last_y": max_delta_mouse,
                "button_flags": 0xFFF,
                "button_data": 0,
            },  # At valid boundary
        ]

        for i, data in enumerate(valid_cases):
            with subtests.test(case="valid", index=i, data=data):
                msg = McapMessage(
                    topic="mouse/raw",
                    timestamp=3000000000 + i,
                    message=orjson.dumps(data),
                    message_type="desktop/RawMouseEvent",
                )

                # Should work without errors
                encoded, images = encoder.encode(msg)
                decoded = encoder.decode(encoded, images)
                result = orjson.loads(decoded.message)

                # Should produce valid results by checking for data fidelity
                assert result["last_x"] == data["last_x"]
                assert result["last_y"] == data["last_y"]
                assert result["button_flags"] == data["button_flags"]
                expected_data = (
                    (data["button_data"] // 120) * 120 if data["button_flags"] & 0x400 else data["button_data"]
                )
                assert result["button_data"] == expected_data

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

    def test_extreme_keyboard_values(self, encoder, subtests):
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
            with subtests.test(event_type=event_type, vk=vk):
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
                assert result["event_type"] == event_type
                assert isinstance(result["vk"], int)

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

    def test_extreme_screen_values(self, encoder, subtests):
        """Extreme screen values should be handled gracefully."""
        extreme_cases = [
            # Extreme resolutions
            {"utc_ns": 0, "source_shape": [1, 1], "shape": [1, 1], "media_ref": {"uri": "tiny.png"}},
            {
                "utc_ns": 9223372036854775807,
                "source_shape": [7680, 4320],
                "shape": [7680, 4320],
                "media_ref": {"uri": "8k.png"},
            },
            # Extreme timestamps
            {
                "utc_ns": 1000000000000000000,
                "source_shape": [1920, 1080],
                "shape": [1920, 1080],
                "media_ref": {"uri": "future.png"},
            },
            # Different source vs processed shapes
            {
                "utc_ns": 5000000000,
                "source_shape": [3840, 2160],
                "shape": [1920, 1080],
                "media_ref": {"uri": "downscaled.png"},
            },
        ]

        for i, data in enumerate(extreme_cases):
            with subtests.test(index=i, utc_ns=data["utc_ns"], uri=data["media_ref"]["uri"]):
                msg = McapMessage(
                    topic="screen",
                    timestamp=6000000000 + i,
                    message=orjson.dumps(data),
                    message_type="desktop/ScreenCaptured",
                )

                # Should handle gracefully without crashing
                encoded, images = encoder.encode(msg)
                decoded = encoder.decode(encoded, images)

                # Should produce valid results
                assert len(images) == 1
                assert isinstance(images[0], ScreenCaptured)
                assert isinstance(decoded.timestamp, int)
                assert decoded.topic == "screen"

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
        assert "<press>" in vocab
        assert "<release>" in vocab


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
