"""
Tests for EventEncoder class.

This module contains comprehensive tests for the EventEncoder functionality,
including encoding/decoding of different event types and error handling.
"""

import pytest

from owa.data.event_encoder import EventEncoder
from owa.env.gst.msg import ScreenEmitted


class TestEventEncoder:
    """Test suite for EventEncoder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = EventEncoder()  # Default: drop_file_path=True
        self.encoder_keep_path = EventEncoder(drop_file_path=False)

        # Sample keyboard event
        self.keyboard_event = {
            "file_path": "/mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-new-jy-1.mcap",
            "topic": "keyboard",
            "timestamp_ns": 1745362786814673800,
            "message_type": "owa.env.desktop.msg.KeyboardEvent",
            "msg": b'{"event_type":"press","vk":37}',
        }

        # Sample mouse event
        self.mouse_event = {
            "file_path": "/mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-jy-2.mcap",
            "topic": "mouse",
            "timestamp_ns": 1745362786814673900,
            "message_type": "owa.env.desktop.msg.MouseEvent",
            "msg": b'{"event_type":"move","x":100,"y":200}',
        }

        # Sample screen event
        self.screen_event = {
            "file_path": "/mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-jy-3.mcap",
            "topic": "screen",
            "timestamp_ns": 1743128886688495300,
            "message_type": "owa.env.gst.msg.ScreenEmitted",
            "msg": b'{"path":"expert-jy-3.mkv","pts":70350000000,"utc_ns":1743128886688495300}',
        }

    def test_encode_keyboard_event(self):
        """Test encoding of keyboard events with default drop_file_path=True."""
        text, images = self.encoder.encode(self.keyboard_event)

        # Check text format
        assert text.startswith("<EVENT_START>")
        assert text.endswith("<EVENT_END>")
        assert "keyboard" in text
        assert "KeyboardEvent" in text
        # file_path should be dropped by default
        assert "file_path" not in text

        # Keyboard events should have no images
        assert images == []

    def test_encode_keyboard_event_keep_file_path(self):
        """Test encoding of keyboard events with drop_file_path=False."""
        text, images = self.encoder_keep_path.encode(self.keyboard_event)

        # Check text format
        assert text.startswith("<EVENT_START>")
        assert text.endswith("<EVENT_END>")
        assert "keyboard" in text
        assert "KeyboardEvent" in text
        # file_path should be kept
        assert "file_path" in text

        # Keyboard events should have no images
        assert images == []

    def test_encode_mouse_event(self):
        """Test encoding of mouse events."""
        text, images = self.encoder.encode(self.mouse_event)

        # Check text format
        assert text.startswith("<EVENT_START>")
        assert text.endswith("<EVENT_END>")
        assert "mouse" in text
        assert "MouseEvent" in text

        # Mouse events should have no images
        assert images == []

    def test_encode_screen_event(self):
        """Test encoding of screen events with image data."""
        text, images = self.encoder.encode(self.screen_event)

        # Check text format
        assert text.startswith("<EVENT_START>")
        assert text.endswith("<EVENT_END>")
        assert "screen" in text
        assert "ScreenEmitted" in text
        assert "<IMAGE>" in text

        # Screen events should have image data
        assert len(images) == 1
        assert isinstance(images[0], dict)
        assert "screen_event" in images[0]
        assert "original_msg" in images[0]
        assert isinstance(images[0]["screen_event"], ScreenEmitted)
        assert images[0]["screen_event"].path == "expert-jy-3.mkv"
        assert images[0]["screen_event"].pts == 70350000000

    def test_decode_keyboard_event(self):
        """Test decoding of keyboard events with file_path preservation."""
        # Use encoder that keeps file_path for exact round-trip
        text, images = self.encoder_keep_path.encode(self.keyboard_event)

        # Then decode
        decoded = self.encoder_keep_path.decode(text, images)

        # Should match original exactly
        assert decoded == self.keyboard_event

    def test_decode_mouse_event(self):
        """Test decoding of mouse events with file_path preservation."""
        # Use encoder that keeps file_path for exact round-trip
        text, images = self.encoder_keep_path.encode(self.mouse_event)

        # Then decode
        decoded = self.encoder_keep_path.decode(text, images)

        # Should match original exactly
        assert decoded == self.mouse_event

    def test_decode_screen_event(self):
        """Test decoding of screen events with file_path preservation."""
        # Use encoder that keeps file_path for exact round-trip
        text, images = self.encoder_keep_path.encode(self.screen_event)

        # Then decode
        decoded = self.encoder_keep_path.decode(text, images)

        # Should match original exactly
        assert decoded == self.screen_event

    def test_decode_with_dropped_file_path(self):
        """Test decoding with default drop_file_path=True behavior."""
        # Encode with default encoder (drops file_path)
        text, images = self.encoder.encode(self.keyboard_event)

        # Decode
        decoded = self.encoder.decode(text, images)

        # Should have file_path set to "<DROPPED>"
        expected = self.keyboard_event.copy()
        expected["file_path"] = "<DROPPED>"
        assert decoded == expected

    def test_encode_batch(self):
        """Test batch encoding of multiple events."""
        events = [self.keyboard_event, self.mouse_event, self.screen_event]

        texts, all_images = self.encoder.encode_batch(events)

        assert len(texts) == 3
        assert len(all_images) == 3

        # Check individual results
        assert all(text.startswith("<EVENT_START>") for text in texts)
        assert all(text.endswith("<EVENT_END>") for text in texts)

        # Only screen event should have images
        assert all_images[0] == []  # keyboard
        assert all_images[1] == []  # mouse
        assert len(all_images[2]) == 1  # screen
        assert isinstance(all_images[2][0], dict)  # screen image data format

    def test_decode_batch(self):
        """Test batch decoding of multiple events with file_path preservation."""
        events = [self.keyboard_event, self.mouse_event, self.screen_event]

        # Use encoder that keeps file_path for exact round-trip
        texts, all_images = self.encoder_keep_path.encode_batch(events)

        # Then decode
        decoded_events = self.encoder_keep_path.decode_batch(texts, all_images)

        # Should match originals exactly
        assert decoded_events == events

    def test_invalid_raw_event_type(self):
        """Test error handling for invalid raw event type."""
        with pytest.raises(ValueError, match="raw_event must be a dictionary"):
            self.encoder.encode("not a dict")

    def test_missing_required_keys(self):
        """Test error handling for missing required keys."""
        incomplete_event = {"file_path": "/path/to/file.mcap"}

        with pytest.raises(ValueError, match="missing required keys"):
            self.encoder.encode(incomplete_event)

    def test_invalid_serialized_format(self):
        """Test error handling for invalid serialized format."""
        invalid_text = "missing start and end tokens"

        with pytest.raises(ValueError, match="missing <EVENT_START> or <EVENT_END> tokens"):
            self.encoder.decode(invalid_text)

    def test_invalid_event_content(self):
        """Test error handling for invalid event content."""
        invalid_text = "<EVENT_START>not valid python<EVENT_END>"

        with pytest.raises(ValueError, match="Failed to parse event content"):
            self.encoder.decode(invalid_text)

    def test_screen_event_missing_images(self):
        """Test error handling when screen event is missing image data."""
        text, _ = self.encoder.encode(self.screen_event)

        with pytest.raises(ValueError, match="Screen event requires image data but none provided"):
            self.encoder.decode(text, images=[])

    def test_batch_length_mismatch(self):
        """Test error handling for batch length mismatch."""
        texts = ["<EVENT_START>test<EVENT_END>"]
        images = [[], []]  # Different length

        with pytest.raises(ValueError, match="Length mismatch between texts and images"):
            self.encoder.decode_batch(texts, images)

    def test_invalid_json_in_message(self):
        """Test error handling for invalid JSON in message content."""
        invalid_event = self.keyboard_event.copy()
        invalid_event["msg"] = b"invalid json content"

        # This should still work for non-screen events since we don't parse the JSON
        text, images = self.encoder.encode(invalid_event)
        assert text.startswith("<EVENT_START>")
        assert images == []

    def test_invalid_screen_event_json(self):
        """Test error handling for invalid JSON in screen event."""
        invalid_screen_event = self.screen_event.copy()
        invalid_screen_event["msg"] = b"invalid json content"

        with pytest.raises(ValueError, match="Failed to parse screen event message"):
            self.encoder.encode(invalid_screen_event)


class TestEventEncoderIntegration:
    """Integration tests for EventEncoder with real-world scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = EventEncoder()

    def test_round_trip_consistency(self):
        """Test that encode->decode is consistent for all event types."""
        events = [
            {
                "file_path": "/test/keyboard.mcap",
                "topic": "keyboard",
                "timestamp_ns": 1000000000,
                "message_type": "owa.env.desktop.msg.KeyboardEvent",
                "msg": b'{"event_type":"press","vk":65}',
            },
            {
                "file_path": "/test/mouse.mcap",
                "topic": "mouse",
                "timestamp_ns": 2000000000,
                "message_type": "owa.env.desktop.msg.MouseEvent",
                "msg": b'{"event_type":"click","x":500,"y":300,"button":"left","pressed":true}',
            },
            {
                "file_path": "/test/screen.mcap",
                "topic": "screen",
                "timestamp_ns": 3000000000,
                "message_type": "owa.env.gst.msg.ScreenEmitted",
                "msg": b'{"path":"test.mkv","pts":1000000000,"utc_ns":3000000000}',
            },
        ]

        # Use encoder that keeps file_path for exact round-trip
        encoder_keep_path = EventEncoder(drop_file_path=False)

        for event in events:
            # Encode then decode
            text, images = encoder_keep_path.encode(event)
            decoded = encoder_keep_path.decode(text, images)

            # Should be identical
            assert decoded == event

    def test_empty_batch(self):
        """Test handling of empty batches."""
        texts, images = self.encoder.encode_batch([])
        assert texts == []
        assert images == []

        decoded = self.encoder.decode_batch([], [])
        assert decoded == []
