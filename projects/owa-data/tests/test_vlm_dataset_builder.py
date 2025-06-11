"""
Tests for VLMDatasetBuilder class.

This module contains tests for the VLM dataset building functionality,
including event processing and format conversion for VLM training.
"""

import pytest

from owa.data.vlm_dataset_builder import VLMDatasetBuilder


class TestVLMDatasetBuilder:
    """Test suite for VLMDatasetBuilder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = VLMDatasetBuilder(drop_file_path=True)

        # Sample events
        self.keyboard_event = {
            "file_path": "/test/keyboard.mcap",
            "topic": "keyboard",
            "timestamp_ns": 1000000000,
            "message_type": "owa.env.desktop.msg.KeyboardEvent",
            "msg": b'{"event_type":"press","vk":65}',
        }

        self.mouse_event = {
            "file_path": "/test/mouse.mcap",
            "topic": "mouse",
            "timestamp_ns": 2000000000,
            "message_type": "owa.env.desktop.msg.MouseEvent",
            "msg": b'{"event_type":"click","x":100,"y":200,"button":"left","pressed":true}',
        }

        self.screen_event = {
            "file_path": "/test/screen.mcap",
            "topic": "screen",
            "timestamp_ns": 3000000000,
            "message_type": "owa.env.gst.msg.ScreenEmitted",
            "msg": b'{"path":"test.mkv","pts":1000000000,"utc_ns":3000000000}',
        }

    def test_initialization(self):
        """Test VLMDatasetBuilder initialization."""
        builder = VLMDatasetBuilder(drop_file_path=False)
        assert builder.encoder.drop_file_path == False

        builder_default = VLMDatasetBuilder()
        assert builder_default.encoder.drop_file_path == True

    def test_process_empty_sequence(self):
        """Test processing empty event sequence."""
        instruction = "Type 'Hello World'"
        result = self.builder.process_event_sequence([], instruction)

        assert result["encoded_events"] == []
        assert result["images"] == []
        assert result["instruction"] == instruction

    def test_process_keyboard_sequence(self):
        """Test processing keyboard event sequence."""
        events = [self.keyboard_event]
        instruction = "Type the letter A"

        result = self.builder.process_event_sequence(events, instruction)

        assert len(result["encoded_events"]) == 1
        assert result["images"] == []  # No images for keyboard events
        assert result["instruction"] == instruction

    def test_process_mouse_sequence(self):
        """Test processing mouse event sequence."""
        events = [self.mouse_event]
        instruction = "Click the button"

        result = self.builder.process_event_sequence(events, instruction)

        assert len(result["encoded_events"]) == 1
        assert result["images"] == []  # No images for mouse events
        assert result["instruction"] == instruction

    def test_process_mixed_sequence(self):
        """Test processing mixed event sequence."""
        events = [self.keyboard_event, self.mouse_event]
        instruction = "Type A then click the button"

        result = self.builder.process_event_sequence(events, instruction)

        assert len(result["encoded_events"]) == 2
        assert result["images"] == []
        assert result["instruction"] == instruction

    def test_process_with_custom_instruction(self):
        """Test processing with custom instruction."""
        events = [self.keyboard_event]
        custom_instruction = "Custom task instruction"

        result = self.builder.process_event_sequence(events, instruction=custom_instruction)

        assert result["instruction"] == custom_instruction

    def test_process_screen_sequence_basic(self):
        """Test processing screen event sequence (basic test without actual image loading)."""
        events = [self.screen_event]
        instruction = "Look at the screen and perform the task"

        # This will process the screen event but won't load actual images
        # since we don't have the actual video file
        result = self.builder.process_event_sequence(events, instruction)

        assert len(result["encoded_events"]) == 1
        # Images list may be empty if lazy_load fails (which is expected in tests)
        assert isinstance(result["images"], list)
        assert result["instruction"] == instruction

    def test_process_batch(self):
        """Test batch processing of event sequences."""
        sequences = [[self.keyboard_event], [self.mouse_event]]

        instructions = ["Instruction 1", "Instruction 2"]

        results = self.builder.process_batch(sequences, instructions)

        assert len(results) == 2
        assert results[0]["instruction"] == "Instruction 1"
        assert results[1]["instruction"] == "Instruction 2"
        assert "action_sequence" not in results[0]
        assert "action_sequence" not in results[1]

    def test_process_batch_validation(self):
        """Test batch processing validation."""
        sequences = [[self.keyboard_event], [self.mouse_event]]
        instructions = ["Instruction 1"]  # Only one instruction for two sequences

        # Should raise error when lengths don't match
        with pytest.raises(ValueError, match="Number of sequences .* must match number of instructions"):
            self.builder.process_batch(sequences, instructions)

    def test_create_huggingface_dataset(self):
        """Test HuggingFace dataset creation."""
        sequences = [[self.keyboard_event], [self.mouse_event]]
        instructions = ["Type A", "Click button"]

        hf_dataset = self.builder.create_huggingface_dataset(sequences, instructions)

        assert len(hf_dataset) == 2

        # Check required fields for VLA training
        for sample in hf_dataset:
            assert "encoded_events" in sample
            assert "images" in sample
            assert "instruction" in sample
            assert "action_sequence" not in sample  # No longer included

            assert isinstance(sample["encoded_events"], list)
            assert isinstance(sample["images"], list)
            assert isinstance(sample["instruction"], str)

    def test_error_handling_invalid_event(self):
        """Test error handling for invalid events."""
        invalid_event = {
            "topic": "invalid",
            "timestamp_ns": 1000000000,
            "message_type": "invalid.type",
            "msg": b"invalid json",
        }

        # Should not crash, but may produce warnings
        instruction = "Handle invalid event"
        result = self.builder.process_event_sequence([invalid_event], instruction)

        # Should still return valid structure
        assert "encoded_events" in result
        assert "images" in result
        assert "instruction" in result
        assert result["instruction"] == instruction


class TestVLMDatasetBuilderIntegration:
    """Integration tests for VLMDatasetBuilder."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from events to VLM format."""
        builder = VLMDatasetBuilder(drop_file_path=True)

        # Sample workflow
        event_sequences = [
            [
                {
                    "file_path": "/test/session.mcap",
                    "topic": "keyboard",
                    "timestamp_ns": 1000000000,
                    "message_type": "owa.env.desktop.msg.KeyboardEvent",
                    "msg": b'{"event_type":"press","vk":65}',
                }
            ]
        ]

        # Process to VLA format
        instructions = ["Type the letter A"]
        vlm_data = builder.create_huggingface_dataset(event_sequences, instructions)

        # Verify output format
        assert len(vlm_data) == 1
        sample = vlm_data[0]

        # Should have all required fields for VLA training
        assert len(sample["encoded_events"]) > 0
        assert isinstance(sample["images"], list)
        assert len(sample["instruction"]) > 0
        assert "action_sequence" not in sample  # No longer included

        # Encoded events should not contain file_path (due to drop_file_path=True)
        encoded_text = sample["encoded_events"][0]
        assert "file_path" not in encoded_text or "<DROPPED>" in encoded_text
