#!/usr/bin/env python3
"""
Test script for Event and Binned Dataset transforms.

This script provides basic validation that the transforms work correctly.
"""

from unittest.mock import Mock, patch

from owa.data.transforms import (
    create_binned_dataset_transform,
    create_event_dataset_transform,
)


def test_event_dataset_transform_single():
    """Test create_event_dataset_transform with a single example."""
    # Mock the dependencies
    with (
        patch("owa.data.transforms.McapMessage") as mock_mcap_msg,
        patch("owa.data.transforms.ScreenCaptured") as mock_screen_captured,
        patch("owa.data.transforms.create_encoder") as mock_create_encoder,
    ):
        # Setup mocks
        mock_encoder = Mock()
        mock_encoder.encode.return_value = ("encoded_action_text", [])
        mock_create_encoder.return_value = mock_encoder

        mock_mcap_instance = Mock()
        mock_mcap_msg.model_validate_json.return_value = mock_mcap_instance

        mock_screen_instance = Mock()
        mock_screen_instance.to_pil_image.return_value = Mock()  # Mock PIL Image
        mock_screen_captured.model_validate.return_value = mock_screen_instance

        # Create transform function
        transform = create_event_dataset_transform(encoder_type="hierarchical")

        # Test action event
        action_example = {
            "file_path": "/test/file.mcap",
            "topic": "keyboard",
            "timestamp_ns": 123456789,
            "message_type": "KeyboardEvent",
            "mcap_message": b'{"test": "data"}',
        }

        result = transform(action_example)

        # Verify result structure
        assert "encoded_event" in result
        assert "image" in result
        assert result["encoded_event"] == "encoded_action_text"
        assert result["image"] is None  # Should be None for action events

        print("✓ create_event_dataset_transform single example test passed")


def test_binned_dataset_transform_single():
    """Test create_binned_dataset_transform with a single example."""
    # Mock the dependencies
    with (
        patch("owa.data.transforms.McapMessage") as mock_mcap_msg,
        patch("owa.data.transforms.ScreenCaptured") as mock_screen_captured,
        patch("owa.data.transforms.create_encoder") as mock_create_encoder,
    ):
        # Setup mocks
        mock_encoder = Mock()
        mock_encoder.encode.return_value = ("encoded_action_text", [])
        mock_create_encoder.return_value = mock_encoder

        mock_mcap_instance = Mock()
        mock_mcap_msg.model_validate_json.return_value = mock_mcap_instance

        mock_screen_instance = Mock()
        mock_screen_instance.to_pil_image.return_value = Mock()  # Mock PIL Image
        mock_screen_captured.model_validate.return_value = mock_screen_instance

        # Create transform function
        transform = create_binned_dataset_transform(encoder_type="hierarchical", instruction="Test instruction")

        # Test binned example
        binned_example = {
            "file_path": "/test/file.mcap",
            "bin_idx": 0,
            "timestamp_ns": 123456789,
            "state": [b'{"screen": "data"}'],
            "actions": [b'{"action": "data"}'],
        }

        result = transform(binned_example)

        # Verify result structure
        assert "instruction" in result
        assert "images" in result
        assert "encoded_events" in result
        assert result["instruction"] == "Test instruction"
        assert isinstance(result["images"], list)
        assert isinstance(result["encoded_events"], list)

        print("✓ create_binned_dataset_transform single example test passed")


def test_convenience_functions():
    """Test the convenience functions work correctly."""
    # Test create_event_dataset_transform
    event_transform = create_event_dataset_transform(
        encoder_type="hierarchical", load_images=True, encode_actions=True
    )
    assert callable(event_transform)

    # Test create_binned_dataset_transform
    binned_transform = create_binned_dataset_transform(
        encoder_type="json", instruction="Custom instruction", load_images=False, encode_actions=True
    )
    assert callable(binned_transform)

    print("✓ Convenience functions test passed")


def test_batch_processing():
    """Test batch processing works correctly."""
    # Mock the dependencies
    with (
        patch("owa.data.transforms.McapMessage") as mock_mcap_msg,
        patch("owa.data.transforms.create_encoder") as mock_create_encoder,
    ):
        # Setup mocks
        mock_encoder = Mock()
        mock_encoder.encode.return_value = ("encoded_action_text", [])
        mock_create_encoder.return_value = mock_encoder

        mock_mcap_instance = Mock()
        mock_mcap_msg.model_validate_json.return_value = mock_mcap_instance

        # Create transform function
        transform = create_event_dataset_transform(encoder_type="hierarchical")

        # Test batch of action events
        batch_examples = {
            "file_path": ["/test/file1.mcap", "/test/file2.mcap"],
            "topic": ["keyboard", "mouse"],
            "timestamp_ns": [123456789, 123456790],
            "message_type": ["KeyboardEvent", "MouseEvent"],
            "mcap_message": [b'{"test": "data1"}', b'{"test": "data2"}'],
        }

        result = transform(batch_examples)

        # Verify result structure
        assert "encoded_event" in result
        assert "image" in result
        assert len(result["encoded_event"]) == 2
        assert len(result["image"]) == 2

        print("✓ Batch processing test passed")


if __name__ == "__main__":
    print("Running basic validation tests for transforms...")
    print("=" * 50)

    try:
        test_event_dataset_transform_single()
        test_binned_dataset_transform_single()
        test_convenience_functions()
        test_batch_processing()

        print("=" * 50)
        print("✅ All tests passed! Transforms are working correctly.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
