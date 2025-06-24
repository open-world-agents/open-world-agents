"""
Tests for OWADataset class.

This module contains tests for the new OWADataset which provides
PyTorch Dataset interface for MLLM datasets with lazy image loading.
"""

from datasets import Dataset, Features, Sequence, Value

from owa.data import OWADataset
from owa.msgs.desktop.screen import ScreenCaptured


class TestOWADataset:
    """Test suite for OWADataset class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock ScreenCaptured objects and serialize them
        screen1 = ScreenCaptured(
            utc_ns=3000000000, path="test1.mkv", pts=1000000000, original_shape=(1920, 1080), shape=(1920, 1080)
        )
        screen2 = ScreenCaptured(
            utc_ns=4000000000, path="test2.mkv", pts=2000000000, original_shape=(1920, 1080), shape=(1920, 1080)
        )

        # Serialize to bytes (as expected by the new format)
        screen1_bytes = screen1.model_dump_json().encode("utf-8")
        screen2_bytes = screen2.model_dump_json().encode("utf-8")

        # Create sample MLLM dataset
        self.sample_data = [
            {
                "instruction": "Complete the computer task",
                "encoded_events": [
                    "<EVENT_START>{'topic': 'keyboard', 'timestamp_ns': 1000000000}<EVENT_END>",
                    "<EVENT_START>{'topic': 'mouse', 'timestamp_ns': 2000000000}<EVENT_END>",
                ],
                "image_refs": [screen1_bytes, screen2_bytes],
                "metadata": {
                    "file_path": "/test/session.mcap",
                    "bin_idx": 0,
                    "timestamp_ns": 1000000000,
                    "num_actions": 2,
                },
            }
        ]

        # Create HuggingFace dataset with updated features for bytes
        features = Features(
            {
                "instruction": Value("string"),
                "encoded_events": Sequence(Value("string")),
                "image_refs": Sequence(Value("binary")),  # Changed to binary for serialized bytes
                "metadata": {
                    "file_path": Value("string"),
                    "bin_idx": Value("int32"),
                    "timestamp_ns": Value("int64"),
                    "num_actions": Value("int32"),
                },
            }
        )

        self.mllm_dataset = Dataset.from_list(self.sample_data, features=features)

    def test_initialization(self):
        """Test OWADataset initialization."""
        dataset = OWADataset(self.mllm_dataset)
        assert dataset.dataset == self.mllm_dataset

    def test_dataset_length(self):
        """Test dataset length."""
        dataset = OWADataset(self.mllm_dataset)
        assert len(dataset) == 1

    def test_getitem_structure(self):
        """Test __getitem__ returns correct structure."""
        dataset = OWADataset(self.mllm_dataset)

        # Mock the image loading to avoid file system dependencies
        def mock_load_images(_image_refs, _metadata):
            import numpy as np
            from PIL import Image

            # Return mock PIL images
            test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            return [Image.fromarray(test_array, mode="RGB"), Image.fromarray(test_array, mode="RGB")]

        dataset._load_images = mock_load_images

        item = dataset[0]

        # Check structure
        assert "instruction" in item
        assert "images" in item
        assert "encoded_events" in item

        # Check content
        assert item["instruction"] == "Complete the computer task"
        assert isinstance(item["images"], list)
        assert len(item["images"]) == 2
        assert isinstance(item["encoded_events"], list)
        assert len(item["encoded_events"]) == 2

    def test_image_loading_failure_handling(self):
        """Test handling of image loading failures."""
        dataset = OWADataset(self.mllm_dataset)

        # Test with invalid image references (should handle gracefully)
        invalid_bytes = [b"invalid_json_data"]
        mock_metadata = {"file_path": "/test/session.mcap"}
        images = dataset._load_images(invalid_bytes, mock_metadata)

        # Should return empty list when images can't be loaded
        assert isinstance(images, list)
        # Length should be 0 since all images fail to load
        assert len(images) == 0

    def test_deserialization(self):
        """Test ScreenCaptured deserialization functionality."""
        dataset = OWADataset(self.mllm_dataset)

        # Test with valid serialized ScreenCaptured
        screen = ScreenCaptured(
            utc_ns=1000000000, path="test.mkv", pts=1000000000, original_shape=(1920, 1080), shape=(1920, 1080)
        )
        screen_bytes = screen.model_dump_json().encode("utf-8")

        # This test verifies deserialization works (actual image loading would fail without video file)
        try:
            deserialized = ScreenCaptured.model_validate_json(screen_bytes.decode("utf-8"))
            assert deserialized.utc_ns == screen.utc_ns
            assert deserialized.path == screen.path
            assert deserialized.pts == screen.pts
        except Exception:
            # If deserialization fails, that's a problem
            assert False, "ScreenCaptured deserialization should work"


class TestOWADatasetIntegration:
    """Integration tests for OWADataset."""

    def test_pytorch_dataset_interface(self):
        """Test that OWADataset properly implements PyTorch Dataset interface."""
        from torch.utils.data import Dataset as TorchDataset

        # Create sample MLLM dataset with serialized ScreenCaptured
        screen = ScreenCaptured(
            utc_ns=1000000000, path="test.mkv", pts=1000000000, original_shape=(1920, 1080), shape=(1920, 1080)
        )
        screen_bytes = screen.model_dump_json().encode("utf-8")

        sample_data = [
            {
                "instruction": "Test instruction",
                "encoded_events": ["event1", "event2"],
                "image_refs": [screen_bytes],
                "metadata": {"bin_idx": 0, "timestamp_ns": 1000000000, "num_actions": 2},
            }
        ]

        features = Features(
            {
                "instruction": Value("string"),
                "encoded_events": Sequence(Value("string")),
                "image_refs": Sequence(Value("binary")),
                "metadata": {
                    "bin_idx": Value("int32"),
                    "timestamp_ns": Value("int64"),
                    "num_actions": Value("int32"),
                },
            }
        )

        hf_dataset = Dataset.from_list(sample_data, features=features)
        owa_dataset = OWADataset(hf_dataset)

        # Test Dataset interface
        assert len(owa_dataset) == 1
        assert isinstance(owa_dataset, TorchDataset)

        # Mock image loading for testing
        def mock_load_images(_refs, _metadata):
            import numpy as np
            from PIL import Image

            test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            return [Image.fromarray(test_array, mode="RGB")]

        owa_dataset._load_images = mock_load_images

        item = owa_dataset[0]
        assert isinstance(item, dict)
        assert "instruction" in item
        assert "images" in item
        assert "encoded_events" in item

    def test_integration_with_empty_dataset(self):
        """Test behavior with empty dataset."""
        # Create empty dataset without features to avoid the schema mismatch
        empty_data = []
        hf_dataset = Dataset.from_list(empty_data)
        owa_dataset = OWADataset(hf_dataset)

        assert len(owa_dataset) == 0
