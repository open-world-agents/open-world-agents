"""
Tests for VLADataset class.

This module contains tests for the unified VLADataset which provides
PyTorch Dataset interface for both binned and MLLM datasets.
"""

from datasets import Dataset, Features, Sequence, Value

from mcap_owa.highlevel import McapMessage
from owa.data import HierarchicalEventEncoder, VLADataset
from owa.msgs.desktop.screen import ScreenCaptured


class TestVLADataset:
    """Test suite for VLADataset class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock ScreenCaptured objects and serialize them
        screen1 = ScreenCaptured(
            utc_ns=3000000000, path="test1.mkv", pts=1000000000, original_shape=(1920, 1080), shape=(1920, 1080)
        )
        screen2 = ScreenCaptured(
            utc_ns=4000000000, path="test2.mkv", pts=2000000000, original_shape=(1920, 1080), shape=(1920, 1080)
        )

        # Serialize to bytes (as expected by the MLLM format)
        screen1_bytes = screen1.model_dump_json().encode("utf-8")
        screen2_bytes = screen2.model_dump_json().encode("utf-8")

        # Create sample MLLM dataset
        self.mllm_sample_data = [
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

        # Create HuggingFace dataset with MLLM format
        mllm_features = Features(
            {
                "instruction": Value("string"),
                "encoded_events": Sequence(Value("string")),
                "image_refs": Sequence(Value("binary")),
                "metadata": {
                    "file_path": Value("string"),
                    "bin_idx": Value("int32"),
                    "timestamp_ns": Value("int64"),
                    "num_actions": Value("int32"),
                },
            }
        )

        self.mllm_dataset = Dataset.from_list(self.mllm_sample_data, features=mllm_features)

    def test_initialization_mllm_format(self):
        """Test VLADataset initialization with MLLM format."""
        dataset = VLADataset(self.mllm_dataset)
        assert dataset.dataset == self.mllm_dataset
        assert dataset.is_mllm_format == True

    def test_dataset_length(self):
        """Test dataset length."""
        dataset = VLADataset(self.mllm_dataset)
        assert len(dataset) == 1

    def test_getitem_structure_mllm(self):
        """Test __getitem__ returns correct structure for MLLM format."""
        dataset = VLADataset(self.mllm_dataset)

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

    def test_encoder_configuration(self):
        """Test different encoder configurations."""
        # Test with different encoder types
        for encoder_type in ["hierarchical", "json", "flat"]:
            dataset = VLADataset(self.mllm_dataset, encoder_type=encoder_type)
            assert dataset.encoder is not None

    def test_custom_instruction(self):
        """Test custom instruction override."""
        custom_instruction = "Custom task instruction"
        dataset = VLADataset(self.mllm_dataset, instruction=custom_instruction)

        # Mock image loading
        def mock_load_images(_image_refs, _metadata):
            return []

        dataset._load_images = mock_load_images

        item = dataset[0]
        assert item["instruction"] == custom_instruction

    def test_caching_functionality(self):
        """Test sample caching functionality."""
        dataset = VLADataset(self.mllm_dataset, cache_samples=True)

        # Mock image loading
        def mock_load_images(_image_refs, _metadata):
            return []

        dataset._load_images = mock_load_images

        # First access should populate cache
        item1 = dataset[0]
        assert 0 in dataset._cache

        # Second access should use cache
        item2 = dataset[0]
        assert item1 == item2


class TestVLADatasetIntegration:
    """Integration tests for VLADataset."""

    def test_pytorch_dataset_interface(self):
        """Test that VLADataset properly implements PyTorch Dataset interface."""
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
        vla_dataset = VLADataset(hf_dataset)

        # Test Dataset interface
        assert len(vla_dataset) == 1
        assert isinstance(vla_dataset, TorchDataset)

        # Mock image loading for testing
        def mock_load_images(_refs, _metadata):
            import numpy as np
            from PIL import Image

            test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            return [Image.fromarray(test_array, mode="RGB")]

        vla_dataset._load_images = mock_load_images

        item = vla_dataset[0]
        assert isinstance(item, dict)
        assert "instruction" in item
        assert "images" in item
        assert "encoded_events" in item

    def test_integration_with_empty_dataset(self):
        """Test behavior with empty dataset."""
        # Create empty dataset without features to avoid the schema mismatch
        empty_data = []
        hf_dataset = Dataset.from_list(empty_data)
        vla_dataset = VLADataset(hf_dataset)

        assert len(vla_dataset) == 0
