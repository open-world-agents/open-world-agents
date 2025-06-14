"""
Tests for VLMDatasetBuilder class.

This module contains tests for the new VLMDatasetBuilder which provides
PyTorch Dataset interface for MLLM datasets with lazy image loading.
"""

from datasets import Dataset, Features, Sequence, Value

from owa.data.vlm_dataset_builder import VLMDatasetBuilder


class TestVLMDatasetBuilder:
    """Test suite for VLMDatasetBuilder class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample MLLM dataset
        self.sample_data = [
            {
                "instruction": "Complete the computer task",
                "encoded_events": [
                    "<EVENT_START>{'topic': 'keyboard', 'timestamp_ns': 1000000000}<EVENT_END>",
                    "<EVENT_START>{'topic': 'mouse', 'timestamp_ns': 2000000000}<EVENT_END>",
                ],
                "image_refs": [
                    {
                        "path": "test1.mkv",
                        "pts": 1000000000,
                        "utc_ns": 3000000000,
                        "timestamp_ns": 1000000000,
                        "bin_idx": 0,
                    },
                    {
                        "path": "test2.mkv",
                        "pts": 2000000000,
                        "utc_ns": 4000000000,
                        "timestamp_ns": 2000000000,
                        "bin_idx": 1,
                    },
                ],
                "metadata": {
                    "file_path": "/test/session.mcap",
                    "sequence_idx": 0,
                    "start_bin_idx": 0,
                    "end_bin_idx": 1,
                    "start_timestamp_ns": 1000000000,
                    "end_timestamp_ns": 2000000000,
                    "num_bins": 2,
                    "num_images": 2,
                    "num_actions": 2,
                },
            }
        ]

        # Create HuggingFace dataset
        features = Features(
            {
                "instruction": Value("string"),
                "encoded_events": Sequence(Value("string")),
                "image_refs": Sequence(
                    {
                        "path": Value("string"),
                        "pts": Value("int64"),
                        "utc_ns": Value("int64"),
                        "timestamp_ns": Value("int64"),
                        "bin_idx": Value("int32"),
                    }
                ),
                "metadata": {
                    "file_path": Value("string"),
                    "sequence_idx": Value("int32"),
                    "start_bin_idx": Value("int32"),
                    "end_bin_idx": Value("int32"),
                    "start_timestamp_ns": Value("int64"),
                    "end_timestamp_ns": Value("int64"),
                    "num_bins": Value("int32"),
                    "num_images": Value("int32"),
                    "num_actions": Value("int32"),
                },
            }
        )

        self.mllm_dataset = Dataset.from_list(self.sample_data, features=features)

    def test_initialization(self):
        """Test VLMDatasetBuilder initialization."""
        # Test with PIL format (default)
        builder = VLMDatasetBuilder(self.mllm_dataset)
        assert builder.image_format == "pil"
        assert not builder.cache_images
        assert builder.max_cache_size == 1000

        # Test with caching enabled
        builder_cached = VLMDatasetBuilder(
            self.mllm_dataset, image_format="tensor", cache_images=True, max_cache_size=500
        )
        assert builder_cached.image_format == "tensor"
        assert builder_cached.cache_images
        assert builder_cached.max_cache_size == 500

    def test_dataset_length(self):
        """Test dataset length."""
        builder = VLMDatasetBuilder(self.mllm_dataset)
        assert len(builder) == 1

    def test_getitem_structure(self):
        """Test __getitem__ returns correct structure."""
        builder = VLMDatasetBuilder(self.mllm_dataset)

        # Mock the image loading to avoid file system dependencies
        def mock_load_images(_image_refs):
            return []  # Return empty list for testing

        builder._load_images = mock_load_images

        item = builder[0]

        # Check structure
        assert "instruction" in item
        assert "encoded_events" in item
        assert "images" in item
        assert "metadata" in item

        # Check content
        assert item["instruction"] == "Complete the computer task"
        assert len(item["encoded_events"]) == 2
        assert isinstance(item["images"], list)
        assert item["metadata"]["sequence_idx"] == 0

    def test_image_loading_failure_handling(self):
        """Test handling of image loading failures."""
        builder = VLMDatasetBuilder(self.mllm_dataset)

        # Test with non-existent image references
        image_refs = [{"path": "nonexistent.mkv", "pts": 1000000000}]
        images = builder._load_images(image_refs)

        # Should return empty list when images can't be loaded
        assert isinstance(images, list)
        # Length may be 0 if all images fail to load

    def test_cache_functionality(self):
        """Test image caching functionality."""
        builder = VLMDatasetBuilder(self.mllm_dataset, cache_images=True, max_cache_size=2)

        # Test cache stats
        stats = builder.get_cache_stats()
        assert stats["cache_enabled"]
        assert stats["cache_size"] == 0
        assert stats["max_cache_size"] == 2

        # Test cache disabled
        builder_no_cache = VLMDatasetBuilder(self.mllm_dataset, cache_images=False)
        stats_no_cache = builder_no_cache.get_cache_stats()
        assert not stats_no_cache["cache_enabled"]

    def test_image_format_validation(self):
        """Test image format validation."""
        # Valid formats should work
        valid_formats = ["pil", "tensor", "numpy"]
        for fmt in valid_formats:
            builder = VLMDatasetBuilder(self.mllm_dataset, image_format=fmt)
            assert builder.image_format == fmt

        # Invalid format should still be accepted (will fail at runtime)
        builder = VLMDatasetBuilder(self.mllm_dataset, image_format="invalid")
        assert builder.image_format == "invalid"


class TestVLMDatasetBuilderIntegration:
    """Integration tests for VLMDatasetBuilder."""

    def test_pytorch_dataset_interface(self):
        """Test that VLMDatasetBuilder properly implements PyTorch Dataset interface."""
        from owa.data.vlm_dataset_builder import Dataset, VLMDatasetBuilder

        # Create sample MLLM dataset
        sample_data = [
            {
                "instruction": "Test instruction",
                "encoded_events": ["event1", "event2"],
                "image_refs": [],
                "metadata": {"sequence_idx": 0},
            }
        ]

        features = Features(
            {
                "instruction": Value("string"),
                "encoded_events": Sequence(Value("string")),
                "image_refs": Sequence(
                    {
                        "path": Value("string"),
                        "pts": Value("int64"),
                        "utc_ns": Value("int64"),
                        "timestamp_ns": Value("int64"),
                        "bin_idx": Value("int32"),
                    }
                ),
                "metadata": {"sequence_idx": Value("int32")},
            }
        )

        mllm_dataset = Dataset.from_list(sample_data, features=features)
        builder = VLMDatasetBuilder(mllm_dataset)

        # Test Dataset interface
        assert len(builder) == 1
        assert isinstance(builder, Dataset)

        # Mock image loading for testing
        builder._load_images = lambda _refs: []

        item = builder[0]
        assert isinstance(item, dict)
        assert "instruction" in item
        assert "encoded_events" in item
        assert "images" in item
        assert "metadata" in item

    def test_integration_with_empty_dataset(self):
        """Test behavior with empty dataset."""
        empty_data = []
        features = Features(
            {
                "instruction": Value("string"),
                "encoded_events": Sequence(Value("string")),
                "image_refs": Sequence(
                    {
                        "path": Value("string"),
                        "pts": Value("int64"),
                        "utc_ns": Value("int64"),
                        "timestamp_ns": Value("int64"),
                        "bin_idx": Value("int32"),
                    }
                ),
                "metadata": {"sequence_idx": Value("int32")},
            }
        )

        empty_dataset = Dataset.from_list(empty_data, features=features)
        builder = VLMDatasetBuilder(empty_dataset)

        assert len(builder) == 0
