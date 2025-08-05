#!/usr/bin/env python3
"""
Test for owa.data.datasets - focuses on essential functionality and configuration sanity.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from owa.data.datasets import (
    DatasetConfig,
    DatasetStage,
    create_transform,
    list_datasets,
)
from owa.data.datasets.transforms import (
    FSLTransformConfig,
    create_fsl_transform,
    resolve_episode_path,
)


class TestDatasetConfig:
    """Test dataset configuration functionality."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = DatasetConfig()
        assert config.stage == DatasetStage.UNKNOWN
        assert config.mcap_root_directory is None

    def test_custom_config(self):
        """Test custom configuration creation."""
        config = DatasetConfig(stage=DatasetStage.EVENT, mcap_root_directory="/path/to/mcap")
        assert config.stage == DatasetStage.EVENT
        assert config.mcap_root_directory == "/path/to/mcap"

    def test_config_serialization(self):
        """Test configuration serialization to/from dict."""
        config = DatasetConfig(stage=DatasetStage.BINNED, mcap_root_directory="/test/path")

        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["stage"] == "binned"
        assert config_dict["mcap_root_directory"] == "/test/path"

        # Test from_dict
        restored_config = DatasetConfig.from_dict(config_dict)
        assert restored_config.stage == DatasetStage.BINNED
        assert restored_config.mcap_root_directory == "/test/path"

    def test_config_json_io(self):
        """Test configuration JSON file I/O."""
        config = DatasetConfig(stage=DatasetStage.TOKENIZED, mcap_root_directory="/json/test")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            # Test to_json
            config.to_json(json_path)

            # Verify file contents
            with open(json_path, "r") as f:
                data = json.load(f)
            assert data["stage"] == "tokenized"
            assert data["mcap_root_directory"] == "/json/test"

            # Test from_json
            restored_config = DatasetConfig.from_json(json_path)
            assert restored_config.stage == DatasetStage.TOKENIZED
            assert restored_config.mcap_root_directory == "/json/test"
        finally:
            Path(json_path).unlink(missing_ok=True)


class TestDatasetStage:
    """Test dataset stage enumeration."""

    def test_stage_values(self):
        """Test all stage values are correct."""
        assert DatasetStage.EVENT == "event"
        assert DatasetStage.BINNED == "binned"
        assert DatasetStage.TOKENIZED == "tokenized"
        assert DatasetStage.FSL == "fsl"
        assert DatasetStage.UNKNOWN == "unknown"

    def test_stage_comparison(self):
        """Test stage comparison and string conversion."""
        stage = DatasetStage.EVENT
        assert stage == "event"
        assert str(stage) == "event"


class TestDatasetDiscovery:
    """Test dataset discovery functionality."""

    @patch("owa.data.datasets.discovery.hf_list_datasets")
    def test_list_datasets_success(self, mock_hf_list):
        """Test successful dataset listing."""
        # Mock HuggingFace response
        mock_dataset = Mock()
        mock_dataset.id = "test-org/test-dataset"
        mock_hf_list.return_value = [mock_dataset]

        datasets = list_datasets()
        assert datasets == ["test-org/test-dataset"]
        mock_hf_list.assert_called_once_with(filter="OWA")

    @patch("owa.data.datasets.discovery.hf_list_datasets")
    def test_list_datasets_custom_filter(self, mock_hf_list):
        """Test dataset listing with custom filter."""
        mock_hf_list.return_value = []

        datasets = list_datasets(format_filter="CUSTOM")
        assert datasets == []
        mock_hf_list.assert_called_once_with(filter="CUSTOM")

    @patch("owa.data.datasets.discovery.hf_list_datasets")
    def test_list_datasets_error_handling(self, mock_hf_list):
        """Test error handling in dataset listing."""
        mock_hf_list.side_effect = Exception("Network error")

        with patch("builtins.print") as mock_print:
            datasets = list_datasets()
            assert datasets == []
            mock_print.assert_called_once()
            assert "Warning: Could not list datasets" in mock_print.call_args[0][0]

    @pytest.mark.network
    def test_list_datasets_network(self):
        """Test actual dataset listing from HuggingFace (requires network)."""
        datasets = list_datasets()
        assert isinstance(datasets, list)
        # Should find the example dataset
        assert any("open-world-agents/example_dataset" in ds for ds in datasets)


class TestTransforms:
    """Test transform creation and functionality."""

    def test_create_transform_event(self):
        """Test event transform creation."""
        transform = create_transform(stage=DatasetStage.EVENT, mcap_root_directory="/test/mcap")
        assert callable(transform)

    def test_create_transform_binned(self):
        """Test binned transform creation."""
        transform = create_transform(stage=DatasetStage.BINNED, mcap_root_directory="/test/mcap")
        assert callable(transform)

    def test_create_transform_tokenized(self):
        """Test tokenized transform creation."""
        transform = create_transform(stage=DatasetStage.TOKENIZED, mcap_root_directory="/test/mcap")
        assert callable(transform)

    def test_create_transform_fsl(self):
        """Test FSL transform creation."""
        transform = create_transform(stage=DatasetStage.FSL, mcap_root_directory="/test/mcap")
        assert callable(transform)

    def test_create_transform_invalid_stage(self):
        """Test error handling for invalid stage."""
        with pytest.raises(ValueError, match="Unknown dataset stage"):
            create_transform(stage="invalid_stage", mcap_root_directory="/test/mcap")


class TestFSLTransform:
    """Test FSL transform specific functionality."""

    def test_fsl_transform_config(self):
        """Test FSL transform configuration."""
        config = FSLTransformConfig(load_images=True, mcap_root_directory="/test/mcap", pad_token_id=42)
        assert config.load_images is True
        assert config.mcap_root_directory == "/test/mcap"
        assert config.pad_token_id == 42

    def test_fsl_transform_creation(self):
        """Test FSL transform creation with config."""
        transform = create_fsl_transform(mcap_root_directory="/test/mcap", load_images=False, pad_token_id=123)
        assert callable(transform)

    def test_resolve_episode_path_absolute(self):
        """Test episode path resolution with absolute path."""
        absolute_path = "/absolute/path/episode.mcap"
        resolved = resolve_episode_path(absolute_path, "/mcap/root")
        assert resolved == absolute_path

    def test_resolve_episode_path_relative(self):
        """Test episode path resolution with relative path."""
        relative_path = "relative/episode.mcap"
        mcap_root = "/mcap/root"
        resolved = resolve_episode_path(relative_path, mcap_root)
        expected = "/mcap/root/relative/episode.mcap"
        assert resolved == expected

    def test_resolve_episode_path_no_root(self):
        """Test episode path resolution without mcap root raises error."""
        path = "some/path.mcap"
        with pytest.raises(ValueError, match="mcap_root_directory required"):
            resolve_episode_path(path, None)


class TestDatasetIntegration:
    """Test integration between different dataset components."""

    def test_config_stage_transform_consistency(self):
        """Test that config stages match transform creation."""
        for stage in [DatasetStage.EVENT, DatasetStage.BINNED, DatasetStage.TOKENIZED, DatasetStage.FSL]:
            config = DatasetConfig(stage=stage, mcap_root_directory="/test")

            # Should be able to create transform for each stage
            transform = create_transform(stage=stage, mcap_root_directory=config.mcap_root_directory)
            assert callable(transform)

    def test_config_serialization_roundtrip(self):
        """Test complete config serialization roundtrip."""
        original_config = DatasetConfig(stage=DatasetStage.FSL, mcap_root_directory="/complex/path/with spaces")

        # Serialize to dict and back
        config_dict = original_config.to_dict()
        restored_config = DatasetConfig.from_dict(config_dict)

        # Should be identical
        assert restored_config.stage == original_config.stage
        assert restored_config.mcap_root_directory == original_config.mcap_root_directory


class TestDatasetLoading:
    """Test dataset loading functionality."""

    def test_load_dataset_not_implemented(self):
        """Test that load_dataset raises NotImplementedError."""
        from owa.data.datasets.load import load_dataset

        with pytest.raises(NotImplementedError):
            load_dataset("test/path")

    @pytest.mark.network
    def test_load_example_dataset_network(self):
        """Test loading example dataset from HuggingFace (requires network)."""
        from owa.data.datasets import load_dataset

        # This will download a small dataset from HuggingFace (network required)
        # For now, catch NotImplementedError since load_dataset is not fully implemented
        with pytest.raises(NotImplementedError):
            ds = load_dataset("open-world-agents/example_dataset")  # noqa: F841
            # Future: assert "train" in ds or len(ds) > 0

    @patch("owa.data.datasets.load.fsspec")
    @patch("owa.data.datasets.load.url_to_fs")
    def test_load_from_disk_imports(self, mock_url_to_fs, mock_fsspec):
        """Test that load_from_disk function can be imported and has correct signature."""
        from owa.data.datasets.load import load_from_disk

        # Function should be callable
        assert callable(load_from_disk)

        # Should have the expected parameters (test by checking function signature)
        import inspect

        sig = inspect.signature(load_from_disk)
        expected_params = {"dataset_path", "keep_in_memory", "storage_options"}
        actual_params = set(sig.parameters.keys())
        assert expected_params.issubset(actual_params)


class TestDatasetFilesystem:
    """Test dataset filesystem operations."""

    def test_filesystem_module_imports(self):
        """Test that filesystem module can be imported."""
        try:
            from owa.data.datasets import filesystem

            assert filesystem is not None
        except ImportError:
            pytest.skip("Filesystem module not available")

    def test_dataset_module_imports(self):
        """Test that dataset module can be imported."""
        try:
            from owa.data.datasets.dataset import Dataset, DatasetDict

            assert Dataset is not None
            assert DatasetDict is not None
        except ImportError:
            pytest.skip("Dataset module not available")


class TestTransformUtils:
    """Test transform utility functions."""

    def test_resolve_episode_path_edge_cases(self):
        """Test edge cases for episode path resolution."""
        # Empty path
        assert resolve_episode_path("", "/mcap/root") == ""

        # Path with spaces
        path_with_spaces = "path with spaces/episode.mcap"
        mcap_root = "/mcap/root"
        resolved = resolve_episode_path(path_with_spaces, mcap_root)
        expected = "/mcap/root/path with spaces/episode.mcap"
        assert resolved == expected

    def test_resolve_episode_path_windows_style(self):
        """Test Windows-style absolute path handling."""
        # On Linux, Windows paths are treated as relative, so test the actual behavior
        windows_path = "C:\\absolute\\path\\episode.mcap"
        resolved = resolve_episode_path(windows_path, "/mcap/root")
        # On Linux, this will be treated as relative and joined with mcap_root
        expected = "/mcap/root/C:\\absolute\\path\\episode.mcap"
        assert resolved == expected


class TestDatasetStageIntegration:
    """Test integration between stages and other components."""

    def test_all_stages_have_string_values(self):
        """Test that all stages have proper string values."""
        stages = [
            DatasetStage.EVENT,
            DatasetStage.BINNED,
            DatasetStage.TOKENIZED,
            DatasetStage.FSL,
            DatasetStage.UNKNOWN,
        ]

        for stage in stages:
            assert isinstance(stage.value, str)
            assert len(stage.value) > 0
            assert stage.value.islower()

    def test_stage_enum_completeness(self):
        """Test that we have all expected stages."""
        expected_stages = {"event", "binned", "tokenized", "fsl", "unknown"}
        actual_stages = {stage.value for stage in DatasetStage}
        assert actual_stages == expected_stages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
