"""Tests for event to FSL processing functionality."""

from unittest.mock import Mock, patch

import pytest

from owa.data.datasets.fsl_dataset import FSLDatasetConfig
from owa.data.processing.event_to_fsl import (
    EventToFSLConfig,
    create_fsl_dataset_from_events,
    create_fsl_dataset_from_events_and_save,
)


class TestEventToFSLConfig:
    """Test the EventToFSLConfig dataclass."""

    def test_config_creation(self):
        """Test that EventToFSLConfig can be created with required parameters."""
        config = EventToFSLConfig(
            tokenizer_name="microsoft/DialoGPT-medium",
            episode_tokenize_config={},
            fsl_dataset=FSLDatasetConfig(),
            num_proc=16,
            fsl_workers=4,
        )

        assert config.tokenizer_name == "microsoft/DialoGPT-medium"
        assert config.episode_tokenize_config == {}
        assert isinstance(config.fsl_dataset, FSLDatasetConfig)
        assert config.num_proc == 16
        assert config.fsl_workers == 4

    def test_config_defaults(self):
        """Test that EventToFSLConfig has proper defaults."""
        config = EventToFSLConfig(tokenizer_name="test-tokenizer")

        assert config.tokenizer_name == "test-tokenizer"
        assert config.episode_tokenize_config == {}
        assert isinstance(config.fsl_dataset, FSLDatasetConfig)
        assert config.num_proc == 32
        assert config.fsl_workers == 4


class TestCreateFSLDatasetFromEvents:
    """Test the high-level create_fsl_dataset_from_events function."""

    def test_import_availability(self):
        """Test that the function can be imported from the event_to_fsl module."""
        from owa.data.processing.event_to_fsl import create_fsl_dataset_from_events

        assert callable(create_fsl_dataset_from_events)

    @patch("owa.data.processing.event_to_fsl.AutoTokenizer")
    @patch("owa.data.processing.event_to_fsl.EpisodeTokenizer")
    @patch("owa.data.processing.event_to_fsl.precompute_fsl_dataset")
    def test_create_fsl_dataset_basic(self, mock_precompute, mock_episode_tokenizer, mock_auto_tokenizer):
        """Test basic functionality of create_fsl_dataset_from_events."""
        # Mock the dataset
        mock_dataset = Mock()
        mock_dataset.owa_config.stage = "EVENT"
        mock_dataset.__len__ = Mock(return_value=100)  # Add len() support

        # Mock the tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock the episode tokenizer
        mock_ep_tokenizer_instance = Mock()
        mock_episode_tokenizer.from_transformers.return_value = mock_ep_tokenizer_instance

        # Mock tokenized dataset
        mock_tokenized_dataset = Mock()
        mock_tokenized_dataset.__len__ = Mock(return_value=80)  # Add len() support
        mock_ep_tokenizer_instance.tokenize_event_dataset.return_value = mock_tokenized_dataset

        # Mock FSL dataset
        mock_fsl_dataset = Mock()
        mock_fsl_dataset.__len__ = Mock(return_value=60)  # Add len() support
        mock_precompute.return_value = mock_fsl_dataset

        # Create config
        config = EventToFSLConfig(
            tokenizer_name="test-tokenizer",
            episode_tokenize_config={},
            fsl_dataset=FSLDatasetConfig(),
            num_proc=4,
            fsl_workers=2,
        )

        # Test the function
        with patch("owa.data.processing.event_to_fsl.DatasetStage") as mock_stage:
            mock_stage.EVENT = "EVENT"
            result = create_fsl_dataset_from_events(mock_dataset, config)

        # Verify the result
        assert result == mock_fsl_dataset

        # Verify function calls
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("test-tokenizer")
        mock_episode_tokenizer.from_transformers.assert_called_once_with("test-tokenizer")
        mock_ep_tokenizer_instance.prepare_model.assert_called_once_with(tokenizer=mock_tokenizer)
        mock_ep_tokenizer_instance.tokenize_event_dataset.assert_called_once()
        mock_precompute.assert_called_once()

    def test_create_fsl_dataset_wrong_stage(self):
        """Test that function raises error for wrong dataset stage."""
        # Mock dataset with wrong stage
        mock_dataset = Mock()
        mock_dataset.owa_config.stage = "WRONG_STAGE"

        config = EventToFSLConfig(tokenizer_name="test-tokenizer")

        # Test that it raises ValueError
        with patch("owa.data.processing.event_to_fsl.DatasetStage") as mock_stage:
            mock_stage.EVENT = "EVENT"
            with pytest.raises(ValueError, match="Input dataset must be EVENT stage"):
                create_fsl_dataset_from_events(mock_dataset, config)

    def test_create_fsl_dataset_with_save_function(self):
        """Test the create_fsl_dataset_from_events_and_save function."""
        # This test verifies the complete pipeline function

        config = EventToFSLConfig(tokenizer_name="test-tokenizer")

        # Test the save function by mocking its dependencies
        with patch("owa.data.processing.event_to_fsl.load_from_disk") as mock_load:
            with patch("owa.data.processing.event_to_fsl.create_fsl_dataset_from_events") as mock_create:
                with patch("owa.data.processing.event_to_fsl.Path") as mock_path:
                    mock_event_dataset = Mock()
                    mock_load.return_value = mock_event_dataset

                    mock_fsl_dataset = Mock()
                    mock_create.return_value = mock_fsl_dataset

                    # Mock Path operations
                    mock_output_path = Mock()
                    mock_path.return_value = mock_output_path

                    # Test the function
                    result = create_fsl_dataset_from_events_and_save("/input/path", "/output/path", config)

                    # Verify the result and calls
                    assert result == mock_fsl_dataset
                    mock_load.assert_called_once_with("/input/path")
                    mock_create.assert_called_once_with(mock_event_dataset, config)
                    mock_output_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
                    mock_fsl_dataset.save_to_disk.assert_called_once()


class TestIntegration:
    """Integration tests for event to FSL processing."""

    def test_functions_exist_and_importable(self):
        """Test that all expected functions can be imported."""
        from owa.data.processing.event_to_fsl import (
            EventToFSLConfig,
            create_fsl_dataset_from_events,
            create_fsl_dataset_from_events_and_save,
        )

        assert callable(create_fsl_dataset_from_events)
        assert callable(create_fsl_dataset_from_events_and_save)
        assert EventToFSLConfig is not None

    def test_config_integration_with_fsl_dataset_config(self):
        """Test that EventToFSLConfig integrates properly with FSLDatasetConfig."""
        fsl_config = FSLDatasetConfig(
            max_sequence_length=4096,
            include_samples_without_images=True,
            action_topics=["keyboard", "mouse"],
        )

        config = EventToFSLConfig(
            tokenizer_name="test-tokenizer",
            fsl_dataset=fsl_config,
        )

        assert config.fsl_dataset.max_sequence_length == 4096
        assert config.fsl_dataset.include_samples_without_images is True
        assert config.fsl_dataset.action_topics == ["keyboard", "mouse"]
