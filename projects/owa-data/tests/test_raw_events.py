"""Tests for raw event processing functionality."""

from unittest.mock import Mock, patch


from owa.data.processing.raw_events import process_raw_events_file, generate_event_examples


class TestProcessRawEventsFile:
    """Test the process_raw_events_file function."""

    def test_import_availability(self):
        """Test that the function can be imported from the raw_events module."""
        # Test import from raw_events module
        from owa.data.processing.raw_events import process_raw_events_file

        assert callable(process_raw_events_file)

    @patch("owa.data.processing.raw_events.OWAMcapReader")
    @patch("owa.data.processing.raw_events.InactivityFilter")
    def test_process_raw_events_file_basic(self, mock_filter, mock_reader):
        """Test basic functionality of process_raw_events_file."""
        # Mock the interval extractor
        mock_intervals = Mock()
        mock_intervals.__iter__ = Mock(return_value=iter([Mock(start=1000, end=2000)]))
        mock_filter.return_value.extract_intervals.return_value = mock_intervals

        # Mock the MCAP reader
        mock_message = Mock()
        mock_message.topic = "screen"
        mock_message.timestamp = 1500
        mock_message.message_type = "desktop/ScreenCaptured"
        mock_message.model_dump_json.return_value = '{"test": "data"}'

        mock_reader_instance = Mock()
        mock_reader_instance.iter_messages.return_value = [mock_message]
        mock_reader.return_value.__enter__.return_value = mock_reader_instance

        # Mock the resampler
        with patch("owa.data.processing.raw_events.create_resampler") as mock_create_resampler:
            mock_resampler = Mock()
            mock_resampler.pop_event.return_value = [mock_message]
            mock_create_resampler.return_value = mock_resampler

            # Test the function
            result = process_raw_events_file(
                episode_path="/fake/path.mcap", rate_settings={"screen": 20.0}, keep_topics=["screen"]
            )

            # Verify results
            assert len(result) == 1
            event = result[0]
            assert event["episode_path"] == "/fake/path.mcap"
            assert event["topic"] == "screen"
            assert event["timestamp_ns"] == 1500
            assert event["message_type"] == "desktop/ScreenCaptured"
            assert event["mcap_message"] == b'{"test": "data"}'

    @patch("owa.data.processing.raw_events.OWAMcapReader")
    @patch("owa.data.processing.raw_events.InactivityFilter")
    def test_process_raw_events_file_with_root_directory(self, mock_filter, mock_reader):
        """Test process_raw_events_file with mcap_root_directory parameter."""
        # Mock the interval extractor
        mock_intervals = Mock()
        mock_intervals.__iter__ = Mock(return_value=iter([Mock(start=1000, end=2000)]))
        mock_filter.return_value.extract_intervals.return_value = mock_intervals

        # Mock the MCAP reader
        mock_message = Mock()
        mock_message.topic = "screen"
        mock_message.timestamp = 1500
        mock_message.message_type = "desktop/ScreenCaptured"
        mock_message.model_dump_json.return_value = '{"test": "data"}'

        mock_reader_instance = Mock()
        mock_reader_instance.iter_messages.return_value = [mock_message]
        mock_reader.return_value.__enter__.return_value = mock_reader_instance

        # Mock the resampler
        with patch("owa.data.processing.raw_events.create_resampler") as mock_create_resampler:
            mock_resampler = Mock()
            mock_resampler.pop_event.return_value = [mock_message]
            mock_create_resampler.return_value = mock_resampler

            # Test with root directory
            result = process_raw_events_file(
                episode_path="/root/data/episode.mcap",
                rate_settings={"screen": 20.0},
                keep_topics=["screen"],
                mcap_root_directory="/root",
            )

            # Verify relative path is stored
            assert len(result) == 1
            event = result[0]
            assert event["episode_path"] == "data/episode.mcap"

    @patch("owa.data.processing.raw_events.OWAMcapReader")
    @patch("owa.data.processing.raw_events.InactivityFilter")
    def test_process_raw_events_file_empty_topics(self, mock_filter, mock_reader):
        """Test process_raw_events_file with empty keep_topics."""
        mock_intervals = Mock()
        mock_intervals.__iter__ = Mock(return_value=iter([]))
        mock_filter.return_value.extract_intervals.return_value = mock_intervals

        # Mock the MCAP reader
        mock_reader_instance = Mock()
        mock_reader_instance.iter_messages.return_value = []
        mock_reader.return_value.__enter__.return_value = mock_reader_instance

        result = process_raw_events_file(episode_path="/fake/path.mcap", rate_settings={}, keep_topics=[])

        assert result == []


class TestGenerateEventExamples:
    """Test the generate_event_examples function."""

    def test_import_availability(self):
        """Test that the function can be imported from the raw_events module."""
        # Test import from raw_events module
        from owa.data.processing.raw_events import generate_event_examples

        assert callable(generate_event_examples)

    @patch("owa.data.processing.raw_events.ProcessPoolExecutor")
    @patch("owa.data.processing.raw_events.process_raw_events_file")
    def test_generate_event_examples_basic(self, mock_process_file, mock_executor):
        """Test basic functionality of generate_event_examples."""
        # Mock the process_raw_events_file function
        mock_process_file.return_value = [
            {
                "episode_path": "test.mcap",
                "topic": "screen",
                "timestamp_ns": 1000,
                "message_type": "desktop/ScreenCaptured",
                "mcap_message": b'{"test": "data"}',
            }
        ]

        # Mock the executor
        mock_future = Mock()
        mock_future.result.return_value = mock_process_file.return_value

        mock_executor_instance = Mock()
        mock_executor_instance.submit.return_value = mock_future
        mock_executor_instance.__enter__ = Mock(return_value=mock_executor_instance)
        mock_executor_instance.__exit__ = Mock(return_value=None)
        mock_executor.return_value = mock_executor_instance

        # Mock as_completed to return the future immediately
        with patch("owa.data.processing.raw_events.as_completed") as mock_as_completed:
            mock_as_completed.return_value = [mock_future]

            # Test the generator
            events = list(
                generate_event_examples(
                    episode_paths=["test.mcap"], rate_settings={"screen": 20.0}, keep_topics=["screen"], num_workers=1
                )
            )

            # Verify results
            assert len(events) == 1
            event = events[0]
            assert event["episode_path"] == "test.mcap"
            assert event["topic"] == "screen"
            assert event["timestamp_ns"] == 1000

    @patch("owa.data.processing.raw_events.ProcessPoolExecutor")
    def test_generate_event_examples_error_handling(self, mock_executor):
        """Test error handling in generate_event_examples."""
        # Mock the executor to raise an exception
        mock_future = Mock()
        mock_future.result.side_effect = Exception("Test error")

        mock_executor_instance = Mock()
        mock_executor_instance.submit.return_value = mock_future
        mock_executor_instance.__enter__ = Mock(return_value=mock_executor_instance)
        mock_executor_instance.__exit__ = Mock(return_value=None)
        mock_executor.return_value = mock_executor_instance

        # Mock as_completed to return the future immediately
        with patch("owa.data.processing.raw_events.as_completed") as mock_as_completed:
            mock_as_completed.return_value = [mock_future]

            # Test that the generator handles errors gracefully
            events = list(
                generate_event_examples(
                    episode_paths=["test.mcap"], rate_settings={"screen": 20.0}, keep_topics=["screen"], num_workers=1
                )
            )

            # Should return empty list when all files fail
            assert events == []


class TestIntegration:
    """Integration tests for raw event processing."""

    @patch("owa.data.processing.raw_events.OWAMcapReader")
    @patch("owa.data.processing.raw_events.InactivityFilter")
    def test_functions_work_together(self, mock_filter, mock_reader):
        """Test that the functions can work together in a realistic scenario."""
        # This is a basic integration test that verifies the functions exist
        # and can be called without errors (with mocked dependencies)

        mock_intervals = Mock()
        mock_intervals.__iter__ = Mock(return_value=iter([]))
        mock_filter.return_value.extract_intervals.return_value = mock_intervals

        # Mock the MCAP reader
        mock_reader_instance = Mock()
        mock_reader_instance.iter_messages.return_value = []
        mock_reader.return_value.__enter__.return_value = mock_reader_instance

        # Test that process_raw_events_file can be called
        result = process_raw_events_file(episode_path="/fake/path.mcap", rate_settings={}, keep_topics=[])
        assert isinstance(result, list)

        # Test that generate_event_examples can be called
        with patch("owa.data.processing.raw_events.ProcessPoolExecutor") as mock_executor:
            mock_executor_instance = Mock()
            mock_executor_instance.__enter__ = Mock(return_value=mock_executor_instance)
            mock_executor_instance.__exit__ = Mock(return_value=None)
            mock_executor.return_value = mock_executor_instance

            generator = generate_event_examples(episode_paths=[], rate_settings={}, keep_topics=[], num_workers=1)
            # Just verify it's a generator
            assert hasattr(generator, "__iter__")
            assert hasattr(generator, "__next__")
