"""
Tests for ocap recording functionality.

This module tests the core ocap functionality including:
- Event queue management
- Callback functions
- Plugin checking
- Resource setup
- File management utilities
- CLI argument parsing
"""

import tempfile
import time
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, Mock, patch

import pytest
import typer

from owa.ocap.record import (
    check_plugin,
    enqueue_event,
    ensure_output_files_ready,
    event_queue,
    keyboard_monitor_callback,
    parse_additional_properties,
    screen_capture_callback,
    setup_resources,
)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_event():
    """Create a mock event object for testing."""
    event = Mock()
    event.vk = 65  # 'A' key
    event.event_type = "press"
    event.path = "/some/path/image.png"
    return event


@pytest.fixture
def clear_event_queue():
    """Clear the global event queue before each test."""
    # Clear the queue
    while not event_queue.empty():
        event_queue.get()
    yield
    # Clear again after test
    while not event_queue.empty():
        event_queue.get()


class TestEventQueue:
    """Test event queue functionality."""

    def test_enqueue_event(self, clear_event_queue, mock_event):
        """Test that events are properly enqueued with timestamp."""
        topic = "test_topic"

        # Enqueue an event
        enqueue_event(mock_event, topic=topic)

        # Check that event was added to queue
        assert not event_queue.empty()

        # Get the event and verify contents
        queued_topic, queued_event, timestamp = event_queue.get()
        assert queued_topic == topic
        assert queued_event == mock_event
        assert isinstance(timestamp, int)
        assert timestamp > 0

    def test_enqueue_multiple_events(self, clear_event_queue, mock_event):
        """Test enqueueing multiple events."""
        topics = ["keyboard", "mouse", "screen"]

        # Enqueue multiple events
        for topic in topics:
            enqueue_event(mock_event, topic=topic)

        # Verify all events are in queue
        assert event_queue.qsize() == 3

        # Verify events come out in FIFO order
        for expected_topic in topics:
            topic, event, timestamp = event_queue.get()
            assert topic == expected_topic
            assert event == mock_event


class TestCallbacks:
    """Test callback functions."""

    def test_keyboard_monitor_callback_regular_key(self, clear_event_queue, mock_event):
        """Test keyboard callback with regular key."""
        mock_event.vk = 65  # 'A' key
        mock_event.event_type = "press"

        keyboard_monitor_callback(mock_event)

        # Verify event was enqueued
        assert not event_queue.empty()
        topic, event, timestamp = event_queue.get()
        assert topic == "keyboard"
        assert event == mock_event

    def test_keyboard_monitor_callback_f_key(self, clear_event_queue, mock_event):
        """Test keyboard callback with F1-F12 key (should log info)."""
        mock_event.vk = 0x70  # F1 key
        mock_event.event_type = "press"

        with patch('owa.ocap.record.logger') as mock_logger:
            keyboard_monitor_callback(mock_event)

            # Verify info was logged for F key
            mock_logger.info.assert_called_once_with("F1-F12 key pressed: F1")

        # Verify event was still enqueued
        assert not event_queue.empty()
        topic, event, timestamp = event_queue.get()
        assert topic == "keyboard"

    def test_keyboard_monitor_callback_f12_key(self, clear_event_queue, mock_event):
        """Test keyboard callback with F12 key."""
        mock_event.vk = 0x7B  # F12 key
        mock_event.event_type = "press"

        with patch('owa.ocap.record.logger') as mock_logger:
            keyboard_monitor_callback(mock_event)

            # Verify correct F key number is logged
            mock_logger.info.assert_called_once_with("F1-F12 key pressed: F12")

    def test_screen_capture_callback(self, clear_event_queue, mock_event, temp_output_dir):
        """Test screen capture callback."""
        # Set up global MCAP_LOCATION
        with patch('owa.ocap.record.MCAP_LOCATION', temp_output_dir / "output.mcap"):
            # Create a mock event with a path
            mock_event.path = temp_output_dir / "subdir" / "image.png"

            screen_capture_callback(mock_event)

            # Verify path was converted to relative
            assert mock_event.path == "subdir/image.png"

        # Verify event was enqueued
        assert not event_queue.empty()
        topic, event, timestamp = event_queue.get()
        assert topic == "screen"
        assert event == mock_event


class TestPluginChecking:
    """Test plugin checking functionality."""

    @patch('owa.ocap.record.get_plugin_discovery')
    def test_check_plugin_success(self, mock_get_discovery):
        """Test successful plugin checking."""
        # Mock successful plugin discovery
        mock_discovery = Mock()
        mock_discovery.get_plugin_info.return_value = (["desktop", "gst"], [])
        mock_get_discovery.return_value = mock_discovery

        # Should not raise an exception
        check_plugin()

        # Verify correct plugins were checked
        mock_discovery.get_plugin_info.assert_called_once_with(["desktop", "gst"])

    @patch('owa.ocap.record.get_plugin_discovery')
    def test_check_plugin_failure(self, mock_get_discovery):
        """Test plugin checking with missing plugins."""
        # Mock failed plugin discovery
        mock_discovery = Mock()
        mock_discovery.get_plugin_info.return_value = (["desktop"], ["gst"])
        mock_get_discovery.return_value = mock_discovery

        # Should raise AssertionError
        with pytest.raises(AssertionError, match="Failed to load plugins"):
            check_plugin()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_parse_additional_properties_none(self):
        """Test parsing additional properties with None input."""
        result = parse_additional_properties(None)
        assert result == {}

    def test_parse_additional_properties_single(self):
        """Test parsing single additional property."""
        result = parse_additional_properties("key1=value1")
        assert result == {"key1": "value1"}

    def test_parse_additional_properties_multiple(self):
        """Test parsing multiple additional properties."""
        result = parse_additional_properties("key1=value1,key2=value2,key3=value3")
        assert result == {"key1": "value1", "key2": "value2", "key3": "value3"}

    def test_ensure_output_files_ready_new_file(self, temp_output_dir):
        """Test ensuring output files are ready with new file."""
        file_location = temp_output_dir / "new_recording"

        result = ensure_output_files_ready(file_location)

        assert result == file_location.with_suffix(".mcap")
        assert result.parent.exists()

    def test_ensure_output_files_ready_existing_directory_creation(self, temp_output_dir):
        """Test directory creation when parent doesn't exist."""
        file_location = temp_output_dir / "subdir" / "recording"

        result = ensure_output_files_ready(file_location)

        assert result == file_location.with_suffix(".mcap")
        assert result.parent.exists()

    def test_ensure_output_files_ready_existing_file_abort(self, temp_output_dir):
        """Test aborting when file exists and user chooses not to delete."""
        file_location = temp_output_dir / "existing_recording"
        existing_file = file_location.with_suffix(".mcap")
        existing_file.touch()  # Create the file

        with patch('typer.confirm', return_value=False):
            with pytest.raises(typer.Abort):
                ensure_output_files_ready(file_location)

    def test_ensure_output_files_ready_existing_file_delete(self, temp_output_dir):
        """Test deleting existing files when user confirms."""
        file_location = temp_output_dir / "existing_recording"
        existing_mcap = file_location.with_suffix(".mcap")
        existing_mkv = file_location.with_suffix(".mkv")
        existing_mcap.touch()
        existing_mkv.touch()

        with patch('typer.confirm', return_value=True):
            result = ensure_output_files_ready(file_location)

        assert result == existing_mcap
        assert not existing_mcap.exists()
        assert not existing_mkv.exists()


class TestResourceSetup:
    """Test resource setup context manager."""

    @patch('owa.ocap.record.check_plugin')
    @patch('owa.ocap.record.LISTENERS')
    def test_setup_resources_basic(self, mock_listeners, mock_check_plugin, temp_output_dir):
        """Test basic resource setup and teardown."""
        # Mock all the listeners and recorder
        mock_recorder = Mock()
        mock_keyboard_listener = Mock()
        mock_mouse_listener = Mock()
        mock_window_listener = Mock()
        mock_keyboard_state_listener = Mock()
        mock_mouse_state_listener = Mock()

        # Configure the mock listeners to return configured instances
        mock_listeners.__getitem__.side_effect = lambda key: {
            "gst/omnimodal.appsink_recorder": lambda: mock_recorder,
            "desktop/keyboard": lambda: Mock(configure=lambda **kwargs: mock_keyboard_listener),
            "desktop/mouse": lambda: Mock(configure=lambda **kwargs: mock_mouse_listener),
            "desktop/window": lambda: Mock(configure=lambda **kwargs: mock_window_listener),
            "desktop/keyboard_state": lambda: Mock(configure=lambda **kwargs: mock_keyboard_state_listener),
            "desktop/mouse_state": lambda: Mock(configure=lambda **kwargs: mock_mouse_state_listener),
        }[key]

        # Mock the is_alive method to return False (stopped)
        for listener in [mock_recorder, mock_keyboard_listener, mock_mouse_listener,
                        mock_window_listener, mock_keyboard_state_listener, mock_mouse_state_listener]:
            listener.is_alive.return_value = False

        file_location = temp_output_dir / "test_recording.mcap"

        # Test the context manager
        with setup_resources(
            file_location=file_location,
            record_audio=True,
            record_video=True,
            record_timestamp=True,
            show_cursor=True,
            fps=60.0,
            window_name=None,
            monitor_idx=None,
            width=None,
            height=None,
            additional_properties={}
        ):
            # Verify all resources were started
            mock_recorder.start.assert_called_once()
            mock_keyboard_listener.start.assert_called_once()
            mock_mouse_listener.start.assert_called_once()
            mock_window_listener.start.assert_called_once()
            mock_keyboard_state_listener.start.assert_called_once()
            mock_mouse_state_listener.start.assert_called_once()

        # Verify all resources were stopped
        mock_recorder.stop.assert_called_once()
        mock_keyboard_listener.stop.assert_called_once()
        mock_mouse_listener.stop.assert_called_once()
        mock_window_listener.stop.assert_called_once()
        mock_keyboard_state_listener.stop.assert_called_once()
        mock_mouse_state_listener.stop.assert_called_once()

        # Verify join was called with timeout
        mock_recorder.join.assert_called_once_with(timeout=5)
        mock_keyboard_listener.join.assert_called_once_with(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__])
