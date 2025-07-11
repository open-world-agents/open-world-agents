"""
Tests for owa.core.runner.subprocess_runner module.

This module tests the subprocess management functionality including
SubprocessRunner class and helper functions.
"""

import signal
import subprocess
import threading
from unittest.mock import Mock, patch

import pytest

from owa.core.runner.subprocess_runner import SubprocessRunner


class TestSubprocessRunner:
    """Test SubprocessRunner class."""

    def test_configure(self):
        """Test configure method with default and custom signals."""
        runner = SubprocessRunner()
        subprocess_args = ["echo", "hello"]

        # Test default signal
        configured_runner = runner.configure(subprocess_args)
        assert configured_runner is runner  # Should return self
        assert runner.subprocess_args == subprocess_args
        assert runner._stop_signal == signal.CTRL_BREAK_EVENT
        assert runner._process is None
        assert runner._configured is True  # Should be marked as configured

        # Test custom signal
        runner2 = SubprocessRunner()
        runner2.configure(subprocess_args, stop_signal=signal.SIGTERM)
        assert runner2._stop_signal == signal.SIGTERM
        assert runner2._configured is True

    def test_loop_cleanup_on_exception(self):
        """Test that loop() calls cleanup() even if exception occurs."""
        runner = SubprocessRunner()
        runner.configure(["echo", "hello"])

        with (
            patch.object(runner, "_loop", side_effect=Exception("Test error")),
            patch.object(runner, "cleanup") as mock_cleanup,
        ):
            with pytest.raises(Exception, match="Test error"):
                runner.loop()
            mock_cleanup.assert_called_once()

    @patch("owa.core.runner.subprocess_runner.subprocess.Popen")
    def test_process_lifecycle(self, mock_popen):
        """Test complete process lifecycle including stop event handling."""
        runner = SubprocessRunner()
        runner.configure(["echo", "hello"])
        runner._stop_event = threading.Event()

        # Mock process that responds to stop event
        mock_process = Mock()
        mock_process.poll.side_effect = [None, 0]  # Running, then stopped
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        # Set stop event immediately to trigger signal sending
        runner._stop_event.set()
        runner._loop()

        # Verify process creation and signal handling
        mock_popen.assert_called_once_with(["echo", "hello"], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        mock_process.send_signal.assert_called_once_with(signal.CTRL_BREAK_EVENT)
        mock_process.wait.assert_called_once_with(timeout=5)
        assert runner._process == mock_process

    @patch("owa.core.runner.subprocess_runner.subprocess.Popen")
    def test_timeout_handling(self, mock_popen):
        """Test handling of process wait timeout."""
        runner = SubprocessRunner()
        runner.configure(["echo", "hello"])
        runner._stop_event = threading.Event()

        mock_process = Mock()
        mock_process.poll.side_effect = [None, 0]
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
        mock_popen.return_value = mock_process

        runner._stop_event.set()
        runner._loop()  # Should not raise exception

        mock_process.send_signal.assert_called_once_with(signal.CTRL_BREAK_EVENT)
        mock_process.wait.assert_called_once_with(timeout=5)

    def test_cleanup_scenarios(self):
        """Test cleanup() in various scenarios."""
        runner = SubprocessRunner()

        # Test with no process
        runner._process = None
        runner.cleanup()  # Should not raise exception

        # Test with already terminated process
        mock_process1 = Mock()
        mock_process1.poll.return_value = 0  # Already terminated
        runner._process = mock_process1
        runner.cleanup()
        mock_process1.terminate.assert_not_called()
        mock_process1.kill.assert_not_called()

        # Test graceful termination
        mock_process2 = Mock()
        mock_process2.poll.return_value = None  # Still running
        mock_process2.wait.return_value = 0  # Terminates gracefully
        runner._process = mock_process2
        runner.cleanup()
        mock_process2.terminate.assert_called_once()
        mock_process2.wait.assert_called_once_with(timeout=5)
        mock_process2.kill.assert_not_called()

    def test_cleanup_forceful_termination(self):
        """Test cleanup() with forceful termination after timeout."""
        runner = SubprocessRunner()
        mock_process = Mock()
        mock_process.poll.return_value = None  # Still running
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
        runner._process = mock_process

        runner.cleanup()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)
        mock_process.kill.assert_called_once()

    def test_cleanup_exception_handling(self):
        """Test cleanup() handles exceptions gracefully."""
        runner = SubprocessRunner()
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.terminate.side_effect = Exception("Termination failed")
        runner._process = mock_process

        # Should not raise exception
        runner.cleanup()
