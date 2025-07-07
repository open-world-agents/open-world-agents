"""
Tests for the simplified recording features: health monitoring, delayed start, and auto-stop.
"""

import time
from unittest.mock import MagicMock, patch

from owa.ocap.recorder import check_resources_health, countdown_delay


class TestHealthCheck:
    """Test the health checking functionality."""

    def test_check_resources_health_all_healthy(self):
        """Test health check when all resources are healthy."""
        mock_resource1 = MagicMock()
        mock_resource1.is_alive.return_value = True
        mock_resource2 = MagicMock()
        mock_resource2.is_alive.return_value = True

        resources = [(mock_resource1, "resource1"), (mock_resource2, "resource2")]
        unhealthy = check_resources_health(resources)

        assert unhealthy == []
        mock_resource1.is_alive.assert_called_once()
        mock_resource2.is_alive.assert_called_once()

    def test_check_resources_health_some_unhealthy(self):
        """Test health check when some resources are unhealthy."""
        mock_healthy = MagicMock()
        mock_healthy.is_alive.return_value = True
        mock_unhealthy = MagicMock()
        mock_unhealthy.is_alive.return_value = False

        resources = [(mock_healthy, "healthy_resource"), (mock_unhealthy, "failing_resource")]
        unhealthy = check_resources_health(resources)

        assert unhealthy == ["failing_resource"]

    def test_check_resources_health_all_unhealthy(self):
        """Test health check when all resources are unhealthy."""
        mock_resource1 = MagicMock()
        mock_resource1.is_alive.return_value = False
        mock_resource2 = MagicMock()
        mock_resource2.is_alive.return_value = False

        resources = [(mock_resource1, "resource1"), (mock_resource2, "resource2")]
        unhealthy = check_resources_health(resources)

        assert set(unhealthy) == {"resource1", "resource2"}


class TestCountdownDelay:
    """Test the countdown_delay function."""

    def test_countdown_delay_zero_seconds(self):
        """Test that no delay occurs for zero seconds."""
        start_time = time.time()
        countdown_delay(0)
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should be nearly instant

    def test_countdown_delay_short_duration(self):
        """Test countdown for short duration (< 3 seconds)."""
        with patch("owa.ocap.recorder.logger") as mock_logger, patch("time.sleep") as mock_sleep:
            countdown_delay(1.5)

            # Should log start message and call sleep once
            mock_logger.info.assert_any_call("⏱️ Recording will start in 1.5 seconds...")
            mock_sleep.assert_called_once_with(1.5)

    def test_countdown_delay_long_duration(self):
        """Test countdown for long duration (>= 3 seconds)."""
        with patch("owa.ocap.recorder.logger") as mock_logger, patch("time.sleep") as mock_sleep:  # noqa: F841
            countdown_delay(3.5)

            # Should log start message and countdown messages
            mock_logger.info.assert_any_call("⏱️ Recording will start in 3.5 seconds...")
            mock_logger.info.assert_any_call("Starting in 3...")
            mock_logger.info.assert_any_call("Starting in 2...")
            mock_logger.info.assert_any_call("Starting in 1...")
            mock_logger.info.assert_any_call("🎬 Recording started!")


class TestIntegration:
    """Integration tests for the new features."""

    def test_health_check_integration(self):
        """Test health check integration."""
        mock_recorder = MagicMock()
        mock_recorder.is_alive.return_value = True

        mock_listener = MagicMock()
        mock_listener.is_alive.return_value = True

        resources = [(mock_recorder, "recorder"), (mock_listener, "listener")]

        # All resources healthy
        unhealthy = check_resources_health(resources)
        assert unhealthy == []

        # Simulate one resource failing
        mock_listener.is_alive.return_value = False
        unhealthy = check_resources_health(resources)
        assert unhealthy == ["listener"]
