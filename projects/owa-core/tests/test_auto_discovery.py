"""
Tests for owa.core.auto_discovery module.

This module tests the automatic plugin discovery functionality including
environment variable controls and error handling.
"""

import os
from unittest.mock import patch

from owa.core.auto_discovery import _should_auto_discover, auto_discover_plugins


class TestShouldAutoDiscover:
    """Test _should_auto_discover function."""

    def test_auto_discover_enabled_by_default(self):
        """Test that auto-discovery is enabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            assert _should_auto_discover() is True

    def test_auto_discover_disabled_by_env_var_1(self):
        """Test that auto-discovery can be disabled with '1'."""
        with patch.dict(os.environ, {"OWA_DISABLE_AUTO_DISCOVERY": "1"}):
            assert _should_auto_discover() is False

    def test_auto_discover_disabled_by_env_var_true(self):
        """Test that auto-discovery can be disabled with 'true'."""
        with patch.dict(os.environ, {"OWA_DISABLE_AUTO_DISCOVERY": "true"}):
            assert _should_auto_discover() is False

    def test_auto_discover_disabled_by_env_var_yes(self):
        """Test that auto-discovery can be disabled with 'yes'."""
        with patch.dict(os.environ, {"OWA_DISABLE_AUTO_DISCOVERY": "yes"}):
            assert _should_auto_discover() is False

    def test_auto_discover_disabled_case_insensitive(self):
        """Test that environment variable check is case insensitive."""
        test_cases = ["TRUE", "True", "YES", "Yes", "1"]

        for value in test_cases:
            with patch.dict(os.environ, {"OWA_DISABLE_AUTO_DISCOVERY": value}):
                assert _should_auto_discover() is False, f"Failed for value: {value}"

    def test_auto_discover_enabled_with_invalid_values(self):
        """Test that auto-discovery remains enabled with invalid disable values."""
        invalid_values = ["0", "false", "no", "invalid", ""]

        for value in invalid_values:
            with patch.dict(os.environ, {"OWA_DISABLE_AUTO_DISCOVERY": value}):
                assert _should_auto_discover() is True, f"Failed for value: {value}"

    def test_auto_discover_enabled_with_empty_env_var(self):
        """Test that empty environment variable enables auto-discovery."""
        with patch.dict(os.environ, {"OWA_DISABLE_AUTO_DISCOVERY": ""}):
            assert _should_auto_discover() is True


class TestAutoDiscoverPlugins:
    """Test auto_discover_plugins function."""

    @patch("owa.core.auto_discovery.discover_and_register_plugins")
    @patch("owa.core.auto_discovery._should_auto_discover")
    def test_auto_discover_when_enabled(self, mock_should_discover, mock_discover):
        """Test that auto_discover_plugins calls discover_and_register_plugins when enabled."""
        mock_should_discover.return_value = True

        auto_discover_plugins()

        mock_should_discover.assert_called_once()
        mock_discover.assert_called_once()

    @patch("owa.core.auto_discovery.discover_and_register_plugins")
    @patch("owa.core.auto_discovery._should_auto_discover")
    def test_auto_discover_when_disabled(self, mock_should_discover, mock_discover):
        """Test that auto_discover_plugins does not call discover_and_register_plugins when disabled."""
        mock_should_discover.return_value = False

        auto_discover_plugins()

        mock_should_discover.assert_called_once()
        mock_discover.assert_not_called()

    @patch("owa.core.auto_discovery.logger")
    @patch("owa.core.auto_discovery.discover_and_register_plugins")
    @patch("owa.core.auto_discovery._should_auto_discover")
    def test_auto_discover_logs_when_disabled(self, mock_should_discover, mock_discover, mock_logger):
        """Test that auto_discover_plugins logs debug message when disabled."""
        mock_should_discover.return_value = False

        auto_discover_plugins()

        mock_logger.debug.assert_called_once_with("Auto-discovery disabled")

    @patch("owa.core.auto_discovery.logger")
    @patch("owa.core.auto_discovery.discover_and_register_plugins")
    @patch("owa.core.auto_discovery._should_auto_discover")
    def test_auto_discover_handles_exceptions(self, mock_should_discover, mock_discover, mock_logger):
        """Test that auto_discover_plugins handles exceptions gracefully."""
        mock_should_discover.return_value = True
        mock_discover.side_effect = RuntimeError("Plugin discovery failed")

        # Should not raise exception
        auto_discover_plugins()

        mock_logger.warning.assert_called_once_with("Auto-discovery failed: Plugin discovery failed")

    @patch("owa.core.auto_discovery.logger")
    @patch("owa.core.auto_discovery.discover_and_register_plugins")
    @patch("owa.core.auto_discovery._should_auto_discover")
    def test_auto_discover_handles_various_exceptions(self, mock_should_discover, mock_discover, mock_logger):
        """Test that auto_discover_plugins handles various exception types."""
        mock_should_discover.return_value = True

        # Test different exception types
        exceptions = [
            RuntimeError("Runtime error"),
            ImportError("Import error"),
            ValueError("Value error"),
            Exception("Generic exception"),
        ]

        for exc in exceptions:
            mock_discover.side_effect = exc
            mock_logger.reset_mock()

            # Should not raise exception
            auto_discover_plugins()

            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Auto-discovery failed:" in warning_call
            assert str(exc) in warning_call

    @patch("owa.core.auto_discovery.discover_and_register_plugins")
    def test_auto_discover_integration_with_env_var(self, mock_discover):
        """Test integration of auto_discover_plugins with environment variable."""
        # Test with auto-discovery enabled
        with patch.dict(os.environ, {}, clear=True):
            auto_discover_plugins()
            mock_discover.assert_called_once()

        mock_discover.reset_mock()

        # Test with auto-discovery disabled
        with patch.dict(os.environ, {"OWA_DISABLE_AUTO_DISCOVERY": "true"}):
            auto_discover_plugins()
            mock_discover.assert_not_called()
