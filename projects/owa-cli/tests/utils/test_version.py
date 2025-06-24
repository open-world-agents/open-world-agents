"""
Tests for OWA CLI version checking utilities.
"""

import os
from unittest.mock import MagicMock, patch

import requests

from owa.cli.utils import check_for_update, get_latest_release, get_local_version


class TestVersionUtilityFunctions:
    """Test version checking and update utility functions."""

    def test_get_local_version_success(self):
        """Test successful retrieval of local package version."""
        with patch("importlib.metadata.version") as mock_version:
            mock_version.return_value = "0.4.1"
            version = get_local_version("owa.cli")
            assert version == "0.4.1"

    def test_get_local_version_failure(self):
        """Test handling of failed local version retrieval."""
        with patch("importlib.metadata.version", side_effect=Exception("Package not found")):
            version = get_local_version("nonexistent.package")
            assert version == "unknown"

    def test_get_latest_release_success(self):
        """Test successful retrieval of latest release from GitHub API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"tag_name": "v0.4.2"}
        mock_response.raise_for_status.return_value = None

        # Temporarily unset OWA_DISABLE_VERSION_CHECK
        original_value = os.environ.get("OWA_DISABLE_VERSION_CHECK")
        if "OWA_DISABLE_VERSION_CHECK" in os.environ:
            del os.environ["OWA_DISABLE_VERSION_CHECK"]
        try:
            with patch("owa.cli.utils.requests.get", return_value=mock_response):
                version = get_latest_release()
                assert version == "0.4.2"  # Should strip the 'v' prefix
        finally:
            if original_value is not None:
                os.environ["OWA_DISABLE_VERSION_CHECK"] = original_value

    def test_get_latest_release_with_no_v_prefix(self):
        """Test handling of release tags without 'v' prefix."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"tag_name": "0.4.2"}
        mock_response.raise_for_status.return_value = None

        # Temporarily unset OWA_DISABLE_VERSION_CHECK
        original_value = os.environ.get("OWA_DISABLE_VERSION_CHECK")
        if "OWA_DISABLE_VERSION_CHECK" in os.environ:
            del os.environ["OWA_DISABLE_VERSION_CHECK"]
        try:
            with patch("owa.cli.utils.requests.get", return_value=mock_response):
                version = get_latest_release()
                assert version == "0.4.2"
        finally:
            if original_value is not None:
                os.environ["OWA_DISABLE_VERSION_CHECK"] = original_value

    def test_check_for_update_disabled_by_env_var(self):
        """Test that update check is disabled when OWA_DISABLE_VERSION_CHECK is set."""
        # Save original value
        original_value = os.environ.get("OWA_DISABLE_VERSION_CHECK")

        try:
            # Set environment variable to disable version check
            os.environ["OWA_DISABLE_VERSION_CHECK"] = "1"

            # Should return True without making any API calls
            with patch("owa.cli.utils.requests.get") as mock_get:
                result = check_for_update()
                assert result is True
                # Verify no API call was made
                mock_get.assert_not_called()

        finally:
            # Restore original value
            if original_value is None:
                os.environ.pop("OWA_DISABLE_VERSION_CHECK", None)
            else:
                os.environ["OWA_DISABLE_VERSION_CHECK"] = original_value

    def test_check_for_update_up_to_date(self):
        """Test update check when local version is up to date."""
        with (
            patch("owa.cli.utils.get_local_version", return_value="0.4.2"),
            patch("owa.cli.utils.get_latest_release", return_value="0.4.2"),
        ):
            result = check_for_update()
            assert result is True

    def test_check_for_update_newer_available(self, capsys):
        """Test update check when newer version is available."""
        # Temporarily disable the environment variable set by conftest.py
        original_value = os.environ.get("OWA_DISABLE_VERSION_CHECK")
        if "OWA_DISABLE_VERSION_CHECK" in os.environ:
            del os.environ["OWA_DISABLE_VERSION_CHECK"]

        try:
            with (
                patch("owa.cli.utils.get_local_version", return_value="0.4.1"),
                patch("owa.cli.utils.get_latest_release", return_value="0.4.2"),
            ):
                result = check_for_update()
                assert result is False

                # Check that update message was printed
                captured = capsys.readouterr()
                assert "An update is available" in captured.out
                assert "0.4.1" in captured.out  # Current version
                assert "0.4.2" in captured.out  # Latest version
        finally:
            # Restore original value
            if original_value is not None:
                os.environ["OWA_DISABLE_VERSION_CHECK"] = original_value

    def test_check_for_update_timeout_error(self, capsys):
        """Test handling of timeout errors during update check."""
        # Temporarily disable the environment variable set by conftest.py
        original_value = os.environ.get("OWA_DISABLE_VERSION_CHECK")
        if "OWA_DISABLE_VERSION_CHECK" in os.environ:
            del os.environ["OWA_DISABLE_VERSION_CHECK"]

        try:
            with (
                patch("owa.cli.utils.get_local_version", return_value="0.4.1"),
                patch("owa.cli.utils.get_latest_release", side_effect=requests.Timeout("Timeout")),
            ):
                result = check_for_update()
                assert result is False

                # Check that timeout error message was printed
                captured = capsys.readouterr()
                assert "Timeout occurred" in captured.out
        finally:
            # Restore original value
            if original_value is not None:
                os.environ["OWA_DISABLE_VERSION_CHECK"] = original_value

    def test_check_for_update_request_error(self, capsys):
        """Test handling of request errors during update check."""
        # Temporarily disable the environment variable set by conftest.py
        original_value = os.environ.get("OWA_DISABLE_VERSION_CHECK")
        if "OWA_DISABLE_VERSION_CHECK" in os.environ:
            del os.environ["OWA_DISABLE_VERSION_CHECK"]

        try:
            with (
                patch("owa.cli.utils.get_local_version", return_value="0.4.1"),
                patch("owa.cli.utils.get_latest_release", side_effect=requests.RequestException("Network error")),
            ):
                result = check_for_update()
                assert result is False

                # Check that request error message was printed
                captured = capsys.readouterr()
                assert "Request failed" in captured.out
        finally:
            # Restore original value
            if original_value is not None:
                os.environ["OWA_DISABLE_VERSION_CHECK"] = original_value

    def test_check_for_update_general_error(self, capsys):
        """Test handling of general errors during update check."""
        # Temporarily disable the environment variable set by conftest.py
        original_value = os.environ.get("OWA_DISABLE_VERSION_CHECK")
        if "OWA_DISABLE_VERSION_CHECK" in os.environ:
            del os.environ["OWA_DISABLE_VERSION_CHECK"]

        try:
            with (
                patch("owa.cli.utils.get_local_version", return_value="0.4.1"),
                patch("owa.cli.utils.get_latest_release", side_effect=Exception("Unexpected error")),
            ):
                result = check_for_update()
                assert result is False

                # Check that general error message was printed
                captured = capsys.readouterr()
                assert "An unexpected error occurred" in captured.out
        finally:
            # Restore original value
            if original_value is not None:
                os.environ["OWA_DISABLE_VERSION_CHECK"] = original_value


class TestVersionCheckingIntegration:
    """Test integration of version checking with CLI entry point."""

    def test_main_cli_respects_env_var(self):
        """Test that main CLI entry point respects the environment variable."""
        # Save original value
        original_value = os.environ.get("OWA_DISABLE_VERSION_CHECK")

        try:
            # Set environment variable to disable version check
            os.environ["OWA_DISABLE_VERSION_CHECK"] = "1"

            # Import and test the main function
            from owa.cli import main

            with patch("owa.cli.utils.requests.get") as mock_get:
                # Should not raise any exceptions and not make API calls
                main()
                mock_get.assert_not_called()

        finally:
            # Restore original value
            if original_value is None:
                os.environ.pop("OWA_DISABLE_VERSION_CHECK", None)
            else:
                os.environ["OWA_DISABLE_VERSION_CHECK"] = original_value
