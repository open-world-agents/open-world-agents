"""
Tests for mcap CLI automatic upgrade functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from packaging.version import parse as parse_version

from owa.cli.mcap.info import (
    CURRENT_MCAP_CLI_VERSION,
    MCAP_CLI_DOWNLOAD_URL_TEMPLATES,
    get_latest_mcap_cli_version,
    get_local_mcap_version,
    should_upgrade_mcap,
)


class TestMcapVersionFunctions:
    """Test version checking and upgrade logic."""

    def test_get_latest_version_success(self):
        """Test successful retrieval of latest version from GitHub API."""
        import os

        # Temporarily disable the environment variable set by conftest.py
        original_value = os.environ.get("OWA_DISABLE_VERSION_CHECK")
        if "OWA_DISABLE_VERSION_CHECK" in os.environ:
            del os.environ["OWA_DISABLE_VERSION_CHECK"]

        try:
            mock_response = MagicMock()
            mock_response.json.return_value = [
                {"tag_name": "releases/mcap-cli/v0.0.54"},
                {"tag_name": "releases/rust/v0.19.0"},
                {"tag_name": "releases/mcap-cli/v0.0.53"},
            ]
            mock_response.raise_for_status.return_value = None

            with patch("owa.cli.mcap.info.requests.get", return_value=mock_response):
                version = get_latest_mcap_cli_version()
                assert version == "v0.0.54"
        finally:
            # Restore original value
            if original_value is not None:
                os.environ["OWA_DISABLE_VERSION_CHECK"] = original_value

    def test_get_latest_version_fallback(self):
        """Test fallback to current version when API fails."""
        with patch("owa.cli.mcap.info.requests.get", side_effect=Exception("Network error")):
            version = get_latest_mcap_cli_version()
            assert version == CURRENT_MCAP_CLI_VERSION

    def test_get_latest_version_disabled_by_env_var(self):
        """Test that version check is disabled when OWA_DISABLE_VERSION_CHECK is set."""
        import os

        # Save original value
        original_value = os.environ.get("OWA_DISABLE_VERSION_CHECK")

        try:
            # Set environment variable to disable version check
            os.environ["OWA_DISABLE_VERSION_CHECK"] = "1"

            # Should return current version without making API call
            with patch("owa.cli.mcap.info.requests.get") as mock_get:
                version = get_latest_mcap_cli_version()
                assert version == CURRENT_MCAP_CLI_VERSION
                # Verify no API call was made
                mock_get.assert_not_called()

        finally:
            # Restore original value
            if original_value is None:
                os.environ.pop("OWA_DISABLE_VERSION_CHECK", None)
            else:
                os.environ["OWA_DISABLE_VERSION_CHECK"] = original_value

    def test_get_local_version_success(self):
        """Test successful parsing of local mcap version."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_mcap = Path(temp_dir) / "mcap"
            mock_mcap.touch()

            with patch("owa.cli.mcap.info.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="v0.0.53\n")

                version = get_local_mcap_version(mock_mcap)
                assert version == "v0.0.53"

                # Verify it calls the correct command
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert args[1] == "version"  # Should use "version" subcommand

    def test_get_local_version_failure(self):
        """Test handling of failed version check."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_mcap = Path(temp_dir) / "mcap"
            mock_mcap.touch()

            with patch("owa.cli.mcap.info.subprocess.run", side_effect=FileNotFoundError()):
                version = get_local_mcap_version(mock_mcap)
                assert version == "unknown"

    def test_should_upgrade_nonexistent_file(self):
        """Test upgrade decision for non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_mcap = Path(temp_dir) / "mcap"
            # File doesn't exist
            assert should_upgrade_mcap(mock_mcap) is True

    def test_should_upgrade_force(self):
        """Test force upgrade option."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_mcap = Path(temp_dir) / "mcap"
            mock_mcap.touch()
            assert should_upgrade_mcap(mock_mcap, force=True) is True

    def test_should_upgrade_version_comparison(self):
        """Test version comparison logic."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_mcap = Path(temp_dir) / "mcap"
            mock_mcap.touch()

            # Mock local version as older
            with (
                patch("owa.cli.mcap.info.get_local_mcap_version", return_value="v0.0.52"),
                patch("owa.cli.mcap.info.get_latest_mcap_cli_version", return_value="v0.0.53"),
            ):
                assert should_upgrade_mcap(mock_mcap) is True

            # Mock local version as same
            with (
                patch("owa.cli.mcap.info.get_local_mcap_version", return_value="v0.0.53"),
                patch("owa.cli.mcap.info.get_latest_mcap_cli_version", return_value="v0.0.53"),
            ):
                assert should_upgrade_mcap(mock_mcap) is False

            # Mock local version as newer (shouldn't happen but test anyway)
            with (
                patch("owa.cli.mcap.info.get_local_mcap_version", return_value="v0.0.54"),
                patch("owa.cli.mcap.info.get_latest_mcap_cli_version", return_value="v0.0.53"),
            ):
                assert should_upgrade_mcap(mock_mcap) is False

    def test_should_upgrade_unknown_version(self):
        """Test upgrade decision when local version is unknown."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_mcap = Path(temp_dir) / "mcap"
            mock_mcap.touch()

            with patch("owa.cli.mcap.info.get_local_mcap_version", return_value="unknown"):
                assert should_upgrade_mcap(mock_mcap) is True


class TestVersionParsing:
    """Test version string parsing and comparison."""

    def test_version_parsing(self):
        """Test that version parsing works correctly."""
        # Test that packaging.version can handle our version format
        v1 = parse_version("0.0.53")
        v2 = parse_version("0.0.54")
        v3 = parse_version("0.0.52")

        assert v2 > v1
        assert v1 > v3
        assert v1 == parse_version("0.0.53")

    def test_version_with_v_prefix(self):
        """Test version parsing with 'v' prefix."""
        v1 = parse_version("v0.0.53".lstrip("v"))
        v2 = parse_version("0.0.53")
        assert v1 == v2


class TestConstants:
    """Test that constants are properly defined."""

    def test_current_version_format(self):
        """Test that CURRENT_MCAP_CLI_VERSION has correct format."""
        assert CURRENT_MCAP_CLI_VERSION.startswith("v")
        assert len(CURRENT_MCAP_CLI_VERSION.split(".")) == 3  # v0.0.53 format

    def test_url_templates_format_correctly(self):
        """Test that URL templates format correctly with version."""
        test_version = "v0.0.54"

        for system, template in MCAP_CLI_DOWNLOAD_URL_TEMPLATES.items():
            url = template.format(version=test_version)

            # Should contain the version
            assert test_version in url

            # Should be a valid GitHub releases URL
            assert url.startswith("https://github.com/foxglove/mcap/releases/download/")

            # Should contain the system identifier in the filename
            if system == "windows-amd64":
                assert url.endswith(".exe")
            else:
                assert not url.endswith(".exe")

    def test_all_supported_systems_have_templates(self):
        """Test that all expected systems have URL templates."""
        expected_systems = {"linux-amd64", "linux-arm64", "darwin-amd64", "darwin-arm64", "windows-amd64"}

        assert set(MCAP_CLI_DOWNLOAD_URL_TEMPLATES.keys()) == expected_systems
