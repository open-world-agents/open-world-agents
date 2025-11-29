"""Tests for mcap CLI automatic upgrade functionality."""

from unittest.mock import MagicMock, patch

from packaging.version import parse as parse_version

from owa.cli.mcap.info import (
    CURRENT_MCAP_CLI_VERSION,
    MCAP_CLI_DOWNLOAD_URL_TEMPLATES,
    get_latest_mcap_cli_version,
    get_local_mcap_version,
    should_upgrade_mcap,
)


def test_get_latest_version_success(monkeypatch):
    """Test successful retrieval of latest version from GitHub API."""
    monkeypatch.delenv("OWA_DISABLE_VERSION_CHECK", raising=False)

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


def test_get_latest_version_fallback():
    """Test fallback to current version when API fails."""
    with patch("owa.cli.mcap.info.requests.get", side_effect=Exception("Network error")):
        assert get_latest_mcap_cli_version() == CURRENT_MCAP_CLI_VERSION


def test_get_latest_version_disabled_by_env_var(monkeypatch):
    """Test that version check is disabled when OWA_DISABLE_VERSION_CHECK is set."""
    monkeypatch.setenv("OWA_DISABLE_VERSION_CHECK", "1")

    with patch("owa.cli.mcap.info.requests.get") as mock_get:
        version = get_latest_mcap_cli_version()
        assert version == CURRENT_MCAP_CLI_VERSION
        mock_get.assert_not_called()


def test_get_local_version_success(tmp_path):
    """Test successful parsing of local mcap version."""
    mock_mcap = tmp_path / "mcap"
    mock_mcap.touch()

    with patch("owa.cli.mcap.info.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="v0.0.53\n")
        version = get_local_mcap_version(mock_mcap)
        assert version == "v0.0.53"

        # Verify it calls the correct command
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[1] == "version"


def test_get_local_version_failure(tmp_path):
    """Test handling of failed version check."""
    mock_mcap = tmp_path / "mcap"
    mock_mcap.touch()

    with patch("owa.cli.mcap.info.subprocess.run", side_effect=FileNotFoundError()):
        assert get_local_mcap_version(mock_mcap) == "unknown"


def test_should_upgrade_nonexistent(tmp_path):
    """Test upgrade decision for non-existent file."""
    assert should_upgrade_mcap(tmp_path / "mcap") is True


def test_should_upgrade_force(tmp_path):
    """Test force upgrade option."""
    mock_mcap = tmp_path / "mcap"
    mock_mcap.touch()
    assert should_upgrade_mcap(mock_mcap, force=True) is True


def test_should_upgrade_version_comparison(tmp_path):
    """Test version comparison logic."""
    mock_mcap = tmp_path / "mcap"
    mock_mcap.touch()

    # Local version is older - should upgrade
    with (
        patch("owa.cli.mcap.info.get_local_mcap_version", return_value="v0.0.52"),
        patch("owa.cli.mcap.info.get_latest_mcap_cli_version", return_value="v0.0.53"),
    ):
        assert should_upgrade_mcap(mock_mcap) is True

    # Local version is same - should not upgrade
    with (
        patch("owa.cli.mcap.info.get_local_mcap_version", return_value="v0.0.53"),
        patch("owa.cli.mcap.info.get_latest_mcap_cli_version", return_value="v0.0.53"),
    ):
        assert should_upgrade_mcap(mock_mcap) is False

    # Local version is newer - should not upgrade
    with (
        patch("owa.cli.mcap.info.get_local_mcap_version", return_value="v0.0.54"),
        patch("owa.cli.mcap.info.get_latest_mcap_cli_version", return_value="v0.0.53"),
    ):
        assert should_upgrade_mcap(mock_mcap) is False


def test_should_upgrade_unknown_version(tmp_path):
    """Test upgrade decision when local version is unknown."""
    mock_mcap = tmp_path / "mcap"
    mock_mcap.touch()
    with patch("owa.cli.mcap.info.get_local_mcap_version", return_value="unknown"):
        assert should_upgrade_mcap(mock_mcap) is True


def test_version_parsing():
    """Test that version parsing works correctly."""
    v1 = parse_version("0.0.53")
    v2 = parse_version("0.0.54")
    v3 = parse_version("0.0.52")

    assert v2 > v1
    assert v1 > v3
    assert v1 == parse_version("0.0.53")

    # Test version with 'v' prefix
    v_prefix = parse_version("v0.0.53".lstrip("v"))
    assert v_prefix == v1


def test_constants():
    """Test that constants are properly defined."""
    # Current version format
    assert CURRENT_MCAP_CLI_VERSION.startswith("v")
    assert len(CURRENT_MCAP_CLI_VERSION.split(".")) == 3

    # All supported systems have URL templates
    expected_systems = {"linux-amd64", "linux-arm64", "darwin-amd64", "darwin-arm64", "windows-amd64"}
    assert set(MCAP_CLI_DOWNLOAD_URL_TEMPLATES.keys()) == expected_systems

    # URL templates format correctly
    test_version = "v0.0.54"
    for system, template in MCAP_CLI_DOWNLOAD_URL_TEMPLATES.items():
        url = template.format(version=test_version)
        assert test_version in url
        assert url.startswith("https://github.com/foxglove/mcap/releases/download/")
        if system == "windows-amd64":
            assert url.endswith(".exe")
        else:
            assert not url.endswith(".exe")
