"""Tests for mcap CLI upgrade functionality."""

from unittest.mock import MagicMock, patch

from owa.cli.mcap.info import (
    CURRENT_MCAP_CLI_VERSION,
    MCAP_CLI_DOWNLOAD_URL_TEMPLATES,
    get_latest_mcap_cli_version,
    get_local_mcap_version,
    should_upgrade_mcap,
)


def test_get_latest_version_fallback():
    with patch("owa.cli.mcap.info.requests.get", side_effect=Exception("fail")):
        assert get_latest_mcap_cli_version() == CURRENT_MCAP_CLI_VERSION


def test_get_local_version_success(tmp_path):
    mock_mcap = tmp_path / "mcap"
    mock_mcap.touch()

    with patch("owa.cli.mcap.info.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="v0.0.53\n")
        assert get_local_mcap_version(mock_mcap) == "v0.0.53"


def test_get_local_version_failure(tmp_path):
    mock_mcap = tmp_path / "mcap"
    mock_mcap.touch()

    with patch("owa.cli.mcap.info.subprocess.run", side_effect=FileNotFoundError()):
        assert get_local_mcap_version(mock_mcap) == "unknown"


def test_should_upgrade_nonexistent(tmp_path):
    assert should_upgrade_mcap(tmp_path / "mcap") is True


def test_should_upgrade_force(tmp_path):
    mock_mcap = tmp_path / "mcap"
    mock_mcap.touch()
    assert should_upgrade_mcap(mock_mcap, force=True) is True


def test_should_upgrade_version_comparison(tmp_path):
    mock_mcap = tmp_path / "mcap"
    mock_mcap.touch()

    with (
        patch("owa.cli.mcap.info.get_local_mcap_version", return_value="v0.0.52"),
        patch("owa.cli.mcap.info.get_latest_mcap_cli_version", return_value="v0.0.53"),
    ):
        assert should_upgrade_mcap(mock_mcap) is True

    with (
        patch("owa.cli.mcap.info.get_local_mcap_version", return_value="v0.0.53"),
        patch("owa.cli.mcap.info.get_latest_mcap_cli_version", return_value="v0.0.53"),
    ):
        assert should_upgrade_mcap(mock_mcap) is False


def test_should_upgrade_unknown_version(tmp_path):
    mock_mcap = tmp_path / "mcap"
    mock_mcap.touch()
    with patch("owa.cli.mcap.info.get_local_mcap_version", return_value="unknown"):
        assert should_upgrade_mcap(mock_mcap) is True


def test_constants():
    assert CURRENT_MCAP_CLI_VERSION.startswith("v")
    assert len(MCAP_CLI_DOWNLOAD_URL_TEMPLATES) == 5
    for template in MCAP_CLI_DOWNLOAD_URL_TEMPLATES.values():
        assert "github.com/foxglove/mcap" in template.format(version="v0.0.54")
