"""
Tests for OWA CLI version checking utilities.
"""

from unittest.mock import MagicMock, patch

import requests

from owa.cli.utils import check_for_update, get_latest_release, get_local_version


def test_get_local_version_success():
    """Test successful retrieval of local package version."""
    with patch("importlib.metadata.version", return_value="0.4.1"):
        assert get_local_version("owa.cli") == "0.4.1"


def test_get_local_version_failure():
    """Test handling of failed local version retrieval."""
    with patch("importlib.metadata.version", side_effect=Exception("Package not found")):
        assert get_local_version("nonexistent.package") == "unknown"


def test_get_latest_release_success(monkeypatch):
    """Test successful retrieval of latest release from GitHub API."""
    monkeypatch.delenv("OWA_DISABLE_VERSION_CHECK", raising=False)
    mock_response = MagicMock()
    mock_response.json.return_value = {"tag_name": "v0.4.2"}
    mock_response.raise_for_status.return_value = None

    with patch("owa.cli.utils.requests.get", return_value=mock_response):
        version = get_latest_release()
        assert version == "0.4.2"  # Should strip the 'v' prefix


def test_get_latest_release_with_no_v_prefix(monkeypatch):
    """Test handling of release tags without 'v' prefix."""
    monkeypatch.delenv("OWA_DISABLE_VERSION_CHECK", raising=False)
    mock_response = MagicMock()
    mock_response.json.return_value = {"tag_name": "0.4.2"}
    mock_response.raise_for_status.return_value = None

    with patch("owa.cli.utils.requests.get", return_value=mock_response):
        version = get_latest_release()
        assert version == "0.4.2"


def test_check_for_update_disabled_by_env(monkeypatch):
    """Test that update check is disabled when OWA_DISABLE_VERSION_CHECK is set."""
    monkeypatch.setenv("OWA_DISABLE_VERSION_CHECK", "1")
    with patch("owa.cli.utils.requests.get") as mock_get:
        result = check_for_update()
        assert result is True
        mock_get.assert_not_called()


def test_check_for_update_up_to_date():
    """Test update check when local version is up to date."""
    with (
        patch("owa.cli.utils.get_local_version", return_value="0.4.2"),
        patch("owa.cli.utils.get_latest_release", return_value="0.4.2"),
    ):
        assert check_for_update() is True


def test_check_for_update_newer_available(capsys, monkeypatch):
    """Test update check when newer version is available."""
    monkeypatch.delenv("OWA_DISABLE_VERSION_CHECK", raising=False)
    with (
        patch("owa.cli.utils.get_local_version", return_value="0.4.1"),
        patch("owa.cli.utils.get_latest_release", return_value="0.4.2"),
    ):
        result = check_for_update()
        assert result is False

        # Check that update message was printed
        captured = capsys.readouterr()
        assert "update" in captured.out.lower()


def test_check_for_update_timeout_error(capsys, monkeypatch):
    """Test that timeout errors are handled gracefully."""
    monkeypatch.delenv("OWA_DISABLE_VERSION_CHECK", raising=False)
    with (
        patch("owa.cli.utils.get_local_version", return_value="0.4.1"),
        patch("owa.cli.utils.get_latest_release", side_effect=requests.Timeout("Connection timeout")),
    ):
        result = check_for_update()
        assert result is False
        assert "timeout" in capsys.readouterr().out.lower()


def test_check_for_update_request_error(capsys, monkeypatch):
    """Test that request errors are handled gracefully."""
    monkeypatch.delenv("OWA_DISABLE_VERSION_CHECK", raising=False)
    with (
        patch("owa.cli.utils.get_local_version", return_value="0.4.1"),
        patch("owa.cli.utils.get_latest_release", side_effect=requests.RequestException("Network error")),
    ):
        result = check_for_update()
        assert result is False
        assert "request failed" in capsys.readouterr().out.lower()


def test_check_for_update_general_error(capsys, monkeypatch):
    """Test that general errors are handled gracefully."""
    monkeypatch.delenv("OWA_DISABLE_VERSION_CHECK", raising=False)
    with (
        patch("owa.cli.utils.get_local_version", return_value="0.4.1"),
        patch("owa.cli.utils.get_latest_release", side_effect=Exception("Unexpected error")),
    ):
        result = check_for_update()
        assert result is False
        captured = capsys.readouterr()
        assert "error" in captured.out.lower() or "failed" in captured.out.lower()
