"""Tests for OWA CLI version utilities."""

from unittest.mock import MagicMock, patch

import requests

from owa.cli.utils import check_for_update, get_latest_release, get_local_version


def test_get_local_version_success():
    with patch("importlib.metadata.version", return_value="0.4.1"):
        assert get_local_version("owa.cli") == "0.4.1"


def test_get_local_version_failure():
    with patch("importlib.metadata.version", side_effect=Exception("fail")):
        assert get_local_version("nonexistent") == "unknown"


def test_get_latest_release_success(monkeypatch):
    monkeypatch.delenv("OWA_DISABLE_VERSION_CHECK", raising=False)
    mock_response = MagicMock()
    mock_response.json.return_value = {"tag_name": "v0.4.2"}
    with patch("owa.cli.utils.requests.get", return_value=mock_response):
        assert get_latest_release() == "0.4.2"  # Strips 'v' prefix


def test_check_for_update_disabled_by_env(monkeypatch):
    """Update check returns True (skip) when OWA_DISABLE_VERSION_CHECK is set."""
    monkeypatch.setenv("OWA_DISABLE_VERSION_CHECK", "1")
    with patch("owa.cli.utils.requests.get") as mock_get:
        assert check_for_update() is True
        mock_get.assert_not_called()


def test_check_for_update_up_to_date():
    with (
        patch("owa.cli.utils.get_local_version", return_value="0.4.2"),
        patch("owa.cli.utils.get_latest_release", return_value="0.4.2"),
    ):
        assert check_for_update() is True


def test_check_for_update_newer_available(capsys, monkeypatch):
    monkeypatch.delenv("OWA_DISABLE_VERSION_CHECK", raising=False)
    with (
        patch("owa.cli.utils.get_local_version", return_value="0.4.1"),
        patch("owa.cli.utils.get_latest_release", return_value="0.4.2"),
    ):
        assert check_for_update() is False
        assert "update" in capsys.readouterr().out.lower()


def test_check_for_update_timeout_error(capsys, monkeypatch):
    """Timeout errors are handled gracefully."""
    monkeypatch.delenv("OWA_DISABLE_VERSION_CHECK", raising=False)
    with (
        patch("owa.cli.utils.get_local_version", return_value="0.4.1"),
        patch("owa.cli.utils.get_latest_release", side_effect=requests.Timeout("Timeout")),
    ):
        assert check_for_update() is False
        assert "timeout" in capsys.readouterr().out.lower()


def test_check_for_update_request_error(capsys, monkeypatch):
    """Request errors are handled gracefully."""
    monkeypatch.delenv("OWA_DISABLE_VERSION_CHECK", raising=False)
    with (
        patch("owa.cli.utils.get_local_version", return_value="0.4.1"),
        patch("owa.cli.utils.get_latest_release", side_effect=requests.RequestException("fail")),
    ):
        assert check_for_update() is False
        assert "request failed" in capsys.readouterr().out.lower()
