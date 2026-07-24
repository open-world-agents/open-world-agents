"""Tests for mcap info helper functions."""

from unittest.mock import patch

import pytest

from owa.cli.mcap.info import detect_system, format_duration, format_size, get_mcap_info


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (30.5, "30.5s"),
        (90.0, "1m30.0s"),
        (3661.5, "1h1m1.5s"),
        (0.1, "0.1s"),
        (7200, "2h0m0.0s"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


@pytest.mark.parametrize(
    "size_bytes,expected",
    [
        (100, "100 B"),
        (1024, "1.0 KiB"),
        (1536, "1.5 KiB"),
        (1048576, "1.0 MiB"),
        (1073741824, "1.0 GiB"),
    ],
)
def test_format_size(size_bytes, expected):
    assert format_size(size_bytes) == expected


def test_detect_system():
    """Test system detection returns valid platform key."""
    with patch("platform.system") as mock_sys, patch("platform.machine") as mock_arch:
        mock_sys.return_value = "Linux"
        mock_arch.return_value = "x86_64"
        assert detect_system() == "linux-amd64"

        mock_sys.return_value = "Darwin"
        mock_arch.return_value = "arm64"
        assert detect_system() == "darwin-arm64"

        mock_sys.return_value = "Windows"
        mock_arch.return_value = "AMD64"
        assert detect_system() == "windows-amd64"


def test_detect_system_unsupported_os():
    with patch("platform.system", return_value="FreeBSD"):
        with pytest.raises(RuntimeError, match="Unsupported OS"):
            detect_system()


def test_detect_system_unsupported_arch():
    with patch("platform.system", return_value="Linux"), patch("platform.machine", return_value="mips"):
        with pytest.raises(RuntimeError, match="Unsupported architecture"):
            detect_system()


def test_get_mcap_info(test_data_dir, suppress_mcap_warnings):
    """Test get_mcap_info extracts correct info from real MCAP file."""
    test_file = test_data_dir / "0.4.2.mcap"
    info = get_mcap_info(test_file)

    assert info["file_path"] == test_file
    assert info["messages"] > 0
    assert info["duration_seconds"] >= 0
    assert info["size_bytes"] > 0
    assert info["channels"] > 0
    assert info["has_mkv"] is True
    assert info["mkv_size_bytes"] > 0
