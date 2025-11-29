"""Tests for rename_uri command."""

from unittest.mock import MagicMock, Mock, patch

from owa.cli.mcap import app as mcap_app


def _screen_msg(uri="video.mkv"):
    return Mock(topic="screen", decoded=Mock(media_ref=Mock(uri=uri, pts_ns=123)), timestamp=1000)


def test_rename_uri_success(tmp_path, cli_runner):
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")

    with (
        patch("owa.cli.mcap.rename_uri.OWAMcapReader") as mock_reader,
        patch("owa.cli.mcap.rename_uri.OWAMcapWriter"),
        patch("owa.cli.mcap.rename_uri.MediaRef"),
    ):
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = [_screen_msg()]
        result = cli_runner.invoke(mcap_app, ["rename-uri", str(test_file), "--uri", "new.mkv", "--yes"])
    assert result.exit_code == 0


def test_rename_uri_failure(tmp_path, cli_runner):
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")

    with (
        patch("owa.cli.mcap.rename_uri.OWAMcapReader") as mock_reader,
        patch("owa.cli.mcap.rename_uri.OWAMcapWriter") as mock_writer,
    ):
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = [_screen_msg()]
        mock_writer.return_value.__enter__.side_effect = Exception("fail")
        result = cli_runner.invoke(mcap_app, ["rename-uri", str(test_file), "--uri", "new.mkv", "--yes"])
    assert result.exit_code == 1


def test_rename_uri_dry_run(tmp_path, cli_runner):
    test_file = tmp_path / "test.mcap"
    original = b"content"
    test_file.write_bytes(original)

    with patch("owa.cli.mcap.rename_uri.OWAMcapReader") as mock_reader:
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = [_screen_msg()]
        result = cli_runner.invoke(mcap_app, ["rename-uri", str(test_file), "--uri", "new.mkv", "--dry-run"])
    assert result.exit_code == 0
    assert test_file.read_bytes() == original


def test_rename_uri_empty_uri(tmp_path, cli_runner):
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")
    result = cli_runner.invoke(mcap_app, ["rename-uri", str(test_file), "--uri", ""])
    assert result.exit_code == 1
