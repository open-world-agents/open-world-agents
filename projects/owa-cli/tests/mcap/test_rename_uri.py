"""Tests for rename_uri command."""

from unittest.mock import Mock, patch

from owa.cli.mcap import app as mcap_app


def _screen_msg(uri="video.mkv"):
    return Mock(topic="screen", decoded=Mock(media_ref=Mock(uri=uri, pts_ns=123)), timestamp=1000)


def test_rename_uri_success(tmp_path, cli_runner):
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")

    with (
        patch("owa.cli.mcap.rename_uri.OWAMcapReader") as mock_reader,
        patch("owa.cli.mcap.rename_uri.OWAMcapWriter") as mock_writer,
        patch("owa.cli.mcap.rename_uri.MediaRef") as mock_media_ref,
    ):
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = [_screen_msg()]
        result = cli_runner.invoke(mcap_app, ["rename-uri", str(test_file), "--uri", "new.mkv", "--yes"])

        # Verify writer was called
        mock_writer.return_value.__enter__.return_value.write_message.assert_called()
        # Verify MediaRef was created with new URI
        mock_media_ref.assert_called()
    assert result.exit_code == 0
    assert "Success" in result.output or "Renamed" in result.output or "1" in result.output


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


def test_rename_uri_multiple_files(tmp_path, cli_runner):
    """Test URI renaming with multiple files."""
    file1 = tmp_path / "test1.mcap"
    file2 = tmp_path / "test2.mcap"
    file1.write_bytes(b"content1")
    file2.write_bytes(b"content2")

    with (
        patch("owa.cli.mcap.rename_uri.OWAMcapReader") as mock_reader,
        patch("owa.cli.mcap.rename_uri.OWAMcapWriter") as mock_writer,
        patch("owa.cli.mcap.rename_uri.MediaRef") as mock_media_ref,
    ):
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = [_screen_msg()]
        result = cli_runner.invoke(mcap_app, ["rename-uri", str(file1), str(file2), "--uri", "new.mkv", "--yes"])

        # Both files should have been processed
        assert mock_writer.return_value.__enter__.return_value.write_message.call_count == 2
        assert mock_media_ref.call_count == 2
    assert result.exit_code == 0


def test_rename_uri_with_mixed_message_types(tmp_path, cli_runner):
    """Test behavior with mixed message types (screen + keyboard)."""
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")

    with (
        patch("owa.cli.mcap.rename_uri.OWAMcapReader") as mock_reader,
        patch("owa.cli.mcap.rename_uri.OWAMcapWriter") as mock_writer,
        patch("owa.cli.mcap.rename_uri.MediaRef") as mock_media_ref,
    ):
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = [
            _screen_msg(),
            Mock(topic="keyboard", decoded=Mock(key="a"), timestamp=1001),
        ]
        result = cli_runner.invoke(mcap_app, ["rename-uri", str(test_file), "--uri", "new.mkv", "--yes"])

        # Both messages should be written, but only screen message gets MediaRef update
        assert mock_writer.return_value.__enter__.return_value.write_message.call_count == 2
        assert mock_media_ref.call_count == 1  # Only screen message has media_ref
    assert result.exit_code == 0


def test_rename_uri_with_empty_mcap_file(tmp_path, cli_runner):
    """Test behavior with empty MCAP file (no messages)."""
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")

    with (
        patch("owa.cli.mcap.rename_uri.OWAMcapReader") as mock_reader,
        patch("owa.cli.mcap.rename_uri.OWAMcapWriter") as mock_writer,
    ):
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = []
        result = cli_runner.invoke(mcap_app, ["rename-uri", str(test_file), "--uri", "new.mkv", "--yes"])

        # No messages to write
        mock_writer.return_value.__enter__.return_value.write_message.assert_not_called()
    assert result.exit_code == 0
