"""Tests for sanitize command."""

from unittest.mock import Mock, patch

from owa.cli.mcap import app as mcap_app


def _window_msg(title, ts=1000):
    return Mock(topic="window", decoded=Mock(title=title), timestamp=ts)


def _key_msg(key, ts=1001):
    return Mock(topic="keyboard", decoded={"key": key}, timestamp=ts)


def test_sanitize_success(tmp_path, cli_runner):
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")

    with (
        patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader,
        patch("owa.cli.mcap.sanitize.OWAMcapWriter"),
    ):
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = [
            _window_msg("Test"),
            _key_msg("a"),
        ]
        result = cli_runner.invoke(mcap_app, ["sanitize", str(test_file), "--keep-window", "Test", "--yes"])
    assert result.exit_code == 0


def test_sanitize_failure(tmp_path, cli_runner):
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")

    with (
        patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader,
        patch("owa.cli.mcap.sanitize.OWAMcapWriter") as mock_writer,
    ):
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = [_window_msg("Test")]
        mock_writer.return_value.__enter__.side_effect = Exception("fail")
        result = cli_runner.invoke(mcap_app, ["sanitize", str(test_file), "--keep-window", "Test", "--yes"])
    assert result.exit_code == 1


def test_sanitize_dry_run(tmp_path, cli_runner):
    test_file = tmp_path / "test.mcap"
    original = b"content"
    test_file.write_bytes(original)

    with patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader:
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = [_window_msg("Test")]
        result = cli_runner.invoke(mcap_app, ["sanitize", str(test_file), "--keep-window", "Test", "--dry-run"])
    assert result.exit_code == 0
    assert test_file.read_bytes() == original


def test_sanitize_auto_detect(tmp_path, cli_runner):
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")

    messages = [
        _window_msg("Main", 1000),
        _key_msg("a", 1001),
        _window_msg("Main", 2000),
        _window_msg("Main", 3000),
        _window_msg("Other", 4000),
    ]
    with (
        patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader,
        patch("owa.cli.mcap.sanitize.OWAMcapWriter"),
    ):
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = messages
        result = cli_runner.invoke(
            mcap_app, ["sanitize", str(test_file), "--auto-detect-window", "--max-removal-ratio", "0.5", "--yes"]
        )
    assert result.exit_code == 0
    assert "Main" in result.output


def test_sanitize_validation_errors(tmp_path, cli_runner):
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")

    # Both options
    result = cli_runner.invoke(mcap_app, ["sanitize", str(test_file), "--keep-window", "X", "--auto-detect-window"])
    assert result.exit_code == 1

    # Neither option
    result = cli_runner.invoke(mcap_app, ["sanitize", str(test_file)])
    assert result.exit_code == 1
