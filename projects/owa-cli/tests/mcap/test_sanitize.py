"""Tests for sanitize command."""

from unittest.mock import Mock, patch

from mcap_owa.highlevel import OWAMcapReader
from owa.cli.mcap import app as mcap_app
from owa.cli.mcap.sanitize import window_matches_target


def _window_msg(title, ts=1000):
    return Mock(topic="window", decoded=Mock(title=title), timestamp=ts)


def _key_msg(key, ts=1001):
    return Mock(topic="keyboard", decoded={"key": key}, timestamp=ts)


# === Unit Tests ===
def test_window_matches_target_exact():
    assert window_matches_target("Notepad", "Notepad", exact_match=True)
    assert not window_matches_target("Notepad++", "Notepad", exact_match=True)


def test_window_matches_target_substring():
    assert window_matches_target("Notepad++", "Notepad", exact_match=False)
    assert window_matches_target("My Notepad App", "notepad", exact_match=False)
    assert not window_matches_target("Browser", "Notepad", exact_match=False)


# === CLI Tests with Mocks ===
def test_sanitize_success(tmp_path, cli_runner):
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")

    with (
        patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader,
        patch("owa.cli.mcap.sanitize.OWAMcapWriter") as mock_writer,
    ):
        messages = [_window_msg("Test"), _key_msg("a")]
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = messages
        result = cli_runner.invoke(mcap_app, ["sanitize", str(test_file), "--keep-window", "Test", "--yes"])

        # Verify writer was called and received messages
        mock_writer.return_value.__enter__.return_value.write_message.assert_called()
    assert result.exit_code == 0
    assert "Successful: 1" in result.output


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


# === Multiple Files Test ===
def test_sanitize_multiple_files(tmp_path, cli_runner):
    """Test sanitization with multiple files."""
    test_file1 = tmp_path / "test1.mcap"
    test_file2 = tmp_path / "test2.mcap"
    test_file1.write_bytes(b"content1")
    test_file2.write_bytes(b"content2")

    with (
        patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader,
        patch("owa.cli.mcap.sanitize.OWAMcapWriter") as mock_writer,
    ):
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = [
            _window_msg("Test"),
            _key_msg("a"),
        ]
        result = cli_runner.invoke(
            mcap_app, ["sanitize", str(test_file1), str(test_file2), "--keep-window", "Test", "--yes"]
        )
        # Writer should be called for both files (2 files × 2 messages each)
        assert mock_writer.return_value.__enter__.return_value.write_message.call_count == 4
    assert result.exit_code == 0


def test_sanitize_exact_matching(tmp_path, cli_runner):
    """Test exact window matching with --exact flag."""
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")

    with (
        patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader,
        patch("owa.cli.mcap.sanitize.OWAMcapWriter") as mock_writer,
    ):
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = [
            _window_msg("Exact Window"),
            _key_msg("a"),
        ]
        result = cli_runner.invoke(
            mcap_app, ["sanitize", str(test_file), "--keep-window", "Exact Window", "--exact", "--yes"]
        )
        # Verify messages were written
        assert mock_writer.return_value.__enter__.return_value.write_message.call_count == 2
    assert result.exit_code == 0


def test_sanitize_substring_matching(tmp_path, cli_runner):
    """Test substring window matching with --substring flag."""
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")

    with (
        patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader,
        patch("owa.cli.mcap.sanitize.OWAMcapWriter") as mock_writer,
    ):
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = [
            _window_msg("My Test Window Application"),
            _key_msg("a"),
        ]
        result = cli_runner.invoke(
            mcap_app, ["sanitize", str(test_file), "--keep-window", "Test Window", "--substring", "--yes"]
        )
        # Verify messages were written (substring match should keep both)
        assert mock_writer.return_value.__enter__.return_value.write_message.call_count == 2
    assert result.exit_code == 0


def test_sanitize_window_filtering(tmp_path, cli_runner):
    """Test window filtering keeps multiple messages after window activation."""
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")

    with (
        patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader,
        patch("owa.cli.mcap.sanitize.OWAMcapWriter") as mock_writer,
    ):
        messages = [
            _window_msg("Keep This Window", 1000),
            _key_msg("a", 1001),
            _key_msg("b", 1002),
            _key_msg("c", 1003),
            _key_msg("d", 1004),
        ]
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = messages
        result = cli_runner.invoke(
            mcap_app, ["sanitize", str(test_file), "--keep-window", "Keep This Window", "--yes"]
        )

        # Verify all 5 messages were written (window + 4 keyboard events)
        writer_instance = mock_writer.return_value.__enter__.return_value
        assert writer_instance.write_message.call_count == 5
    assert result.exit_code == 0


# === Real File Tests ===
def test_sanitize_real_file_dry_run(test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings, cli_runner):
    """Test sanitize dry run with real MCAP file to verify message counting works."""
    test_file = copy_test_file(test_data_dir, "0.4.2.mcap", tmp_path)
    original_size = test_file.stat().st_size

    # Get window titles from the real file to find a valid keep-window value
    with OWAMcapReader(test_file) as reader:
        messages = list(reader.iter_messages())
        window_titles = set()
        for msg in messages:
            if msg.topic == "window":
                if hasattr(msg.decoded, "title"):
                    window_titles.add(msg.decoded.title)
                elif isinstance(msg.decoded, dict):
                    window_titles.add(msg.decoded.get("title", ""))

    # Use first window title found, or a substring match if none
    keep_window = next(iter(window_titles), "")[:10] if window_titles else "test"

    result = cli_runner.invoke(
        mcap_app,
        [
            "sanitize",
            str(test_file),
            "--keep-window",
            keep_window,
            "--dry-run",
            "--verbose",
            "--max-removal-ratio",
            "1.0",
        ],
    )
    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    assert test_file.stat().st_size == original_size  # File unchanged


def test_sanitize_safety_check(tmp_path, cli_runner):
    """Test that safety check blocks excessive removal."""
    test_file = tmp_path / "test.mcap"
    test_file.write_bytes(b"content")

    # Create messages where most will be removed (current window is "Other", not "Main")
    messages = [
        _window_msg("Other", 1000),
        _key_msg("a", 1001),
        _key_msg("b", 1002),
        _key_msg("c", 1003),
        _key_msg("d", 1004),
        _window_msg("Main", 5000),
        _key_msg("e", 5001),
    ]
    with patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader:
        mock_reader.return_value.__enter__.return_value.iter_messages.return_value = messages
        # Default max_removal_ratio is 0.2, this should exceed it
        result = cli_runner.invoke(mcap_app, ["sanitize", str(test_file), "--keep-window", "Main", "--yes"])
    assert result.exit_code == 1
    assert "Safety check failed" in result.output or "removal ratio" in result.output
