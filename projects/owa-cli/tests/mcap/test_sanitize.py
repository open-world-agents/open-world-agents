"""
Tests for sanitize command functionality.

This module tests the core sanitization functionality of the sanitize command.
"""

from unittest.mock import patch

from owa.cli.mcap import app as mcap_app


class TestSanitizeIntegration:
    """Integration tests for sanitize command core functionality."""

    def test_sanitize_successful_operation(self, temp_dir, cli_runner):
        """Test successful sanitization operation."""
        test_file = temp_dir / "test.mcap"

        # Mock the MCAP reading/writing to avoid actual MCAP dependencies
        with (
            patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader,
            patch("owa.cli.mcap.sanitize.OWAMcapWriter"),
        ):
            # Mock reader to return some test messages
            mock_reader_instance = mock_reader.return_value.__enter__.return_value
            mock_reader_instance.iter_messages.return_value = [
                # Mock window message
                type(
                    "MockMessage",
                    (),
                    {
                        "topic": "window",
                        "decoded": type("MockDecoded", (), {"title": "Test Window"})(),
                        "timestamp": 1000,
                    },
                )(),
                # Mock other message
                type("MockMessage", (), {"topic": "keyboard", "decoded": {"key": "a"}, "timestamp": 1001})(),
            ]

            # Create the test file
            test_file.write_bytes(b"mock mcap content")

            # Run sanitize command
            result = cli_runner.invoke(
                mcap_app,
                [
                    "sanitize",
                    str(test_file),
                    "--keep-window",
                    "Test Window",
                    "--yes",  # Skip confirmation
                ],
            )

            assert result.exit_code == 0

    def test_sanitize_failed_operation(self, temp_dir, cli_runner):
        """Test failed sanitization operation."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"original mcap content")

        with (
            patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader,
            patch("owa.cli.mcap.sanitize.OWAMcapWriter") as mock_writer,
        ):
            # Mock reader to work initially
            mock_reader_instance = mock_reader.return_value.__enter__.return_value
            mock_reader_instance.iter_messages.return_value = [
                type(
                    "MockMessage",
                    (),
                    {
                        "topic": "window",
                        "decoded": type("MockDecoded", (), {"title": "Test Window"})(),
                        "timestamp": 1000,
                    },
                )(),
            ]

            # Mock writer to fail during writing
            mock_writer.return_value.__enter__.side_effect = Exception("Write failed")

            # Run sanitize command
            result = cli_runner.invoke(mcap_app, ["sanitize", str(test_file), "--keep-window", "Test Window", "--yes"])

            assert result.exit_code == 1

    def test_sanitize_multiple_files(self, temp_dir, cli_runner):
        """Test sanitization with multiple files."""
        test_file1 = temp_dir / "test1.mcap"
        test_file2 = temp_dir / "test2.mcap"

        test_file1.write_bytes(b"mock mcap content 1")
        test_file2.write_bytes(b"mock mcap content 2")

        with (
            patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader,
            patch("owa.cli.mcap.sanitize.OWAMcapWriter"),
        ):
            mock_reader_instance = mock_reader.return_value.__enter__.return_value
            mock_reader_instance.iter_messages.return_value = [
                type(
                    "MockMessage",
                    (),
                    {
                        "topic": "window",
                        "decoded": type("MockDecoded", (), {"title": "Test Window"})(),
                        "timestamp": 1000,
                    },
                )(),
            ]

            # Run sanitize on multiple files
            result = cli_runner.invoke(
                mcap_app,
                [
                    "sanitize",
                    str(test_file1),
                    str(test_file2),
                    "--keep-window",
                    "Test Window",
                    "--yes",
                ],
            )

            assert result.exit_code == 0

    def test_sanitize_dry_run(self, temp_dir, cli_runner):
        """Test dry run mode doesn't modify files."""
        test_file = temp_dir / "test.mcap"
        original_content = b"original mcap content"
        test_file.write_bytes(original_content)

        with patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader:
            mock_reader_instance = mock_reader.return_value.__enter__.return_value
            mock_reader_instance.iter_messages.return_value = [
                type(
                    "MockMessage",
                    (),
                    {
                        "topic": "window",
                        "decoded": type("MockDecoded", (), {"title": "Test Window"})(),
                        "timestamp": 1000,
                    },
                )(),
            ]

            # Run sanitize in dry-run mode
            result = cli_runner.invoke(
                mcap_app, ["sanitize", str(test_file), "--keep-window", "Test Window", "--dry-run"]
            )

            assert result.exit_code == 0
            # Original file should be unchanged
            assert test_file.read_bytes() == original_content

    def test_sanitize_window_filtering(self, temp_dir, cli_runner):
        """Test window filtering functionality."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        with (
            patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader,
            patch("owa.cli.mcap.sanitize.OWAMcapWriter"),
        ):
            mock_reader_instance = mock_reader.return_value.__enter__.return_value
            mock_reader_instance.iter_messages.return_value = [
                # Window message that should be kept
                type(
                    "MockMessage",
                    (),
                    {
                        "topic": "window",
                        "decoded": type("MockDecoded", (), {"title": "Keep This Window"})(),
                        "timestamp": 1000,
                    },
                )(),
                # Messages that should be kept (after window activation)
                type("MockMessage", (), {"topic": "keyboard", "decoded": {"key": "a"}, "timestamp": 1001})(),
                type("MockMessage", (), {"topic": "keyboard", "decoded": {"key": "b"}, "timestamp": 1002})(),
                type("MockMessage", (), {"topic": "keyboard", "decoded": {"key": "c"}, "timestamp": 1003})(),
                type("MockMessage", (), {"topic": "keyboard", "decoded": {"key": "d"}, "timestamp": 1004})(),
            ]

            result = cli_runner.invoke(
                mcap_app, ["sanitize", str(test_file), "--keep-window", "Keep This Window", "--yes"]
            )

            assert result.exit_code == 0

    def test_sanitize_exact_window_matching(self, temp_dir, cli_runner):
        """Test exact window matching functionality."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        with (
            patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader,
            patch("owa.cli.mcap.sanitize.OWAMcapWriter"),
        ):
            mock_reader_instance = mock_reader.return_value.__enter__.return_value
            mock_reader_instance.iter_messages.return_value = [
                # Window message with exact title match
                type(
                    "MockMessage",
                    (),
                    {
                        "topic": "window",
                        "decoded": type("MockDecoded", (), {"title": "Exact Window"})(),
                        "timestamp": 1000,
                    },
                )(),
                # Messages that should be kept (after window activation)
                type("MockMessage", (), {"topic": "keyboard", "decoded": {"key": "a"}, "timestamp": 1001})(),
            ]

            result = cli_runner.invoke(
                mcap_app, ["sanitize", str(test_file), "--keep-window", "Exact Window", "--exact", "--yes"]
            )

            assert result.exit_code == 0

    def test_sanitize_substring_window_matching(self, temp_dir, cli_runner):
        """Test substring window matching functionality."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        with (
            patch("owa.cli.mcap.sanitize.OWAMcapReader") as mock_reader,
            patch("owa.cli.mcap.sanitize.OWAMcapWriter"),
        ):
            mock_reader_instance = mock_reader.return_value.__enter__.return_value
            mock_reader_instance.iter_messages.return_value = [
                # Window message with partial title match
                type(
                    "MockMessage",
                    (),
                    {
                        "topic": "window",
                        "decoded": type("MockDecoded", (), {"title": "My Test Window Application"})(),
                        "timestamp": 1000,
                    },
                )(),
                # Messages that should be kept (after window activation)
                type("MockMessage", (), {"topic": "keyboard", "decoded": {"key": "a"}, "timestamp": 1001})(),
            ]

            result = cli_runner.invoke(
                mcap_app, ["sanitize", str(test_file), "--keep-window", "Test Window", "--substring", "--yes"]
            )

            assert result.exit_code == 0
