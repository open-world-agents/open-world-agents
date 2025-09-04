"""
Tests for rename_uri command functionality.

This module tests the core URI renaming functionality of the rename_uri command.
"""

from unittest.mock import Mock, patch

from owa.cli.mcap import app as mcap_app


class TestRenameUriIntegration:
    """Integration tests for rename_uri command core functionality."""

    def test_rename_uri_successful_operation(self, tmp_path, cli_runner):
        """Test successful URI renaming operation."""
        test_file = tmp_path / "test.mcap"

        # Mock the MCAP reading/writing to avoid actual MCAP dependencies
        with (
            patch("owa.cli.mcap.rename_uri.OWAMcapReader") as mock_reader,
            patch("owa.cli.mcap.rename_uri.OWAMcapWriter"),
            patch("owa.cli.mcap.rename_uri.MediaRef"),
        ):
            # Mock reader to return some test messages
            mock_reader_instance = mock_reader.return_value.__enter__.return_value

            # Create mock objects using Mock for better readability
            mock_media_ref = Mock(uri="any_video.mkv", pts_ns=123456)
            mock_decoded_screen = Mock(media_ref=mock_media_ref)
            mock_screen_message = Mock(topic="screen", decoded=mock_decoded_screen, timestamp=1000)

            mock_decoded_keyboard = Mock(key="a")
            mock_keyboard_message = Mock(topic="keyboard", decoded=mock_decoded_keyboard, timestamp=1001)

            mock_reader_instance.iter_messages.return_value = [
                mock_screen_message,
                mock_keyboard_message,
            ]

            # Create the test file
            test_file.write_bytes(b"mock mcap content")

            # Run rename_uri command
            result = cli_runner.invoke(
                mcap_app,
                [
                    "rename-uri",
                    str(test_file),
                    "--uri",
                    "new_video.mkv",
                    "--yes",  # Skip confirmation
                ],
            )

            assert result.exit_code == 0

    def test_rename_uri_failed_operation(self, tmp_path, cli_runner):
        """Test failed URI renaming operation."""
        test_file = tmp_path / "test.mcap"
        test_file.write_bytes(b"original mcap content")

        with (
            patch("owa.cli.mcap.rename_uri.OWAMcapReader") as mock_reader,
            patch("owa.cli.mcap.rename_uri.OWAMcapWriter") as mock_writer,
        ):
            # Mock reader to work initially
            mock_reader_instance = mock_reader.return_value.__enter__.return_value

            # Create mock objects using Mock for better readability
            mock_media_ref = Mock(uri="any_video.mkv", pts_ns=123456)
            mock_decoded_screen = Mock(media_ref=mock_media_ref)
            mock_screen_message = Mock(topic="screen", decoded=mock_decoded_screen, timestamp=1000)

            mock_reader_instance.iter_messages.return_value = [
                mock_screen_message,
            ]

            # Mock writer to fail during writing
            mock_writer.return_value.__enter__.side_effect = Exception("Write failed")

            # Run rename_uri command
            result = cli_runner.invoke(
                mcap_app,
                ["rename-uri", str(test_file), "--uri", "new_video.mkv", "--yes"],
            )

            assert result.exit_code == 1

    def test_rename_uri_multiple_files(self, tmp_path, cli_runner):
        """Test URI renaming with multiple files."""
        test_file1 = tmp_path / "test1.mcap"
        test_file2 = tmp_path / "test2.mcap"

        test_file1.write_bytes(b"mock mcap content 1")
        test_file2.write_bytes(b"mock mcap content 2")

        with (
            patch("owa.cli.mcap.rename_uri.OWAMcapReader") as mock_reader,
            patch("owa.cli.mcap.rename_uri.OWAMcapWriter"),
            patch("owa.cli.mcap.rename_uri.MediaRef"),
        ):
            mock_reader_instance = mock_reader.return_value.__enter__.return_value

            # Create mock objects using Mock for better readability
            mock_media_ref = Mock(uri="any_video.mkv", pts_ns=123456)
            mock_decoded_screen = Mock(media_ref=mock_media_ref)
            mock_screen_message = Mock(topic="screen", decoded=mock_decoded_screen, timestamp=1000)

            mock_reader_instance.iter_messages.return_value = [
                mock_screen_message,
            ]

            # Run rename_uri on multiple files
            result = cli_runner.invoke(
                mcap_app,
                [
                    "rename-uri",
                    str(test_file1),
                    str(test_file2),
                    "--uri",
                    "new_video.mkv",
                    "--yes",
                ],
            )

            assert result.exit_code == 0

    def test_rename_uri_dry_run(self, tmp_path, cli_runner):
        """Test dry run mode doesn't modify files."""
        test_file = tmp_path / "test.mcap"
        original_content = b"original mcap content"
        test_file.write_bytes(original_content)

        with patch("owa.cli.mcap.rename_uri.OWAMcapReader") as mock_reader:
            mock_reader_instance = mock_reader.return_value.__enter__.return_value

            # Create mock objects using Mock for better readability
            mock_media_ref = Mock(uri="any_video.mkv", pts_ns=123456)
            mock_decoded_screen = Mock(media_ref=mock_media_ref)
            mock_screen_message = Mock(topic="screen", decoded=mock_decoded_screen, timestamp=1000)

            mock_reader_instance.iter_messages.return_value = [
                mock_screen_message,
            ]

            # Run rename_uri in dry-run mode
            result = cli_runner.invoke(
                mcap_app,
                [
                    "rename-uri",
                    str(test_file),
                    "--uri",
                    "new_video.mkv",
                    "--dry-run",
                ],
            )

            assert result.exit_code == 0
            # Original file should be unchanged
            assert test_file.read_bytes() == original_content

    def test_rename_uri_with_mixed_message_types(self, tmp_path, cli_runner):
        """Test behavior with mixed message types."""
        test_file = tmp_path / "test.mcap"
        test_file.write_bytes(b"content")

        with (
            patch("owa.cli.mcap.rename_uri.OWAMcapReader") as mock_reader,
            patch("owa.cli.mcap.rename_uri.OWAMcapWriter"),
        ):
            mock_reader_instance = mock_reader.return_value.__enter__.return_value

            # Create mock objects using Mock for better readability
            mock_media_ref = Mock(uri="different_video.mkv", pts_ns=123456)
            mock_decoded_screen = Mock(media_ref=mock_media_ref)
            mock_screen_message = Mock(topic="screen", decoded=mock_decoded_screen, timestamp=1000)

            mock_decoded_keyboard = Mock(key="a")
            mock_keyboard_message = Mock(topic="keyboard", decoded=mock_decoded_keyboard, timestamp=1001)

            mock_reader_instance.iter_messages.return_value = [
                mock_screen_message,
                mock_keyboard_message,
            ]

            result = cli_runner.invoke(
                mcap_app,
                [
                    "rename-uri",
                    str(test_file),
                    "--uri",
                    "new_video.mkv",
                    "--yes",
                ],
            )

            assert result.exit_code == 0

    def test_rename_uri_with_empty_mcap_file(self, tmp_path, cli_runner):
        """Test behavior with empty MCAP file."""
        test_file = tmp_path / "test.mcap"
        test_file.write_bytes(b"content")

        with (
            patch("owa.cli.mcap.rename_uri.OWAMcapReader") as mock_reader,
            patch("owa.cli.mcap.rename_uri.OWAMcapWriter"),
        ):
            mock_reader_instance = mock_reader.return_value.__enter__.return_value
            mock_reader_instance.iter_messages.return_value = []

            result = cli_runner.invoke(
                mcap_app,
                ["rename-uri", str(test_file), "--uri", "new_video.mkv", "--yes"],
            )

            assert result.exit_code == 0

    def test_rename_uri_empty_uris(self, tmp_path, cli_runner):
        """Test behavior with empty URI parameters."""
        test_file = tmp_path / "test.mcap"
        test_file.write_bytes(b"content")

        # Test empty URI
        result = cli_runner.invoke(mcap_app, ["rename-uri", str(test_file), "--uri", ""])
        assert result.exit_code == 1
