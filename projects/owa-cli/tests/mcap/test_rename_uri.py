"""
Tests for rename_uri command functionality.

This module tests the core URI renaming functionality of the rename_uri command.
"""

from unittest.mock import patch

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
            mock_reader_instance.iter_messages.return_value = [
                # Mock screen message with media_ref
                type(
                    "MockMessage",
                    (),
                    {
                        "topic": "screen",
                        "decoded": type(
                            "MockDecoded",
                            (),
                            {"media_ref": type("MockMediaRef", (), {"uri": "any_video.mkv", "pts_ns": 123456})()},
                        )(),
                        "timestamp": 1000,
                    },
                )(),
                # Mock other message
                type("MockMessage", (), {"topic": "keyboard", "decoded": {"key": "a"}, "timestamp": 1001})(),
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
            mock_reader_instance.iter_messages.return_value = [
                type(
                    "MockMessage",
                    (),
                    {
                        "topic": "screen",
                        "decoded": type(
                            "MockDecoded",
                            (),
                            {"media_ref": type("MockMediaRef", (), {"uri": "any_video.mkv", "pts_ns": 123456})()},
                        )(),
                        "timestamp": 1000,
                    },
                )(),
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
            mock_reader_instance.iter_messages.return_value = [
                type(
                    "MockMessage",
                    (),
                    {
                        "topic": "screen",
                        "decoded": type(
                            "MockDecoded",
                            (),
                            {"media_ref": type("MockMediaRef", (), {"uri": "any_video.mkv", "pts_ns": 123456})()},
                        )(),
                        "timestamp": 1000,
                    },
                )(),
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
            mock_reader_instance.iter_messages.return_value = [
                type(
                    "MockMessage",
                    (),
                    {
                        "topic": "screen",
                        "decoded": type(
                            "MockDecoded",
                            (),
                            {"media_ref": type("MockMediaRef", (), {"uri": "any_video.mkv", "pts_ns": 123456})()},
                        )(),
                        "timestamp": 1000,
                    },
                )(),
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

    def test_rename_uri_no_matching_uris(self, tmp_path, cli_runner):
        """Test behavior when no matching URIs are found."""
        test_file = tmp_path / "test.mcap"
        test_file.write_bytes(b"content")

        with (
            patch("owa.cli.mcap.rename_uri.OWAMcapReader") as mock_reader,
            patch("owa.cli.mcap.rename_uri.OWAMcapWriter"),
        ):
            mock_reader_instance = mock_reader.return_value.__enter__.return_value
            mock_reader_instance.iter_messages.return_value = [
                # Screen message with different URI
                type(
                    "MockMessage",
                    (),
                    {
                        "topic": "screen",
                        "decoded": type(
                            "MockDecoded",
                            (),
                            {
                                "media_ref": type(
                                    "MockMediaRef", (), {"uri": "different_video.mkv", "pts_ns": 123456}
                                )()
                            },
                        )(),
                        "timestamp": 1000,
                    },
                )(),
                # Non-screen message
                type("MockMessage", (), {"topic": "keyboard", "decoded": {"key": "a"}, "timestamp": 1001})(),
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

    def test_rename_uri_valid_operation(self, tmp_path, cli_runner):
        """Test behavior with valid URI parameter."""
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
