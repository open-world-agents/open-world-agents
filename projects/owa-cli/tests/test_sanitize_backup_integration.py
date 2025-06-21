"""
Integration tests for sanitize command with unified backup-rollback utilities.

This module tests that the sanitize command correctly uses the unified backup
and rollback utilities and maintains data safety during file operations.
"""

from unittest.mock import patch

from owa.cli.mcap import app as mcap_app


class TestSanitizeBackupIntegration:
    """Integration tests for sanitize command backup functionality."""

    def test_sanitize_creates_backup_on_success(self, temp_dir, cli_runner):
        """Test that sanitize creates backup files on successful operation."""
        # Create a test MCAP file with some content
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

            # Check that backup was created
            backup_file = test_file.with_suffix(".mcap.backup")
            assert backup_file.exists()
            assert backup_file.read_bytes() == b"mock mcap content"

    def test_sanitize_rollback_on_failure(self, temp_dir, cli_runner):
        """Test that sanitize rolls back changes when operation fails."""
        test_file = temp_dir / "test.mcap"
        original_content = b"original mcap content"
        test_file.write_bytes(original_content)

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

            # Check that original file is restored
            assert test_file.read_bytes() == original_content

            # Check that backup was cleaned up
            backup_file = test_file.with_suffix(".mcap.backup")
            assert not backup_file.exists()

    def test_sanitize_backup_cleanup_keep_backups(self, temp_dir, cli_runner):
        """Test backup cleanup when keeping backups."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"mock mcap content")

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

            # Run sanitize with keep-backups
            result = cli_runner.invoke(
                mcap_app, ["sanitize", str(test_file), "--keep-window", "Test Window", "--keep-backups", "--yes"]
            )

            assert result.exit_code == 0

            # Backup should still exist
            backup_file = test_file.with_suffix(".mcap.backup")
            assert backup_file.exists()

    def test_sanitize_backup_cleanup_no_backups(self, temp_dir, cli_runner):
        """Test backup cleanup when not keeping backups."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"mock mcap content")

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

            # Run sanitize without keeping backups
            result = cli_runner.invoke(
                mcap_app, ["sanitize", str(test_file), "--keep-window", "Test Window", "--no-backups", "--yes"]
            )

            assert result.exit_code == 0

            # Backup should be cleaned up
            backup_file = test_file.with_suffix(".mcap.backup")
            assert not backup_file.exists()

    def test_sanitize_multiple_files_backup_handling(self, temp_dir, cli_runner):
        """Test backup handling with multiple files."""
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
                    "--keep-backups",
                    "--yes",
                ],
            )

            assert result.exit_code == 0

            # Both backups should exist
            backup1 = test_file1.with_suffix(".mcap.backup")
            backup2 = test_file2.with_suffix(".mcap.backup")

            assert backup1.exists()
            assert backup2.exists()
            assert backup1.read_bytes() == b"mock mcap content 1"
            assert backup2.read_bytes() == b"mock mcap content 2"

    def test_sanitize_dry_run_no_backup(self, temp_dir, cli_runner):
        """Test that dry run doesn't create backups."""
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

            # No backup should be created
            backup_file = test_file.with_suffix(".mcap.backup")
            assert not backup_file.exists()

            # Original file should be unchanged
            assert test_file.read_bytes() == original_content

    def test_sanitize_backup_already_exists_error(self, temp_dir, cli_runner):
        """Test error handling when backup file already exists."""
        test_file = temp_dir / "test.mcap"
        backup_file = test_file.with_suffix(".mcap.backup")

        test_file.write_bytes(b"original content")
        backup_file.write_bytes(b"existing backup")

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

            # Run sanitize command
            result = cli_runner.invoke(mcap_app, ["sanitize", str(test_file), "--keep-window", "Test Window", "--yes"])

            # Should fail due to existing backup
            assert result.exit_code == 1
            assert "Backup file already exists" in result.output

            # Original files should be unchanged
            assert test_file.read_bytes() == b"original content"
            assert backup_file.read_bytes() == b"existing backup"

    def test_sanitize_uses_unified_backup_path_generation(self, temp_dir, cli_runner):
        """Test that sanitize uses the unified backup path generation."""
        test_file = temp_dir / "complex.name.test.mcap"
        test_file.write_bytes(b"content")

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

            result = cli_runner.invoke(mcap_app, ["sanitize", str(test_file), "--keep-window", "Test Window", "--yes"])

            assert result.exit_code == 0

            # Check that backup uses the unified path generation
            expected_backup = test_file.with_suffix(".mcap.backup")
            assert expected_backup.exists()
            assert expected_backup.name == "complex.name.test.mcap.backup"
