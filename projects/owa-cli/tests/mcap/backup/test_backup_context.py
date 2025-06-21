"""
Tests for the BackupContext context manager.

This module tests the BackupContext system used across different CLI commands.
Feature-specific integration tests should only verify that features correctly
use BackupContext, not test BackupContext functionality itself.
"""

from unittest.mock import patch

import pytest
from rich.console import Console

from owa.cli.mcap.backup_utils import BackupContext


class TestBackupContext:
    """Test cases for the BackupContext context manager."""

    def test_backup_context_success(self, temp_dir):
        """Test successful backup context usage."""
        console = Console()

        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"original content")

        with BackupContext(test_file, console) as ctx:
            assert ctx.backup_created is True
            assert ctx.backup_path.exists()

            # Modify the file
            test_file.write_bytes(b"modified content")

        # File should remain modified (no exception occurred)
        assert test_file.read_bytes() == b"modified content"
        assert ctx.backup_path.exists()

    def test_backup_context_auto_rollback_on_exception(self, temp_dir):
        """Test automatic rollback when exception occurs."""
        console = Console()

        test_file = temp_dir / "test.mcap"
        original_content = b"original content"
        test_file.write_bytes(original_content)

        with pytest.raises(ValueError):
            with BackupContext(test_file, console) as ctx:
                # Modify the file
                test_file.write_bytes(b"modified content")

                # Raise an exception to trigger rollback
                raise ValueError("Test exception")

        # File should be restored to original content
        assert test_file.read_bytes() == original_content
        assert not ctx.backup_path.exists()  # Backup deleted after rollback

    def test_backup_context_manual_rollback_only(self, temp_dir):
        """Test context with manual rollback only (no automatic rollback)."""
        console = Console()

        test_file = temp_dir / "test.mcap"
        original_content = b"original content"
        test_file.write_bytes(original_content)

        # Test that automatic rollback always happens on exception
        with pytest.raises(ValueError):
            with BackupContext(test_file, console) as ctx:
                test_file.write_bytes(b"modified content")
                raise ValueError("Test exception")

        # File should be restored (automatic rollback always enabled)
        assert test_file.read_bytes() == original_content
        assert not ctx.backup_path.exists()  # Backup deleted after rollback

    def test_backup_context_manual_rollback(self, temp_dir):
        """Test manual rollback functionality."""
        console = Console()

        test_file = temp_dir / "test.mcap"
        original_content = b"original content"
        test_file.write_bytes(original_content)

        with BackupContext(test_file, console) as ctx:
            test_file.write_bytes(b"modified content")

            # Manual rollback
            result = ctx.rollback(delete_backup=True)
            assert result is True
            assert test_file.read_bytes() == original_content
            assert not ctx.backup_path.exists()

    def test_backup_context_cleanup_backup(self, temp_dir):
        """Test manual backup cleanup."""
        console = Console()

        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        with BackupContext(test_file, console) as ctx:
            assert ctx.backup_path.exists()

            ctx.cleanup_backup()
            assert not ctx.backup_path.exists()

    def test_backup_context_custom_suffix(self, temp_dir):
        """Test backup context with custom suffix."""
        console = Console()

        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        with BackupContext(test_file, console, backup_suffix=".bak") as ctx:
            expected_backup = test_file.with_suffix(f"{test_file.suffix}.bak")
            assert ctx.backup_path == expected_backup
            assert ctx.backup_path.exists()

    def test_backup_context_backup_creation_fails(self, temp_dir):
        """Test context when backup creation fails."""
        console = Console()

        # Create a file that doesn't exist
        nonexistent_file = temp_dir / "nonexistent.mcap"

        with pytest.raises(FileNotFoundError):
            with BackupContext(nonexistent_file, console):
                pass

    def test_backup_context_rollback_no_backup_created(self, temp_dir):
        """Test rollback when no backup was created."""
        console = Console()

        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        ctx = BackupContext(test_file, console)
        # Don't enter context, so no backup is created

        result = ctx.rollback()
        assert result is False

    @patch("owa.cli.mcap.backup_utils.Path.unlink")
    def test_backup_context_cleanup_fails(self, mock_unlink, temp_dir):
        """Test cleanup when backup deletion fails."""
        console = Console()

        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        mock_unlink.side_effect = OSError("Permission denied")

        with BackupContext(test_file, console) as ctx:
            # Should not raise exception when cleanup fails
            ctx.cleanup_backup()

    def test_backup_context_auto_cleanup_on_success(self, temp_dir):
        """Test automatic cleanup when keep_backup=False."""
        console = Console()

        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"original content")

        with BackupContext(test_file, console, keep_backup=False) as ctx:
            test_file.write_bytes(b"modified content")
            # No exception, so backup should be cleaned up automatically

        # Backup should be cleaned up
        assert not ctx.backup_path.exists()
        assert test_file.read_bytes() == b"modified content"

    def test_backup_context_keep_backup_on_success(self, temp_dir):
        """Test keeping backup when keep_backup=True."""
        console = Console()

        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"original content")

        with BackupContext(test_file, console, keep_backup=True) as ctx:
            test_file.write_bytes(b"modified content")
            # No exception, so backup should be kept

        # Backup should still exist
        assert ctx.backup_path.exists()
        assert ctx.backup_path.read_bytes() == b"original content"
        assert test_file.read_bytes() == b"modified content"

    def test_nested_backup_contexts(self, temp_dir):
        """Test nested backup contexts."""
        console = Console()

        test_file = temp_dir / "test.mcap"
        original_content = b"original"
        test_file.write_bytes(original_content)

        with BackupContext(test_file, console, backup_suffix=".outer") as outer_ctx:
            test_file.write_bytes(b"outer modification")

            with BackupContext(test_file, console, backup_suffix=".inner") as inner_ctx:
                test_file.write_bytes(b"inner modification")

                # Both backups should exist
                assert outer_ctx.backup_path.exists()
                assert inner_ctx.backup_path.exists()

                # Inner backup should contain outer modification
                assert inner_ctx.backup_path.read_bytes() == b"outer modification"

        # File should contain inner modification (no exceptions)
        assert test_file.read_bytes() == b"inner modification"
