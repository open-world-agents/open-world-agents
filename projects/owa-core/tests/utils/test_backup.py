"""
Tests for the BackupContext context manager.

This module tests the BackupContext system used across different projects.
"""

from unittest.mock import patch

import pytest
from rich.console import Console

from owa.core.utils.backup import BackupContext, DummyConsole


class TestBackupContext:
    """Test cases for the BackupContext context manager."""

    # Core Context Manager Functionality

    def test_successful_context_with_auto_cleanup(self, tmp_path):
        """Test successful backup context usage with automatic cleanup."""
        console = Console()
        test_file = tmp_path / "test.mcap"
        test_file.write_bytes(b"original content")

        with BackupContext(test_file, console=console) as ctx:
            assert ctx.backup_path.exists()
            assert ctx.backup_path.read_bytes() == b"original content"
            test_file.write_bytes(b"modified content")

        # File should remain modified, backup cleaned up
        assert test_file.read_bytes() == b"modified content"
        assert not ctx.backup_path.exists()

    def test_successful_context_with_keep_backup(self, tmp_path):
        """Test successful backup context usage keeping backup."""
        console = Console()
        test_file = tmp_path / "test.mcap"
        test_file.write_bytes(b"original content")

        with BackupContext(test_file, console=console, keep_backup=True) as ctx:
            test_file.write_bytes(b"modified content")

        # File should remain modified, backup kept
        assert test_file.read_bytes() == b"modified content"
        assert ctx.backup_path.exists()
        assert ctx.backup_path.read_bytes() == b"original content"

    def test_automatic_rollback_on_exception(self, tmp_path):
        """Test automatic rollback when exception occurs."""
        console = Console()
        test_file = tmp_path / "test.mcap"
        original_content = b"original content"
        test_file.write_bytes(original_content)

        backup_path = BackupContext.find_backup_path(test_file)
        with pytest.raises(ValueError):
            with BackupContext(test_file, console=console):
                test_file.write_bytes(b"modified content")
                raise ValueError("Test exception")

        # File should be restored, backup deleted
        assert test_file.read_bytes() == original_content
        assert not backup_path.exists()

    # Initialization and Configuration

    def test_custom_backup_suffix(self, tmp_path):
        """Test backup context with custom suffix."""
        console = Console()
        test_file = tmp_path / "test.mcap"
        test_file.write_bytes(b"content")

        with BackupContext(test_file, console=console, backup_suffix=".bak") as ctx:
            expected_backup = test_file.with_suffix(f"{test_file.suffix}.bak")
            assert ctx.backup_path == expected_backup
            assert ctx.backup_path.exists()

    def test_default_dummy_console(self, tmp_path):
        """Test that DummyConsole is used when no console provided."""
        test_file = tmp_path / "test.mcap"
        test_file.write_bytes(b"content")

        ctx = BackupContext(test_file)  # No console provided
        assert isinstance(ctx.console, DummyConsole)

        # Verify DummyConsole doesn't output anything
        with patch("builtins.print") as mock_print:
            ctx.console.print("test message")
            mock_print.assert_not_called()

    # Error Conditions

    def test_file_not_found_error(self, tmp_path):
        """Test context when source file doesn't exist."""
        console = Console()
        nonexistent_file = tmp_path / "nonexistent.mcap"

        with pytest.raises(FileNotFoundError, match="File not found"):
            with BackupContext(nonexistent_file, console=console):
                pass

    def test_backup_already_exists_error(self, tmp_path):
        """Test context when backup file already exists."""
        console = Console()
        test_file = tmp_path / "test.mcap"
        test_file.write_bytes(b"content")

        # Create backup file manually
        backup_path = BackupContext.find_backup_path(test_file)
        backup_path.write_bytes(b"existing backup")

        with pytest.raises(FileExistsError, match="Backup file already exists"):
            with BackupContext(test_file, console=console):
                pass

    @patch("owa.core.utils.backup.BackupContext.rollback_from_backup")
    def test_rollback_failure_during_exception(self, mock_rollback, tmp_path):
        """Test critical error when rollback fails during exception handling."""
        console = Console()
        test_file = tmp_path / "test.mcap"
        test_file.write_bytes(b"original content")

        # Make rollback fail
        mock_rollback.side_effect = OSError("Rollback failed")

        with pytest.raises(OSError, match="Rollback failed"):
            with BackupContext(test_file, console=console):
                test_file.write_bytes(b"modified content")
                raise ValueError("Original exception")

    @patch("owa.core.utils.backup.BackupContext.rollback_from_backup")
    def test_exception_chaining_preserves_original_context(self, mock_rollback, tmp_path):
        """Test that exception chaining preserves the original exception context."""
        console = Console()
        test_file = tmp_path / "test.mcap"
        test_file.write_bytes(b"original content")

        # Make rollback fail
        rollback_error = OSError("Rollback failed")
        mock_rollback.side_effect = rollback_error

        try:
            with BackupContext(test_file, console=console):
                test_file.write_bytes(b"modified content")
                raise ValueError("Original exception")
        except OSError as e:
            # Verify the rollback error is raised
            assert str(e) == "Rollback failed"
            # Verify the original exception is chained as the cause
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "Original exception"

    # Static Method Tests

    def test_find_backup_path_default_suffix(self, tmp_path):
        """Test find_backup_path with default suffix."""
        test_file = tmp_path / "test.mcap"
        backup_path = BackupContext.find_backup_path(test_file)
        expected = test_file.with_suffix(".mcap.backup")
        assert backup_path == expected

    def test_find_backup_path_custom_suffix(self, tmp_path):
        """Test find_backup_path with custom suffix."""
        test_file = tmp_path / "test.mcap"
        backup_path = BackupContext.find_backup_path(test_file, ".bak")
        expected = test_file.with_suffix(".mcap.bak")
        assert backup_path == expected

    def test_rollback_from_backup_success(self, tmp_path):
        """Test successful rollback using static method."""
        console = Console()
        test_file = tmp_path / "test.mcap"
        backup_file = tmp_path / "test.mcap.backup"

        test_file.write_bytes(b"modified content")
        backup_file.write_bytes(b"original content")

        BackupContext.rollback_from_backup(test_file, backup_file, console)

        assert test_file.read_bytes() == b"original content"
        assert backup_file.exists()  # Backup preserved by default

    def test_rollback_from_backup_with_delete(self, tmp_path):
        """Test rollback with backup deletion."""
        console = Console()
        test_file = tmp_path / "test.mcap"
        backup_file = tmp_path / "test.mcap.backup"

        test_file.write_bytes(b"modified content")
        backup_file.write_bytes(b"original content")

        BackupContext.rollback_from_backup(test_file, backup_file, console, delete_backup=True)

        assert test_file.read_bytes() == b"original content"
        assert not backup_file.exists()  # Backup deleted

    def test_rollback_from_backup_missing_backup(self, tmp_path):
        """Test rollback when backup file doesn't exist."""
        console = Console()
        test_file = tmp_path / "test.mcap"
        backup_file = tmp_path / "test.mcap.backup"

        test_file.write_bytes(b"content")

        with pytest.raises(FileNotFoundError, match="Backup file not found"):
            BackupContext.rollback_from_backup(test_file, backup_file, console)

    def test_cleanup_backup_success(self, tmp_path):
        """Test successful backup cleanup."""
        console = Console()
        backup_file = tmp_path / "test.mcap.backup"
        backup_file.write_bytes(b"backup content")

        BackupContext.cleanup_backup(backup_file, console)
        assert not backup_file.exists()

    def test_cleanup_backup_missing_file(self, tmp_path):
        """Test cleanup when backup file doesn't exist (should not raise)."""
        console = Console()
        backup_file = tmp_path / "nonexistent.backup"

        # Should not raise exception due to missing_ok=True
        BackupContext.cleanup_backup(backup_file, console)

    @patch("owa.core.utils.backup.Path.unlink")
    def test_cleanup_backup_permission_error(self, mock_unlink, tmp_path):
        """Test cleanup when deletion fails."""
        console = Console()
        backup_file = tmp_path / "test.mcap.backup"

        mock_unlink.side_effect = OSError("Permission denied")

        with pytest.raises(OSError, match="Permission denied"):
            BackupContext.cleanup_backup(backup_file, console)

    # Complex Scenarios

    def test_nested_backup_contexts(self, tmp_path):
        """Test nested backup contexts with different suffixes."""
        console = Console()
        test_file = tmp_path / "test.mcap"
        original_content = b"original"
        test_file.write_bytes(original_content)

        with BackupContext(test_file, console=console, backup_suffix=".outer") as outer_ctx:
            test_file.write_bytes(b"outer modification")

            with BackupContext(test_file, console=console, backup_suffix=".inner") as inner_ctx:
                test_file.write_bytes(b"inner modification")

                # Both backups should exist
                assert outer_ctx.backup_path.exists()
                assert inner_ctx.backup_path.exists()

                # Inner backup should contain outer modification
                assert inner_ctx.backup_path.read_bytes() == b"outer modification"

        # File should contain inner modification (no exceptions)
        assert test_file.read_bytes() == b"inner modification"

    def test_nested_contexts_with_exception_in_inner(self, tmp_path):
        """Test nested contexts where inner context raises exception."""
        console = Console()
        test_file = tmp_path / "test.mcap"
        original_content = b"original"
        test_file.write_bytes(original_content)

        with BackupContext(test_file, console=console, backup_suffix=".outer"):
            test_file.write_bytes(b"outer modification")

            with pytest.raises(ValueError):
                with BackupContext(test_file, console=console, backup_suffix=".inner"):
                    test_file.write_bytes(b"inner modification")
                    raise ValueError("Inner exception")

            # After inner exception, file should be restored to outer state
            assert test_file.read_bytes() == b"outer modification"

        # After outer context, file should remain with outer modification
        assert test_file.read_bytes() == b"outer modification"
