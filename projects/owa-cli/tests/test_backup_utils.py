"""
Comprehensive tests for backup and rollback utilities.

This module tests all aspects of the backup-rollback functionality including
success cases, failure cases, edge cases, and error conditions to ensure
complete security of file operations.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from owa.cli.mcap.backup_utils import (
    BackupContext,
    cleanup_backup_files,
    create_backup,
    generate_backup_path,
    rollback_from_backup,
)


class TestCreateBackup:
    """Test cases for the create_backup function."""

    def test_create_backup_success(self, temp_dir):
        """Test successful backup creation."""
        # Create a test file
        test_file = temp_dir / "test.mcap"
        test_content = b"test mcap content"
        test_file.write_bytes(test_content)

        backup_path = temp_dir / "test.mcap.backup"

        # Create backup
        create_backup(test_file, backup_path)

        # Verify backup was created
        assert backup_path.exists()
        assert backup_path.read_bytes() == test_content
        assert backup_path.stat().st_size == test_file.stat().st_size

    def test_create_backup_source_not_found(self, temp_dir):
        """Test backup creation when source file doesn't exist."""
        nonexistent_file = temp_dir / "nonexistent.mcap"
        backup_path = temp_dir / "backup.mcap"

        with pytest.raises(FileNotFoundError, match="Source file not found"):
            create_backup(nonexistent_file, backup_path)

    def test_create_backup_already_exists(self, temp_dir):
        """Test backup creation when backup file already exists."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        backup_path = temp_dir / "test.mcap.backup"
        backup_path.write_bytes(b"existing backup")

        with pytest.raises(FileExistsError, match="Backup file already exists"):
            create_backup(test_file, backup_path)

    def test_create_backup_creates_parent_directories(self, temp_dir):
        """Test that backup creation creates parent directories."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        backup_path = temp_dir / "nested" / "dir" / "test.mcap.backup"

        create_backup(test_file, backup_path)

        assert backup_path.exists()
        assert backup_path.parent.exists()

    @patch("owa.cli.mcap.backup_utils.shutil.copy2")
    def test_create_backup_copy_fails(self, mock_copy, temp_dir):
        """Test backup creation when file copy fails."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        backup_path = temp_dir / "test.mcap.backup"

        # Mock copy2 to not create the backup file
        mock_copy.return_value = None

        with pytest.raises(OSError, match="Backup creation failed"):
            create_backup(test_file, backup_path)

    def test_create_backup_size_mismatch(self, temp_dir):
        """Test backup creation when file sizes don't match."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"original content")

        backup_path = temp_dir / "test.mcap.backup"

        # Create a backup with different size
        with patch("owa.cli.mcap.backup_utils.shutil.copy2") as mock_copy:

            def create_wrong_size_backup(src, dst):
                Path(dst).write_bytes(b"wrong")  # Different size

            mock_copy.side_effect = create_wrong_size_backup

            with pytest.raises(OSError, match="Backup verification failed: size mismatch"):
                create_backup(test_file, backup_path)


class TestRollbackFromBackup:
    """Test cases for the rollback_from_backup function."""

    def test_rollback_success(self, temp_dir):
        """Test successful rollback from backup."""
        console = Console()

        # Create original and backup files
        original_file = temp_dir / "test.mcap"
        backup_file = temp_dir / "test.mcap.backup"

        original_content = b"modified content"
        backup_content = b"original content"

        original_file.write_bytes(original_content)
        backup_file.write_bytes(backup_content)

        # Perform rollback
        result = rollback_from_backup(original_file, backup_file, console)

        assert result is True
        assert original_file.read_bytes() == backup_content

    def test_rollback_with_delete_backup(self, temp_dir):
        """Test rollback with backup deletion."""
        console = Console()

        original_file = temp_dir / "test.mcap"
        backup_file = temp_dir / "test.mcap.backup"

        original_file.write_bytes(b"modified")
        backup_file.write_bytes(b"original")

        # Perform rollback with deletion
        result = rollback_from_backup(original_file, backup_file, console, delete_backup=True)

        assert result is True
        assert original_file.read_bytes() == b"original"
        assert not backup_file.exists()

    def test_rollback_backup_not_found(self, temp_dir):
        """Test rollback when backup file doesn't exist."""
        console = Console()

        original_file = temp_dir / "test.mcap"
        backup_file = temp_dir / "nonexistent.backup"

        original_file.write_bytes(b"content")

        result = rollback_from_backup(original_file, backup_file, console)

        assert result is False
        assert original_file.read_bytes() == b"content"  # Unchanged

    @patch("owa.cli.mcap.backup_utils.shutil.copy2")
    def test_rollback_copy_fails(self, mock_copy, temp_dir):
        """Test rollback when file copy fails."""
        console = Console()

        original_file = temp_dir / "test.mcap"
        backup_file = temp_dir / "test.mcap.backup"

        original_file.write_bytes(b"modified")
        backup_file.write_bytes(b"original")

        # Mock copy2 to raise an exception
        mock_copy.side_effect = OSError("Copy failed")

        result = rollback_from_backup(original_file, backup_file, console)

        assert result is False


class TestGenerateBackupPath:
    """Test cases for the generate_backup_path function."""

    def test_generate_backup_path_default_suffix(self):
        """Test backup path generation with default suffix."""
        file_path = Path("test.mcap")
        backup_path = generate_backup_path(file_path)

        assert backup_path == Path("test.mcap.backup")

    def test_generate_backup_path_custom_suffix(self):
        """Test backup path generation with custom suffix."""
        file_path = Path("test.mcap")
        backup_path = generate_backup_path(file_path, ".bak")

        assert backup_path == Path("test.mcap.bak")

    def test_generate_backup_path_complex_path(self):
        """Test backup path generation with complex file path."""
        file_path = Path("nested/dir/complex.file.mcap")
        backup_path = generate_backup_path(file_path)

        assert backup_path == Path("nested/dir/complex.file.mcap.backup")


class TestCleanupBackupFiles:
    """Test cases for the cleanup_backup_files function."""

    def test_cleanup_keep_backups(self, temp_dir):
        """Test cleanup when keeping backups."""
        console = Console()

        backup1 = temp_dir / "test1.mcap.backup"
        backup2 = temp_dir / "test2.mcap.backup"

        backup1.write_bytes(b"backup1")
        backup2.write_bytes(b"backup2")

        backup_paths = [backup1, backup2]

        cleanup_backup_files(backup_paths, console, keep_backups=True)

        # Files should still exist
        assert backup1.exists()
        assert backup2.exists()

    def test_cleanup_delete_backups(self, temp_dir):
        """Test cleanup when deleting backups."""
        console = Console()

        backup1 = temp_dir / "test1.mcap.backup"
        backup2 = temp_dir / "test2.mcap.backup"

        backup1.write_bytes(b"backup1")
        backup2.write_bytes(b"backup2")

        backup_paths = [backup1, backup2]

        cleanup_backup_files(backup_paths, console, keep_backups=False)

        # Files should be deleted
        assert not backup1.exists()
        assert not backup2.exists()

    def test_cleanup_empty_list(self, temp_dir):
        """Test cleanup with empty backup list."""
        console = Console()

        # Should not raise any errors
        cleanup_backup_files([], console, keep_backups=False)

    def test_cleanup_nonexistent_files(self, temp_dir):
        """Test cleanup with nonexistent backup files."""
        console = Console()

        nonexistent1 = temp_dir / "nonexistent1.backup"
        nonexistent2 = temp_dir / "nonexistent2.backup"

        backup_paths = [nonexistent1, nonexistent2]

        # Should not raise errors
        cleanup_backup_files(backup_paths, console, keep_backups=False)

    def test_cleanup_deletion_error(self, temp_dir):
        """Test cleanup when file deletion fails."""
        console = Console()

        backup_file = temp_dir / "test.mcap.backup"
        backup_file.write_bytes(b"backup")

        # Make file read-only to cause deletion error on some systems
        backup_file.chmod(0o444)

        try:
            with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
                cleanup_backup_files([backup_file], console, keep_backups=False)
            # Should not raise exception, just print warning
        finally:
            # Restore permissions for cleanup
            backup_file.chmod(0o644)

    def test_backup_context_class_method_cleanup(self, temp_dir):
        """Test BackupContext.cleanup_backup_files class method."""
        console = Console()

        backup1 = temp_dir / "test1.mcap.backup"
        backup2 = temp_dir / "test2.mcap.backup"

        backup1.write_bytes(b"backup1")
        backup2.write_bytes(b"backup2")

        backup_paths = [backup1, backup2]

        # Test keeping backups
        BackupContext.cleanup_backup_files(backup_paths, console, keep_backups=True)
        assert backup1.exists()
        assert backup2.exists()

        # Test deleting backups
        BackupContext.cleanup_backup_files(backup_paths, console, keep_backups=False)
        assert not backup1.exists()
        assert not backup2.exists()


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


class TestIntegrationScenarios:
    """Integration tests for complex backup-rollback scenarios."""

    def test_multiple_backup_operations(self, temp_dir):
        """Test multiple backup operations in sequence."""
        console = Console()

        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"version 1")

        # First backup and modification
        backup1 = generate_backup_path(test_file, ".backup1")
        create_backup(test_file, backup1)
        test_file.write_bytes(b"version 2")

        # Second backup and modification
        backup2 = generate_backup_path(test_file, ".backup2")
        create_backup(test_file, backup2)
        test_file.write_bytes(b"version 3")

        # Rollback to version 2
        rollback_from_backup(test_file, backup2, console)
        assert test_file.read_bytes() == b"version 2"

        # Rollback to version 1
        rollback_from_backup(test_file, backup1, console)
        assert test_file.read_bytes() == b"version 1"

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

    def test_large_file_backup_rollback(self, temp_dir):
        """Test backup and rollback with larger files."""
        console = Console()

        test_file = temp_dir / "large_test.mcap"

        # Create a larger file (1MB)
        large_content = b"x" * (1024 * 1024)
        test_file.write_bytes(large_content)

        backup_path = generate_backup_path(test_file)

        # Create backup
        create_backup(test_file, backup_path)

        # Modify file
        modified_content = b"y" * (1024 * 1024)
        test_file.write_bytes(modified_content)

        # Rollback
        result = rollback_from_backup(test_file, backup_path, console)

        assert result is True
        assert test_file.read_bytes() == large_content
        assert len(test_file.read_bytes()) == 1024 * 1024
