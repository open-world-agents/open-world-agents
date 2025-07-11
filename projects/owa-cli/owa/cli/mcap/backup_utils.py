"""
Unified backup and rollback utilities for MCAP file operations.

This module provides secure backup creation, rollback, and cleanup functionality
that can be shared across different MCAP CLI commands to ensure data safety
during file modifications.
"""

import shutil
from pathlib import Path
from typing import List

from rich.console import Console


def create_backup(file_path: Path, backup_path: Path) -> None:
    """
    Create a backup of the file with high reliability and verification.

    Args:
        file_path: Path to the source file to backup
        backup_path: Path where the backup should be created

    Raises:
        FileNotFoundError: If the source file doesn't exist
        FileExistsError: If the backup file already exists
        OSError: If backup creation or verification fails
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    if backup_path.exists():
        raise FileExistsError(f"Backup file already exists: {backup_path}")

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, backup_path)

    if not backup_path.exists():
        raise OSError(f"Backup creation failed: {backup_path}")

    if backup_path.stat().st_size != file_path.stat().st_size:
        raise OSError(f"Backup verification failed: size mismatch for {backup_path}")


def rollback_from_backup(file_path: Path, backup_path: Path, console: Console, delete_backup: bool = False) -> bool:
    """
    Rollback file by restoring from backup.

    Args:
        file_path: Path to the file to restore
        backup_path: Path to the backup file
        console: Rich console for output
        delete_backup: Whether to delete the backup file after successful rollback

    Returns:
        True if rollback was successful, False otherwise
    """
    console.print("[yellow]Rolling back changes...[/yellow]")

    if backup_path.exists():
        try:
            shutil.copy2(backup_path, file_path)
            console.print(f"[green]Restored from backup: {backup_path}[/green]")

            if delete_backup:
                backup_path.unlink()
                console.print(f"[dim]Backup file deleted: {backup_path}[/dim]")

            return True
        except Exception as e:
            console.print(f"[red]Rollback failed: {e}[/red]")
            return False
    else:
        console.print("[red]No valid backup found for rollback[/red]")
        return False


def generate_backup_path(file_path: Path, suffix: str = ".backup") -> Path:
    """
    Generate a standardized backup path for a given file.

    Args:
        file_path: Path to the original file
        suffix: Suffix to append to the file extension (default: ".backup")

    Returns:
        Path object for the backup file
    """
    return file_path.with_suffix(f"{file_path.suffix}{suffix}")


class BackupContext:
    """
    Context manager for safe file operations with automatic backup, rollback, and cleanup.

    This class provides a context manager that automatically creates backups
    before file operations, handles rollback on exceptions, and manages backup cleanup.
    """

    def __init__(self, file_path: Path, console: Console, backup_suffix: str = ".backup", keep_backup: bool = True):
        """
        Initialize the backup context.

        Args:
            file_path: Path to the file to protect with backup
            console: Rich console for output
            backup_suffix: Suffix for backup file (default: ".backup")
            keep_backup: Whether to keep the backup file after successful operation
        """
        self.file_path = file_path
        self.console = console
        self.backup_path = generate_backup_path(file_path, backup_suffix)
        self.backup_created = False
        self.keep_backup = keep_backup

    def __enter__(self) -> "BackupContext":
        """Create backup when entering context."""
        try:
            create_backup(self.file_path, self.backup_path)
            self.backup_created = True
        except Exception as e:
            self.console.print(f"[red]Failed to create backup: {e}[/red]")
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Handle rollback on exceptions and cleanup."""
        if exc_type is not None and self.backup_created:
            # On exception: always rollback and delete backup
            rollback_from_backup(self.file_path, self.backup_path, self.console, delete_backup=True)
        elif self.backup_created and not self.keep_backup:
            # On success: cleanup backup if not keeping it
            try:
                if self.backup_path.exists():
                    self.backup_path.unlink()
                    self.console.print(f"[dim]Backup cleaned up: {self.backup_path}[/dim]")
            except Exception as e:
                self.console.print(f"[red]Warning: Could not delete backup {self.backup_path}: {e}[/red]")
        elif self.backup_created and self.keep_backup:
            # On success: inform user about backup location
            self.console.print(f"[blue]Backup saved: {self.backup_path}[/blue]")

    def rollback(self, delete_backup: bool = False) -> bool:
        """
        Manually trigger rollback.

        Args:
            delete_backup: Whether to delete backup after rollback

        Returns:
            True if rollback was successful, False otherwise
        """
        if self.backup_created:
            return rollback_from_backup(self.file_path, self.backup_path, self.console, delete_backup)
        return False

    def cleanup_backup(self) -> None:
        """Remove the backup file if it exists."""
        if self.backup_created and self.backup_path.exists():
            try:
                self.backup_path.unlink()
            except Exception as e:
                self.console.print(f"[red]Warning: Could not delete backup {self.backup_path}: {e}[/red]")

    @classmethod
    def cleanup_backup_files(cls, backup_paths: List[Path], console: Console, keep_backups: bool = True) -> None:
        """
        Handle cleanup of backup files based on user preference.

        Args:
            backup_paths: List of backup file paths to potentially clean up
            console: Rich console for output
            keep_backups: Whether to keep backup files (if False, they will be deleted)
        """
        if not backup_paths:
            return

        if not keep_backups:
            console.print(f"\n[yellow]Cleaning up {len(backup_paths)} backup files...[/yellow]")
            for backup_path in backup_paths:
                try:
                    if backup_path.exists():
                        backup_path.unlink()
                except Exception as e:
                    console.print(f"[red]Warning: Could not delete backup {backup_path}: {e}[/red]")
        else:
            console.print("\n[blue]Backup files saved with .mcap.backup extension[/blue]")
