"""Backup utilities for file operations."""

import shutil
from pathlib import Path
from typing import Optional

from rich.console import Console


class DummyConsole(Console):
    """A console that doesn't print anything."""

    def print(self, *args, **kwargs):
        pass


class BackupContext:
    """Simple context manager for file backup and rollback."""

    def __init__(
        self,
        file_path: Path,
        *,
        console: Optional[Console] = None,
        backup_suffix: str = ".backup",
        keep_backup: bool = False,
    ):
        self.file_path = file_path
        self.console = console or DummyConsole()
        self.backup_path = self.find_backup_path(file_path, backup_suffix)
        self.keep_backup = keep_backup

    @staticmethod
    def find_backup_path(file_path: Path, suffix: str = ".backup") -> Path:
        """Find backup file path for a given file path."""
        return file_path.with_suffix(f"{file_path.suffix}{suffix}")

    @staticmethod
    def cleanup_backup(backup_path: Path, console: Console) -> None:
        """Clean up a single backup file."""
        backup_path.unlink(missing_ok=True)
        console.print(f"[dim]Backup cleaned up: {backup_path}[/dim]")

    @staticmethod
    def rollback_from_backup(
        file_path: Path, backup_path: Path, console: Console, delete_backup: bool = False
    ) -> None:
        """Rollback file by restoring from backup."""
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        shutil.copy2(backup_path, file_path)
        console.print(f"[green]Restored from backup: {backup_path}[/green]")

        if delete_backup:
            backup_path.unlink()
            console.print(f"[dim]Backup file deleted: {backup_path}[/dim]")

    def __enter__(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        if self.backup_path.exists():
            raise FileExistsError(f"Backup file already exists: {self.backup_path}")

        shutil.copy2(self.file_path, self.backup_path)
        self.console.print(f"[dim]Created backup: {self.backup_path}[/dim]")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Rollback on error using static method
            try:
                self.rollback_from_backup(self.file_path, self.backup_path, self.console, delete_backup=True)
            except Exception as rollback_error:
                self.console.print(f"[red]CRITICAL: Rollback failed! {rollback_error}[/red]")
                self.console.print(f"[red]Manual recovery needed. Backup file: {self.backup_path}[/red]")
                # Chain the rollback error with the original exception to preserve context
                raise rollback_error from exc_val
        elif not self.keep_backup:
            # Clean up backup on success
            self.cleanup_backup(self.backup_path, self.console)
