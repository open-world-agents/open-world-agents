"""
MCAP migration rollback functionality.

This module provides functionality to rollback MCAP files from their backup files,
typically used when a migration fails or needs to be undone.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from owa.cli.mcap.backup_utils import generate_backup_path, rollback_from_backup
from owa.cli.mcap.migrate.file_utils import (
    add_rollback_row,
    create_file_info_table,
    detect_mcap_version,
    format_datetime,
    format_file_size,
)


@dataclass
class BackupInfo:
    """Information about a backup file and its corresponding original file."""

    original_path: Path
    backup_path: Path
    backup_exists: bool
    original_exists: bool
    backup_size: Optional[int] = None
    backup_modified: Optional[datetime] = None
    original_size: Optional[int] = None
    original_modified: Optional[datetime] = None
    original_version: Optional[str] = None
    backup_version: Optional[str] = None


def find_backup_files(file_paths: List[Path], console: Console) -> List[BackupInfo]:
    """
    Find backup files corresponding to the given MCAP files.

    Args:
        file_paths: List of original MCAP file paths
        console: Rich console for output

    Returns:
        List of BackupInfo objects with backup information
    """
    backup_infos = []

    for file_path in file_paths:
        if not file_path.suffix == ".mcap":
            console.print(f"[yellow]Skipping non-MCAP file: {file_path}[/yellow]")
            continue

        backup_path = generate_backup_path(file_path)
        backup_exists = backup_path.exists()
        original_exists = file_path.exists()

        backup_info = BackupInfo(
            original_path=file_path,
            backup_path=backup_path,
            backup_exists=backup_exists,
            original_exists=original_exists,
        )

        # Get backup file information if it exists
        if backup_exists:
            try:
                stat = backup_path.stat()
                backup_info.backup_size = stat.st_size
                backup_info.backup_modified = datetime.fromtimestamp(stat.st_mtime)
                backup_info.backup_version = detect_mcap_version(backup_path)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not read backup info for {backup_path}: {e}[/yellow]")

        # Get original file information if it exists
        if original_exists:
            try:
                stat = file_path.stat()
                backup_info.original_size = stat.st_size
                backup_info.original_modified = datetime.fromtimestamp(stat.st_mtime)
                backup_info.original_version = detect_mcap_version(file_path)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not read original file info for {file_path}: {e}[/yellow]")

        backup_infos.append(backup_info)

    return backup_infos


def display_rollback_summary(backup_infos: List[BackupInfo], console: Console) -> List[BackupInfo]:
    """
    Display a summary table of files that can be rolled back.

    Args:
        backup_infos: List of backup information
        console: Rich console for output

    Returns:
        List of BackupInfo objects that can be rolled back
    """
    # Filter for files that can actually be rolled back
    rollbackable = [info for info in backup_infos if info.backup_exists]

    if not rollbackable:
        console.print("[yellow]No backup files found for rollback[/yellow]")
        return []

    # Create unified table
    table = create_file_info_table("rollback")

    for info in rollbackable:
        # Format dates and sizes using shared utilities
        backup_date = format_datetime(info.backup_modified)
        current_date = format_datetime(info.original_modified)
        current_size_str = format_file_size(info.original_size)
        backup_size_str = format_file_size(info.backup_size)

        # Determine status
        if not info.original_exists:
            status = "MISSING"
        elif info.original_version == info.backup_version:
            status = "SAME"
        else:
            status = "READY"

        # Add row using unified function
        add_rollback_row(table, info, current_date, current_size_str, backup_date, backup_size_str, status)

    console.print("\n[bold]Rollback Summary[/bold]")
    console.print(table)

    return rollbackable


def rollback(
    files: List[Path] = typer.Argument(..., help="MCAP files to rollback from backup"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed rollback information"),
) -> None:
    """
    Rollback MCAP files from their backup files.

    This command finds backup files (.mcap.backup) corresponding to the specified
    MCAP files and restores the original files from the backups. The backup files
    are removed after successful rollback.
    """
    console = Console()

    console.print("[bold blue]MCAP Rollback Tool[/bold blue]")
    console.print(f"Files to check: {len(files)}")

    # Find backup files
    backup_infos = find_backup_files(files, console)

    if not backup_infos:
        console.print("[yellow]No files to process[/yellow]")
        return

    # Display summary and get rollbackable files
    rollbackable = display_rollback_summary(backup_infos, console)

    if not rollbackable:
        return

    # Show warnings for problematic cases
    for info in rollbackable:
        if not info.original_exists:
            console.print(f"[yellow]Warning: Current file missing, will restore: {info.original_path}[/yellow]")
        elif info.original_version == info.backup_version:
            console.print(f"[yellow]Warning: Current and backup have same version: {info.original_path}[/yellow]")

    # Confirm rollback
    if not yes and not typer.confirm(f"\nProceed with rolling back {len(rollbackable)} files?", default=False):
        console.print("Rollback cancelled.")
        return

    # Perform rollbacks
    console.print(f"\n[bold]Starting rollback of {len(rollbackable)} files...[/bold]")

    successful_rollbacks = 0
    failed_rollbacks = 0

    for i, info in enumerate(rollbackable, 1):
        console.print(f"\n[bold cyan]File {i}/{len(rollbackable)}: {info.original_path}[/bold cyan]")

        if verbose:
            console.print(f"Restoring from: {info.backup_path}")

        try:
            success = rollback_from_backup(info.original_path, info.backup_path, console, delete_backup=True)

            if success:
                successful_rollbacks += 1
            else:
                failed_rollbacks += 1

        except Exception as e:
            console.print(f"[red]Unexpected error during rollback: {e}[/red]")
            failed_rollbacks += 1

    # Final summary
    console.print("\n[bold]Rollback Complete[/bold]")
    console.print(f"[green]Successful: {successful_rollbacks}[/green]")
    console.print(f"[red]Failed: {failed_rollbacks}[/red]")

    if failed_rollbacks > 0:
        raise typer.Exit(1)
