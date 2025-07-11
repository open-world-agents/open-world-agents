"""
MCAP backup cleanup functionality.

This module provides functionality to find and clean up backup files (.mcap.backup)
created during migration operations.
"""

import glob
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from owa.cli.mcap.backup_utils import generate_backup_path
from owa.cli.mcap.migrate.file_utils import (
    add_cleanup_row,
    create_file_info_table,
    detect_mcap_version,
    format_datetime,
    format_file_size,
)

from ...console import console


@dataclass
class BackupFileInfo:
    """Information about a backup file."""

    backup_path: Path
    original_path: Path
    backup_size: Optional[int] = None
    backup_modified: Optional[datetime] = None
    original_exists: bool = False
    original_size: Optional[int] = None
    original_modified: Optional[datetime] = None
    backup_version: Optional[str] = None
    original_version: Optional[str] = None


def find_backup_files_by_pattern(patterns: List[str], console: Console) -> List[BackupFileInfo]:
    """
    Find backup files using glob patterns.

    Args:
        patterns: List of glob patterns to search for backup files
        console: Rich console for output

    Returns:
        List of BackupFileInfo objects
    """
    backup_infos = []
    all_backup_paths = set()

    for pattern in patterns:
        try:
            # Expand the pattern to find backup files
            if pattern.endswith(".backup") or ".backup" in pattern:
                # Direct backup file pattern
                backup_paths = [Path(p) for p in glob.glob(pattern, recursive=True)]
            else:
                # MCAP file pattern - find corresponding backup files
                mcap_paths = [Path(p) for p in glob.glob(pattern, recursive=True)]
                backup_paths = []
                for mcap_path in mcap_paths:
                    if mcap_path.suffix == ".mcap":
                        backup_path = generate_backup_path(mcap_path)
                        if backup_path.exists():
                            backup_paths.append(backup_path)

            all_backup_paths.update(backup_paths)

        except Exception as e:
            console.print(f"[yellow]Warning: Error processing pattern '{pattern}': {e}[/yellow]")

    # Convert to BackupFileInfo objects
    for backup_path in all_backup_paths:
        if not backup_path.exists():
            continue

        # Determine original file path
        if backup_path.name.endswith(".mcap.backup"):
            original_path = backup_path.with_suffix("")  # Remove .backup
        else:
            console.print(f"[yellow]Warning: Unexpected backup file name format: {backup_path}[/yellow]")
            continue

        backup_info = BackupFileInfo(backup_path=backup_path, original_path=original_path)

        # Get backup file information
        try:
            stat = backup_path.stat()
            backup_info.backup_size = stat.st_size
            backup_info.backup_modified = datetime.fromtimestamp(stat.st_mtime)
            backup_info.backup_version = detect_mcap_version(backup_path)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read backup file info for {backup_path}: {e}[/yellow]")

        # Check if original file exists and get its info
        if original_path.exists():
            backup_info.original_exists = True
            try:
                stat = original_path.stat()
                backup_info.original_size = stat.st_size
                backup_info.original_modified = datetime.fromtimestamp(stat.st_mtime)
                backup_info.original_version = detect_mcap_version(original_path)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not read original file info for {original_path}: {e}[/yellow]")

        backup_infos.append(backup_info)

    return backup_infos


def find_all_backup_files(directories: List[Path], console: Console) -> List[BackupFileInfo]:
    """
    Find all backup files in the specified directories.

    Args:
        directories: List of directories to search
        console: Rich console for output

    Returns:
        List of BackupFileInfo objects
    """
    patterns = []

    for directory in directories:
        if directory.is_file():
            # Single file - check if it's a backup or find its backup
            if directory.name.endswith(".mcap.backup"):
                patterns.append(str(directory))
            elif directory.suffix == ".mcap":
                patterns.append(str(directory))
            else:
                console.print(f"[yellow]Skipping non-MCAP file: {directory}[/yellow]")
        elif directory.is_dir():
            # Directory - find all backup files recursively
            patterns.append(str(directory / "**" / "*.mcap.backup"))
        else:
            console.print(f"[yellow]Path not found: {directory}[/yellow]")

    return find_backup_files_by_pattern(patterns, console)


def display_cleanup_summary(backup_infos: List[BackupFileInfo], console: Console) -> None:
    """
    Display a summary table of backup files to be cleaned up.

    Args:
        backup_infos: List of backup file information
        console: Rich console for output
    """
    if not backup_infos:
        console.print("[yellow]No backup files found[/yellow]")
        return

    # Create unified table
    table = create_file_info_table("cleanup")
    total_size = 0

    for info in backup_infos:
        # Format dates and sizes using shared utilities
        backup_date = format_datetime(info.backup_modified)
        current_date = format_datetime(info.original_modified)
        backup_size_str = format_file_size(info.backup_size)
        current_size_str = format_file_size(info.original_size)
        current_version = info.original_version or "Unknown"
        backup_version = info.backup_version or "Unknown"

        # Track total size for summary
        if info.backup_size is not None:
            total_size += info.backup_size

        # Determine status
        if info.original_exists:
            if info.original_size == info.backup_size:
                status = "SAME SIZE"
            else:
                status = "DIFFERENT"
        else:
            status = "ORPHANED"

        # Add row using unified function
        add_cleanup_row(
            table,
            info,
            current_date,
            current_size_str,
            backup_date,
            backup_size_str,
            current_version,
            backup_version,
            status,
        )

    console.print("\n[bold]Cleanup Summary[/bold]")
    console.print(table)

    # Show total size
    if total_size > 0:
        total_str = format_file_size(total_size)
        console.print(f"\n[bold]Total backup size: {total_str}[/bold]")


def cleanup(
    patterns: Optional[List[str]] = typer.Argument(
        None,
        help="Patterns to search for backup files (default: all .mcap.backup files in current directory and subdirectories)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed cleanup information"),
) -> None:
    """
    Clean up MCAP backup files.

    This command finds backup files (.mcap.backup) using the specified patterns
    and removes them after confirmation. Use --dry-run to preview what would be deleted.

    Examples:
        owl mcap migrate cleanup                    # Clean all backup files in current directory tree
        owl mcap migrate cleanup "*.mcap.backup"   # Clean backup files in current directory only
        owl mcap migrate cleanup /path/to/backups  # Clean backup files in specific directory
        owl mcap migrate cleanup file.mcap         # Clean backup for specific MCAP file
    """
    # Set default patterns if none provided
    if patterns is None:
        patterns = ["**/*.mcap.backup"]

    console.print("[bold blue]MCAP Backup Cleanup Tool[/bold blue]")

    if dry_run:
        console.print("[yellow]DRY RUN MODE - No files will be deleted[/yellow]")

    # Find backup files
    backup_infos = find_backup_files_by_pattern(patterns, console)

    if not backup_infos:
        console.print("[yellow]No backup files found matching the specified patterns[/yellow]")
        return

    # Display summary
    display_cleanup_summary(backup_infos, console)

    if dry_run:
        console.print(f"\n[yellow]Would delete {len(backup_infos)} backup files[/yellow]")
        return

    # Confirm cleanup
    if not yes and not typer.confirm(f"\nProceed with deleting {len(backup_infos)} backup files?", default=False):
        console.print("Cleanup cancelled.")
        return

    # Perform cleanup
    console.print(f"\n[bold]Starting cleanup of {len(backup_infos)} backup files...[/bold]")

    successful_deletions = 0
    failed_deletions = 0

    for i, info in enumerate(backup_infos, 1):
        if verbose:
            console.print(f"\n[bold cyan]File {i}/{len(backup_infos)}: {info.backup_path}[/bold cyan]")

        try:
            info.backup_path.unlink()
            successful_deletions += 1

            if verbose:
                console.print(f"[green]Deleted: {info.backup_path}[/green]")

        except Exception as e:
            console.print(f"[red]Failed to delete {info.backup_path}: {e}[/red]")
            failed_deletions += 1

    # Final summary
    console.print("\n[bold]Cleanup Complete[/bold]")
    console.print(f"[green]Deleted: {successful_deletions}[/green]")
    console.print(f"[red]Failed: {failed_deletions}[/red]")

    if failed_deletions > 0:
        raise typer.Exit(1)
