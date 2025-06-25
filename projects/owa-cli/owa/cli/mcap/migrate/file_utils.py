"""
Shared utilities for MCAP file operations.

This module provides common functionality used across migrate, rollback, and cleanup commands
to avoid code duplication and ensure consistency.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.table import Table

from mcap_owa.highlevel import OWAMcapReader


def format_file_size(size_bytes: Optional[int]) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes, or None if unknown

    Returns:
        Formatted size string (e.g., "1.5 KB", "2.3 MB", "Unknown")
    """
    if size_bytes is None:
        return "Unknown"

    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_datetime(dt: Optional[datetime]) -> str:
    """
    Format datetime in consistent format.

    Args:
        dt: Datetime object, or None if unknown

    Returns:
        Formatted datetime string (e.g., "2025-06-25 20:30:15", "Unknown")
    """
    if dt is None:
        return "Unknown"

    return dt.strftime("%Y-%m-%d %H:%M:%S")


def detect_mcap_version(file_path: Path) -> str:
    """
    Safely detect the version of an MCAP file.

    Args:
        file_path: Path to the MCAP file

    Returns:
        Version string or "unknown" if detection fails
    """
    try:
        with OWAMcapReader(file_path) as reader:
            file_version = reader.file_version
            return file_version if file_version and file_version != "unknown" else "unknown"
    except Exception:
        return "unknown"


def get_file_info(file_path: Path) -> tuple[Optional[int], Optional[datetime], str]:
    """
    Get comprehensive file information.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (size_bytes, modified_datetime, version)
        Any value can be None/"unknown" if detection fails
    """
    size = None
    modified = None
    version = "unknown"

    try:
        if file_path.exists():
            stat = file_path.stat()
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime)

            # Only try version detection for MCAP files
            if file_path.suffix == ".mcap":
                version = detect_mcap_version(file_path)
    except Exception:
        pass

    return size, modified, version


def create_file_info_table(table_type: str = "rollback") -> Table:
    """
    Create a standardized table for displaying file information.

    Args:
        table_type: Type of table ("rollback" or "cleanup")

    Returns:
        Configured Rich Table object
    """
    table = Table(show_header=True, header_style="bold magenta")

    # Both tables use consistent column order: Current info first, then Backup info, then Status
    if table_type == "rollback":
        # Rollback table: focus on current file that will be replaced
        table.add_column("Current File", style="cyan")
    else:  # cleanup
        # Cleanup table: focus on backup file that will be deleted
        table.add_column("Backup File", style="cyan")

    # Consistent column order for both tables
    table.add_column("Current Date", style="dim")
    table.add_column("Current Size", justify="right", style="dim")
    table.add_column("Current Version", style="green")
    table.add_column("Backup Date", style="yellow")
    table.add_column("Backup Size", justify="right")
    table.add_column("Backup Version", style="blue")
    table.add_column("Status", justify="center")

    return table


def add_rollback_row(
    table: Table, info, current_date: str, current_size_str: str, backup_date: str, backup_size_str: str, status: str
) -> None:
    """
    Add a row to a rollback table with standardized formatting.

    Args:
        table: Rich Table object
        info: BackupInfo object
        current_date: Formatted current date string
        current_size_str: Formatted current size string
        backup_date: Formatted backup date string
        backup_size_str: Formatted backup size string
        status: Status string
    """
    table.add_row(
        str(info.original_path.name),  # Current File
        current_date,  # Current Date
        current_size_str,  # Current Size
        info.original_version or "Unknown",  # Current Version
        backup_date,  # Backup Date
        backup_size_str,  # Backup Size
        info.backup_version or "Unknown",  # Backup Version
        status,  # Status
    )


def add_cleanup_row(
    table: Table,
    info,
    current_date: str,
    current_size_str: str,
    backup_date: str,
    backup_size_str: str,
    current_version: str,
    backup_version: str,
    status: str,
) -> None:
    """
    Add a row to a cleanup table with standardized formatting.

    Args:
        table: Rich Table object
        info: BackupFileInfo object
        current_date: Formatted current date string
        current_size_str: Formatted current size string
        backup_date: Formatted backup date string
        backup_size_str: Formatted backup size string
        current_version: Current file version
        backup_version: Backup file version
        status: Status string
    """
    table.add_row(
        str(info.backup_path.name),  # Backup File
        current_date,  # Current Date
        current_size_str,  # Current Size
        current_version,  # Current Version
        backup_date,  # Backup Date
        backup_size_str,  # Backup Size
        backup_version,  # Backup Version
        status,  # Status
    )
