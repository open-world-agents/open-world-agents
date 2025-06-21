from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from mcap_owa.highlevel import OWAMcapReader


@dataclass
class FileStats:
    """Statistics about an MCAP file."""

    message_count: int
    file_size: int
    topics: set[str]
    schemas: set[str]


def get_file_stats(file_path: Path) -> FileStats:
    """Get comprehensive statistics about an MCAP file."""
    message_count = 0
    topics = set()
    schemas = set()

    with OWAMcapReader(file_path) as reader:
        # Get schemas
        for schema in reader.schemas.values():
            schemas.add(schema.name)

        # Count messages and collect topics
        for msg in reader.iter_messages():
            message_count += 1
            topics.add(msg.topic)

    file_size = file_path.stat().st_size

    return FileStats(message_count=message_count, file_size=file_size, topics=topics, schemas=schemas)


def verify_message_count(migrated_stats: FileStats, backup_stats: FileStats, console: Console) -> bool:
    """
    Verify that message count is preserved during migration.

    Args:
        migrated_stats: Statistics from the migrated file
        backup_stats: Statistics from the backup file
        console: Rich console for output

    Returns:
        True if message counts match, False otherwise
    """
    if migrated_stats.message_count != backup_stats.message_count:
        console.print(
            f"[red]Message count mismatch: {migrated_stats.message_count} vs {backup_stats.message_count}[/red]"
        )
        return False
    return True


def verify_file_size(
    migrated_stats: FileStats, backup_stats: FileStats, console: Console, tolerance_percent: float = 10.0
) -> bool:
    """
    Verify that file size difference is within acceptable tolerance.

    Args:
        migrated_stats: Statistics from the migrated file
        backup_stats: Statistics from the backup file
        console: Rich console for output
        tolerance_percent: Allowed file size difference percentage

    Returns:
        True if file size difference is within tolerance, False otherwise
    """
    size_diff_percent = abs(migrated_stats.file_size - backup_stats.file_size) / backup_stats.file_size * 100
    if size_diff_percent > tolerance_percent:
        console.print(
            f"[red]File size difference too large: {size_diff_percent:.1f}% (limit: {tolerance_percent}%)[/red]"
        )
        return False
    return True


def verify_topics_preserved(migrated_stats: FileStats, backup_stats: FileStats, console: Console) -> bool:
    """
    Verify that all topics are preserved during migration.

    Args:
        migrated_stats: Statistics from the migrated file
        backup_stats: Statistics from the backup file
        console: Rich console for output

    Returns:
        True if topics match, False otherwise
    """
    if migrated_stats.topics != backup_stats.topics:
        console.print(f"[red]Topic mismatch: {migrated_stats.topics} vs {backup_stats.topics}[/red]")
        return False
    return True


def verify_migration_integrity(
    migrated_file: Path,
    backup_file: Path,
    console: Console,
    check_message_count: bool = True,
    check_file_size: bool = True,
    check_topics: bool = True,
    size_tolerance_percent: float = 10.0,
) -> bool:
    """
    Verify migration integrity by comparing migrated file with backup.

    Args:
        migrated_file: Path to the migrated MCAP file
        backup_file: Path to the backup (original) MCAP file
        console: Rich console for output
        check_message_count: Whether to verify message count preservation
        check_file_size: Whether to verify file size is within tolerance
        check_topics: Whether to verify topic preservation
        size_tolerance_percent: Allowed file size difference percentage (default: 10%)

    Returns:
        True if migration integrity checks pass, False otherwise
    """
    try:
        if not migrated_file.exists():
            console.print(f"[red]Migrated file not found: {migrated_file}[/red]")
            return False

        if not backup_file.exists():
            console.print(f"[red]Backup file not found: {backup_file}[/red]")
            return False

        # Get statistics for both files
        migrated_stats = get_file_stats(migrated_file)
        backup_stats = get_file_stats(backup_file)

        # Perform enabled verifications
        if check_message_count and not verify_message_count(migrated_stats, backup_stats, console):
            return False

        if check_file_size and not verify_file_size(migrated_stats, backup_stats, console, size_tolerance_percent):
            return False

        if check_topics and not verify_topics_preserved(migrated_stats, backup_stats, console):
            return False

        console.print(
            f"[green]✓ Migration integrity verified: {migrated_stats.message_count} messages, {len(migrated_stats.topics)} topics[/green]"
        )
        return True

    except Exception as e:
        console.print(f"[red]Error during integrity verification: {e}[/red]")
        return False
