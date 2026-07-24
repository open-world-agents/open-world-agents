"""
Utilities for migration verification and integrity checks.

This module provides functionality to verify the integrity of migrated MCAP files
by comparing them with their backup counterparts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mcap_owa.highlevel import OWAMcapReader


@dataclass
class FileStats:
    """Statistics about an MCAP file."""

    message_count: int
    file_size: int
    topics: set[str]
    schemas: set[str]


@dataclass
class VerificationResult:
    """Result of migration verification."""

    success: bool
    error: Optional[str] = None
    message_count_match: Optional[bool] = None
    file_size_diff_percent: Optional[float] = None
    topics_match: Optional[bool] = None


def get_file_stats(file_path: Path) -> FileStats:
    """Get comprehensive statistics about an MCAP file."""
    with OWAMcapReader(file_path) as reader:
        schemas = set(reader.schemas)
        topics = set(reader.topics)
        message_count = reader.message_count

    file_size = file_path.stat().st_size
    return FileStats(message_count=message_count, file_size=file_size, topics=topics, schemas=schemas)


def verify_migration_integrity(
    migrated_file: Path,
    backup_file: Path,
    check_message_count: bool = True,
    check_file_size: bool = True,
    check_topics: bool = True,
    size_tolerance_percent: float = 10.0,
) -> VerificationResult:
    """
    Verify migration integrity by comparing migrated file with backup.

    Args:
        migrated_file: Path to the migrated MCAP file
        backup_file: Path to the backup (original) MCAP file
        check_message_count: Whether to verify message count preservation
        check_file_size: Whether to verify file size is within tolerance
        check_topics: Whether to verify topic preservation
        size_tolerance_percent: Allowed file size difference percentage

    Returns:
        VerificationResult with success status and detailed information
    """
    try:
        if not migrated_file.exists():
            return VerificationResult(success=False, error=f"Migrated file not found: {migrated_file}")

        if not backup_file.exists():
            return VerificationResult(success=False, error=f"Backup file not found: {backup_file}")

        migrated_stats = get_file_stats(migrated_file)
        backup_stats = get_file_stats(backup_file)

        result = VerificationResult(success=True)

        # Check message count
        if check_message_count:
            result.message_count_match = migrated_stats.message_count == backup_stats.message_count
            if not result.message_count_match:
                result.success = False
                result.error = (
                    f"Message count mismatch: {migrated_stats.message_count} vs {backup_stats.message_count}"
                )

        # Check file size
        if check_file_size and result.success:
            result.file_size_diff_percent = (
                abs(migrated_stats.file_size - backup_stats.file_size) / backup_stats.file_size * 100
            )
            if result.file_size_diff_percent > size_tolerance_percent:
                result.success = False
                result.error = f"File size difference too large: {result.file_size_diff_percent:.1f}% (limit: {size_tolerance_percent}%)"

        # Check topics
        if check_topics and result.success:
            result.topics_match = migrated_stats.topics == backup_stats.topics
            if not result.topics_match:
                result.success = False
                result.error = f"Topic mismatch: {migrated_stats.topics} vs {backup_stats.topics}"

        return result

    except Exception as e:
        return VerificationResult(success=False, error=f"Error during integrity verification: {e}")
