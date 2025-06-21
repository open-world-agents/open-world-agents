"""
Tests for MCAP migration verification utilities.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from owa.cli.mcap.migrate.utils import (
    FileStats,
    get_file_stats,
    verify_file_size,
    verify_message_count,
    verify_migration_integrity,
    verify_topics_preserved,
)


@patch("owa.cli.mcap.migrate.utils.OWAMcapReader")
def test_get_file_stats(mock_reader_class, temp_dir):
    """Test getting file statistics from an MCAP file."""
    # Setup mock reader
    mock_reader = MagicMock()
    schema1 = MagicMock()
    schema1.name = "desktop/KeyboardEvent"
    schema2 = MagicMock()
    schema2.name = "desktop/MouseEvent"
    mock_reader.schemas = {1: schema1, 2: schema2}

    # Mock messages
    mock_messages = [
        MagicMock(topic="keyboard/events"),
        MagicMock(topic="mouse/events"),
        MagicMock(topic="keyboard/events"),
    ]
    mock_reader.iter_messages.return_value = mock_messages
    mock_reader_class.return_value.__enter__.return_value = mock_reader

    # Create a temporary file
    file_path = temp_dir / "test.mcap"
    file_path.write_bytes(b"test content")

    stats = get_file_stats(file_path)

    assert stats.message_count == 3
    assert stats.file_size == len(b"test content")
    assert stats.topics == {"keyboard/events", "mouse/events"}
    assert stats.schemas == {"desktop/KeyboardEvent", "desktop/MouseEvent"}


def test_verify_message_count_success():
    """Test successful message count verification."""
    console = MagicMock()
    migrated_stats = FileStats(message_count=100, file_size=1000, topics={"topic1"}, schemas={"schema1"})
    backup_stats = FileStats(message_count=100, file_size=1000, topics={"topic1"}, schemas={"schema1"})

    result = verify_message_count(migrated_stats, backup_stats, console)
    assert result is True
    console.print.assert_not_called()


def test_verify_message_count_failure():
    """Test message count verification failure."""
    console = MagicMock()
    migrated_stats = FileStats(message_count=100, file_size=1000, topics={"topic1"}, schemas={"schema1"})
    backup_stats = FileStats(message_count=90, file_size=1000, topics={"topic1"}, schemas={"schema1"})

    result = verify_message_count(migrated_stats, backup_stats, console)
    assert result is False
    console.print.assert_called_with("[red]Message count mismatch: 100 vs 90[/red]")


def test_verify_file_size_success():
    """Test successful file size verification."""
    console = MagicMock()
    migrated_stats = FileStats(message_count=100, file_size=1050, topics={"topic1"}, schemas={"schema1"})
    backup_stats = FileStats(message_count=100, file_size=1000, topics={"topic1"}, schemas={"schema1"})

    result = verify_file_size(migrated_stats, backup_stats, console, tolerance_percent=10.0)
    assert result is True
    console.print.assert_not_called()


def test_verify_file_size_failure():
    """Test file size verification failure."""
    console = MagicMock()
    migrated_stats = FileStats(message_count=100, file_size=1200, topics={"topic1"}, schemas={"schema1"})
    backup_stats = FileStats(message_count=100, file_size=1000, topics={"topic1"}, schemas={"schema1"})

    result = verify_file_size(migrated_stats, backup_stats, console, tolerance_percent=10.0)
    assert result is False
    console.print.assert_called_with("[red]File size difference too large: 20.0% (limit: 10.0%)[/red]")


def test_verify_topics_preserved_success():
    """Test successful topic preservation verification."""
    console = MagicMock()
    topics = {"topic1", "topic2"}
    migrated_stats = FileStats(message_count=100, file_size=1000, topics=topics, schemas={"schema1"})
    backup_stats = FileStats(message_count=100, file_size=1000, topics=topics, schemas={"schema1"})

    result = verify_topics_preserved(migrated_stats, backup_stats, console)
    assert result is True
    console.print.assert_not_called()


def test_verify_topics_preserved_failure():
    """Test topic preservation verification failure."""
    console = MagicMock()
    migrated_stats = FileStats(message_count=100, file_size=1000, topics={"topic1", "topic2"}, schemas={"schema1"})
    backup_stats = FileStats(message_count=100, file_size=1000, topics={"topic1"}, schemas={"schema1"})

    result = verify_topics_preserved(migrated_stats, backup_stats, console)
    assert result is False
    # Check that the error message contains the expected content
    console.print.assert_called_once()
    call_args = console.print.call_args[0][0]
    assert "[red]Topic mismatch:" in call_args
    assert "vs {'topic1'}[/red]" in call_args


def test_verify_migration_integrity_missing_files():
    """Test verification when files are missing."""
    console = MagicMock()

    # Test missing migrated file
    migrated_path = Path("/nonexistent/migrated.mcap")
    backup_path = Path("/nonexistent/backup.mcap")
    result = verify_migration_integrity(migrated_path, backup_path, console)
    assert result is False
    console.print.assert_called_with(f"[red]Migrated file not found: {migrated_path}[/red]")


@patch("owa.cli.mcap.migrate.utils.get_file_stats")
def test_verify_migration_integrity_message_count_mismatch(mock_get_stats, temp_dir):
    """Test verification when message counts don't match."""
    console = MagicMock()

    # Create temporary files
    migrated_path = temp_dir / "migrated.mcap"
    backup_path = temp_dir / "backup.mcap"
    migrated_path.touch()
    backup_path.touch()

    # Mock different message counts
    mock_get_stats.side_effect = [
        FileStats(message_count=100, file_size=1000, topics={"topic1"}, schemas={"schema1"}),
        FileStats(message_count=90, file_size=1000, topics={"topic1"}, schemas={"schema1"}),
    ]

    result = verify_migration_integrity(migrated_path, backup_path, console)
    assert result is False
    console.print.assert_called_with("[red]Message count mismatch: 100 vs 90[/red]")


@patch("owa.cli.mcap.migrate.utils.get_file_stats")
def test_verify_migration_integrity_success(mock_get_stats, temp_dir):
    """Test successful verification."""
    console = MagicMock()

    # Create temporary files
    migrated_path = temp_dir / "migrated.mcap"
    backup_path = temp_dir / "backup.mcap"
    migrated_path.touch()
    backup_path.touch()

    # Mock matching statistics
    mock_get_stats.side_effect = [
        FileStats(message_count=100, file_size=1050, topics={"topic1", "topic2"}, schemas={"schema1"}),
        FileStats(message_count=100, file_size=1000, topics={"topic1", "topic2"}, schemas={"schema1"}),
    ]

    result = verify_migration_integrity(migrated_path, backup_path, console)
    assert result is True
    console.print.assert_called_with("[green]✓ Migration integrity verified: 100 messages, 2 topics[/green]")
