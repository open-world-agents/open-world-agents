"""
Tests for MCAP migration verification utilities.
"""

import tempfile
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


class TestFileStats:
    """Test FileStats functionality."""

    @patch("owa.cli.mcap.migrate.utils.OWAMcapReader")
    def test_get_file_stats(self, mock_reader_class):
        """Test getting file statistics from an MCAP file."""
        # Setup mock reader
        mock_reader = MagicMock()
        schema1 = MagicMock()
        schema1.name = "desktop/KeyboardEvent"
        schema2 = MagicMock()
        schema2.name = "desktop/MouseEvent"
        mock_reader.schemas = {
            1: schema1,
            2: schema2,
        }

        # Mock messages
        mock_messages = [
            MagicMock(topic="keyboard/events"),
            MagicMock(topic="mouse/events"),
            MagicMock(topic="keyboard/events"),
        ]
        mock_reader.iter_messages.return_value = mock_messages
        mock_reader_class.return_value.__enter__.return_value = mock_reader

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            file_path = Path(tmp_file.name)

        try:
            stats = get_file_stats(file_path)

            assert stats.message_count == 3
            assert stats.file_size == len(b"test content")
            assert stats.topics == {"keyboard/events", "mouse/events"}
            assert stats.schemas == {"desktop/KeyboardEvent", "desktop/MouseEvent"}

        finally:
            # Close the file handle before unlinking on Windows
            tmp_file.close()
            if file_path.exists():
                file_path.unlink()


class TestIndividualVerificationFunctions:
    """Test individual verification functions."""

    def test_verify_message_count_success(self):
        """Test successful message count verification."""
        console = MagicMock()
        migrated_stats = FileStats(message_count=100, file_size=1000, topics={"topic1"}, schemas={"schema1"})
        backup_stats = FileStats(message_count=100, file_size=1000, topics={"topic1"}, schemas={"schema1"})

        result = verify_message_count(migrated_stats, backup_stats, console)
        assert result is True
        console.print.assert_not_called()

    def test_verify_message_count_failure(self):
        """Test message count verification failure."""
        console = MagicMock()
        migrated_stats = FileStats(message_count=100, file_size=1000, topics={"topic1"}, schemas={"schema1"})
        backup_stats = FileStats(message_count=90, file_size=1000, topics={"topic1"}, schemas={"schema1"})

        result = verify_message_count(migrated_stats, backup_stats, console)
        assert result is False
        console.print.assert_called_with("[red]Message count mismatch: 100 vs 90[/red]")

    def test_verify_file_size_success(self):
        """Test successful file size verification."""
        console = MagicMock()
        migrated_stats = FileStats(message_count=100, file_size=1050, topics={"topic1"}, schemas={"schema1"})
        backup_stats = FileStats(message_count=100, file_size=1000, topics={"topic1"}, schemas={"schema1"})

        result = verify_file_size(migrated_stats, backup_stats, console, tolerance_percent=10.0)
        assert result is True
        console.print.assert_not_called()

    def test_verify_file_size_failure(self):
        """Test file size verification failure."""
        console = MagicMock()
        migrated_stats = FileStats(message_count=100, file_size=1200, topics={"topic1"}, schemas={"schema1"})
        backup_stats = FileStats(message_count=100, file_size=1000, topics={"topic1"}, schemas={"schema1"})

        result = verify_file_size(migrated_stats, backup_stats, console, tolerance_percent=10.0)
        assert result is False
        console.print.assert_called_with("[red]File size difference too large: 20.0% (limit: 10.0%)[/red]")

    def test_verify_topics_preserved_success(self):
        """Test successful topic preservation verification."""
        console = MagicMock()
        topics = {"topic1", "topic2"}
        migrated_stats = FileStats(message_count=100, file_size=1000, topics=topics, schemas={"schema1"})
        backup_stats = FileStats(message_count=100, file_size=1000, topics=topics, schemas={"schema1"})

        result = verify_topics_preserved(migrated_stats, backup_stats, console)
        assert result is True
        console.print.assert_not_called()

    def test_verify_topics_preserved_failure(self):
        """Test topic preservation verification failure."""
        console = MagicMock()
        migrated_stats = FileStats(message_count=100, file_size=1000, topics={"topic1", "topic2"}, schemas={"schema1"})
        backup_stats = FileStats(message_count=100, file_size=1000, topics={"topic1"}, schemas={"schema1"})

        result = verify_topics_preserved(migrated_stats, backup_stats, console)
        assert result is False
        # Check that the error message contains the expected content (set order may vary)
        console.print.assert_called_once()
        call_args = console.print.call_args[0][0]
        assert "[red]Topic mismatch:" in call_args
        assert "vs {'topic1'}[/red]" in call_args


class TestVerifyMigrationIntegrity:
    """Test migration integrity verification."""

    def test_verify_migration_integrity_missing_files(self):
        """Test verification when files are missing."""
        console = MagicMock()

        # Test missing migrated file
        migrated_path = Path("/nonexistent/migrated.mcap")
        backup_path = Path("/nonexistent/backup.mcap")
        result = verify_migration_integrity(migrated_path, backup_path, console)
        assert result is False
        # Use str() to handle path separator differences between platforms
        console.print.assert_called_with(f"[red]Migrated file not found: {migrated_path}[/red]")

    @patch("owa.cli.mcap.migrate.utils.get_file_stats")
    def test_verify_migration_integrity_message_count_mismatch(self, mock_get_stats):
        """Test verification when message counts don't match."""
        console = MagicMock()

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as migrated_file:
            migrated_path = Path(migrated_file.name)
        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as backup_file:
            backup_path = Path(backup_file.name)

        try:
            # Mock different message counts
            mock_get_stats.side_effect = [
                FileStats(message_count=100, file_size=1000, topics={"topic1"}, schemas={"schema1"}),
                FileStats(message_count=90, file_size=1000, topics={"topic1"}, schemas={"schema1"}),
            ]

            result = verify_migration_integrity(migrated_path, backup_path, console)
            assert result is False
            console.print.assert_called_with("[red]Message count mismatch: 100 vs 90[/red]")

        finally:
            migrated_path.unlink()
            backup_path.unlink()

    @patch("owa.cli.mcap.migrate.utils.get_file_stats")
    def test_verify_migration_integrity_success(self, mock_get_stats):
        """Test successful verification."""
        console = MagicMock()

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as migrated_file:
            migrated_path = Path(migrated_file.name)
        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as backup_file:
            backup_path = Path(backup_file.name)

        try:
            # Mock matching statistics
            mock_get_stats.side_effect = [
                FileStats(message_count=100, file_size=1050, topics={"topic1", "topic2"}, schemas={"schema1"}),
                FileStats(message_count=100, file_size=1000, topics={"topic1", "topic2"}, schemas={"schema1"}),
            ]

            result = verify_migration_integrity(migrated_path, backup_path, console)
            assert result is True
            console.print.assert_called_with("[green]✓ Migration integrity verified: 100 messages, 2 topics[/green]")

        finally:
            migrated_path.unlink()
            backup_path.unlink()
