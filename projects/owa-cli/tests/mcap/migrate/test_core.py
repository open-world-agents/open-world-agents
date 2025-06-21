"""
Tests for migrate command functionality.

This module tests the core migration functionality of the migrate command.
"""

from unittest.mock import MagicMock, patch

import pytest

from owa.cli.mcap import app as mcap_app
from owa.cli.mcap.migrate.migrate import MigrationOrchestrator


class TestMigrateIntegration:
    """Integration tests for migrate command core functionality."""

    def test_migrate_successful_operation(
        self,
        temp_dir,
        cli_runner,
        mock_migration_orchestrator,
        mock_migration_result,
        mock_file_info,
        mock_file_detection,
    ):
        """Test successful migration operation."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"original mcap content")

        # Setup mocks
        mock_migration_orchestrator.migrate_file.return_value = [mock_migration_result]
        mock_info = mock_file_info(test_file)
        mock_file_detection.return_value = [mock_info]

        # Run migrate command
        result = cli_runner.invoke(mcap_app, ["migrate", str(test_file), "--yes"])

        assert result.exit_code == 0
        mock_migration_orchestrator.migrate_file.assert_called_once()

    def test_migrate_failed_operation(
        self, temp_dir, cli_runner, mock_migration_orchestrator, mock_file_info, mock_file_detection
    ):
        """Test failed migration operation."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"original mcap content")

        # Mock failed migration
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "Migration failed"
        mock_migration_orchestrator.migrate_file.return_value = [mock_result]

        # Mock file detection
        mock_info = mock_file_info(test_file)
        mock_file_detection.return_value = [mock_info]

        # Run migrate command
        result = cli_runner.invoke(mcap_app, ["migrate", str(test_file), "--yes"])

        assert result.exit_code == 1
        mock_migration_orchestrator.migrate_file.assert_called_once()

    def test_migrate_multiple_files(self, temp_dir, cli_runner):
        """Test migration with multiple files."""
        test_file1 = temp_dir / "test1.mcap"
        test_file2 = temp_dir / "test2.mcap"

        test_file1.write_bytes(b"content 1")
        test_file2.write_bytes(b"content 2")

        with patch("owa.cli.mcap.migrate.migrate.MigrationOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock successful migrations
            mock_result = MagicMock()
            mock_result.success = True
            mock_orchestrator.migrate_file.return_value = [mock_result]

            # Mock file detection
            mock_info1 = MagicMock()
            mock_info1.file_path = test_file1
            mock_info1.detected_version = "0.3.0"
            mock_info1.needs_migration = True
            mock_info1.target_version = "0.4.0"

            mock_info2 = MagicMock()
            mock_info2.file_path = test_file2
            mock_info2.detected_version = "0.3.0"
            mock_info2.needs_migration = True
            mock_info2.target_version = "0.4.0"

            with patch("owa.cli.mcap.migrate.migrate.detect_files_needing_migration") as mock_detect:
                mock_detect.return_value = [mock_info1, mock_info2]

                # Run migrate on multiple files
                result = cli_runner.invoke(mcap_app, ["migrate", str(test_file1), str(test_file2), "--yes"])

                assert result.exit_code == 0
                # Should be called twice, once for each file
                assert mock_orchestrator.migrate_file.call_count == 2

    def test_migrate_dry_run(self, temp_dir, cli_runner):
        """Test dry run mode doesn't perform actual migration."""
        test_file = temp_dir / "test.mcap"
        original_content = b"original content"
        test_file.write_bytes(original_content)

        with patch("owa.cli.mcap.migrate.migrate.MigrationOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock file detection
            mock_info = MagicMock()
            mock_info.file_path = test_file
            mock_info.detected_version = "0.3.0"
            mock_info.needs_migration = True
            mock_info.target_version = "0.4.0"

            with patch("owa.cli.mcap.migrate.migrate.detect_files_needing_migration") as mock_detect:
                mock_detect.return_value = [mock_info]

                # Run migrate in dry-run mode
                result = cli_runner.invoke(mcap_app, ["migrate", str(test_file), "--dry-run"])

                assert result.exit_code == 0
                # Original file should be unchanged
                assert test_file.read_bytes() == original_content
                # Migration should not be attempted
                mock_orchestrator.migrate_file.assert_not_called()

    def test_migrate_no_files_need_migration(self, temp_dir, cli_runner):
        """Test behavior when no files need migration."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        with patch("owa.cli.mcap.migrate.migrate.detect_files_needing_migration") as mock_detect:
            mock_detect.return_value = []  # No files need migration

            # Run migrate command
            result = cli_runner.invoke(mcap_app, ["migrate", str(test_file), "--yes"])

            assert result.exit_code == 0
            assert "No files found matching the pattern" in result.output


class TestMigrationOrchestrator:
    """Test MigrationOrchestrator core functionality."""

    def test_orchestrator_successful_migration(self, temp_dir):
        """Test successful migration through orchestrator."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        orchestrator = MigrationOrchestrator()

        with patch("owa.cli.mcap.migrate.migrate.OWAMcapReader") as mock_reader:
            # Mock version detection
            mock_reader.return_value.__enter__.return_value.file_version = "0.3.0"

            # Mock migration path with one migrator
            mock_migrator = MagicMock()
            mock_migrator.from_version = "0.3.0"
            mock_migrator.to_version = "0.4.0"
            mock_migrator.migrate.return_value = MagicMock(success=True, changes_made=1)
            mock_migrator.verify_migration.return_value = True

            with patch.object(orchestrator, "get_migration_path") as mock_get_path:
                mock_get_path.return_value = [mock_migrator]

                # Call migrate_file
                from rich.console import Console

                console = Console()
                results = orchestrator.migrate_file(test_file, "0.4.0", console)

                # Verify migration was successful
                assert len(results) == 1
                assert results[0].success

    def test_orchestrator_failed_migration(self, temp_dir):
        """Test failed migration through orchestrator."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        orchestrator = MigrationOrchestrator()

        with patch("owa.cli.mcap.migrate.migrate.OWAMcapReader") as mock_reader:
            # Mock version detection
            mock_reader.return_value.__enter__.return_value.file_version = "0.3.0"

            # Mock migration path with failing migrator
            mock_migrator = MagicMock()
            mock_migrator.from_version = "0.3.0"
            mock_migrator.to_version = "0.4.0"
            mock_migrator.migrate.return_value = MagicMock(success=False, error_message="Failed")

            with patch.object(orchestrator, "get_migration_path") as mock_get_path:
                mock_get_path.return_value = [mock_migrator]

                # Call migrate_file and expect it to raise an exception
                from rich.console import Console

                console = Console()
                with pytest.raises(RuntimeError, match="Migration failed"):
                    orchestrator.migrate_file(test_file, "0.4.0", console)

    def test_orchestrator_no_migration_needed(self, temp_dir):
        """Test orchestrator when no migration is needed."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        orchestrator = MigrationOrchestrator()

        with patch("owa.cli.mcap.migrate.migrate.OWAMcapReader") as mock_reader:
            # Mock version detection
            mock_reader.return_value.__enter__.return_value.file_version = "0.4.0"

            # Mock migration path
            with patch.object(orchestrator, "get_migration_path") as mock_get_path:
                mock_get_path.return_value = []  # No migration needed

                # Call migrate_file
                from rich.console import Console

                console = Console()
                results = orchestrator.migrate_file(test_file, "0.4.0", console)

                # Should return empty results if no migration needed
                assert results == []
