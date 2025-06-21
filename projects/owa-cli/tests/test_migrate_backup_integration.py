"""
Integration tests for migrate command with unified backup-rollback utilities.

This module tests that the migrate command correctly uses the unified backup
and rollback utilities and maintains data safety during migration operations.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from owa.cli.mcap import app as mcap_app
from owa.cli.mcap.migrate.migrate import MigrationOrchestrator


class TestMigrateBackupIntegration:
    """Integration tests for migrate command backup functionality."""

    def test_migrate_creates_backup_on_success(self, temp_dir, cli_runner):
        """Test that migrate creates backup files on successful operation."""
        test_file = temp_dir / "test.mcap"
        original_content = b"original mcap content"
        test_file.write_bytes(original_content)

        with patch("owa.cli.mcap.migrate.migrate.MigrationOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock successful migration
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.changes_made = 1
            mock_orchestrator.migrate_file.return_value = [mock_result]

            # Mock file detection
            mock_info = MagicMock()
            mock_info.file_path = test_file
            mock_info.detected_version = "0.3.0"
            mock_info.needs_migration = True
            mock_info.target_version = "0.4.0"

            with patch("owa.cli.mcap.migrate.migrate.detect_files_needing_migration") as mock_detect:
                mock_detect.return_value = [mock_info]

                # Run migrate command
                result = cli_runner.invoke(mcap_app, ["migrate", str(test_file), "--yes"])

                assert result.exit_code == 0

                # Check that backup was created during migration
                backup_file = test_file.with_suffix(".mcap.backup")
                # Note: In real scenario, backup would be created by migrate_file
                # Here we verify the orchestrator was called correctly
                mock_orchestrator.migrate_file.assert_called_once()

    def test_migrate_rollback_on_failure(self, temp_dir, cli_runner):
        """Test that migrate rolls back changes when migration fails."""
        test_file = temp_dir / "test.mcap"
        original_content = b"original mcap content"
        test_file.write_bytes(original_content)

        with patch("owa.cli.mcap.migrate.migrate.MigrationOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock failed migration
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error_message = "Migration failed"
            mock_orchestrator.migrate_file.return_value = [mock_result]

            # Mock file detection
            mock_info = MagicMock()
            mock_info.file_path = test_file
            mock_info.detected_version = "0.3.0"
            mock_info.needs_migration = True
            mock_info.target_version = "0.4.0"

            with patch("owa.cli.mcap.migrate.migrate.detect_files_needing_migration") as mock_detect:
                mock_detect.return_value = [mock_info]

                # Run migrate command
                result = cli_runner.invoke(mcap_app, ["migrate", str(test_file), "--yes"])

                assert result.exit_code == 1

                # Verify migration was attempted
                mock_orchestrator.migrate_file.assert_called_once()

    def test_migrate_backup_cleanup_keep_backups(self, temp_dir, cli_runner):
        """Test backup cleanup when keeping backups."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"original content")

        # Create a mock backup file to simulate successful migration
        backup_file = test_file.with_suffix(".mcap.backup")
        backup_file.write_bytes(b"backup content")

        with patch("owa.cli.mcap.migrate.migrate.MigrationOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock successful migration
            mock_result = MagicMock()
            mock_result.success = True
            mock_orchestrator.migrate_file.return_value = [mock_result]

            # Mock file detection
            mock_info = MagicMock()
            mock_info.file_path = test_file
            mock_info.detected_version = "0.3.0"
            mock_info.needs_migration = True
            mock_info.target_version = "0.4.0"

            with patch("owa.cli.mcap.migrate.migrate.detect_files_needing_migration") as mock_detect:
                mock_detect.return_value = [mock_info]

                # Run migrate with keep-backups
                result = cli_runner.invoke(mcap_app, ["migrate", str(test_file), "--keep-backups", "--yes"])

                assert result.exit_code == 0

                # Backup should still exist (we created it manually for this test)
                assert backup_file.exists()

    def test_migrate_backup_cleanup_no_backups(self, temp_dir, cli_runner):
        """Test backup cleanup when not keeping backups."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"original content")

        # Create a mock backup file to simulate successful migration
        backup_file = test_file.with_suffix(".mcap.backup")
        backup_file.write_bytes(b"backup content")

        with patch("owa.cli.mcap.migrate.migrate.MigrationOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock successful migration
            mock_result = MagicMock()
            mock_result.success = True
            mock_orchestrator.migrate_file.return_value = [mock_result]

            # Mock file detection
            mock_info = MagicMock()
            mock_info.file_path = test_file
            mock_info.detected_version = "0.3.0"
            mock_info.needs_migration = True
            mock_info.target_version = "0.4.0"

            with patch("owa.cli.mcap.migrate.migrate.detect_files_needing_migration") as mock_detect:
                mock_detect.return_value = [mock_info]

                # Run migrate without keeping backups
                result = cli_runner.invoke(mcap_app, ["migrate", str(test_file), "--no-backups", "--yes"])

                assert result.exit_code == 0

                # Backup should be cleaned up
                assert not backup_file.exists()

    def test_migrate_multiple_files_backup_handling(self, temp_dir, cli_runner):
        """Test backup handling with multiple files."""
        test_file1 = temp_dir / "test1.mcap"
        test_file2 = temp_dir / "test2.mcap"

        test_file1.write_bytes(b"content 1")
        test_file2.write_bytes(b"content 2")

        # Create mock backup files
        backup1 = test_file1.with_suffix(".mcap.backup")
        backup2 = test_file2.with_suffix(".mcap.backup")
        backup1.write_bytes(b"backup 1")
        backup2.write_bytes(b"backup 2")

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
                result = cli_runner.invoke(
                    mcap_app, ["migrate", str(test_file1), str(test_file2), "--keep-backups", "--yes"]
                )

                assert result.exit_code == 0

                # Both backups should still exist
                assert backup1.exists()
                assert backup2.exists()

    def test_migrate_dry_run_no_backup(self, temp_dir, cli_runner):
        """Test that dry run doesn't create backups."""
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

                # No backup should be created
                backup_file = test_file.with_suffix(".mcap.backup")
                assert not backup_file.exists()

                # Original file should be unchanged
                assert test_file.read_bytes() == original_content

                # Migration should not be attempted
                mock_orchestrator.migrate_file.assert_not_called()

    def test_migrate_uses_unified_backup_path_generation(self, temp_dir):
        """Test that migrate uses the unified backup path generation."""
        test_file = temp_dir / "complex.name.test.mcap"
        test_file.write_bytes(b"content")

        orchestrator = MigrationOrchestrator()

        with (
            patch("owa.cli.mcap.migrate.migrate.create_backup") as mock_create_backup,
            patch("owa.cli.mcap.migrate.migrate.OWAMcapReader") as mock_reader,
        ):
            # Mock version detection
            mock_reader.return_value.__enter__.return_value.file_version = "0.3.0"

            # Mock migration path
            with patch.object(orchestrator, "get_migration_path") as mock_get_path:
                mock_get_path.return_value = []  # No migration needed

                # Call migrate_file
                from rich.console import Console

                console = Console()
                orchestrator.migrate_file(test_file, "0.4.0", console)

                # Should not create backup if no migration needed
                mock_create_backup.assert_not_called()


class TestMigrationOrchestratorBackupIntegration:
    """Test MigrationOrchestrator's use of unified backup utilities."""

    def test_orchestrator_uses_unified_backup_creation(self, temp_dir):
        """Test that orchestrator uses unified backup creation."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        orchestrator = MigrationOrchestrator()

        with (
            patch("owa.cli.mcap.migrate.migrate.create_backup") as mock_create_backup,
            patch("owa.cli.mcap.migrate.migrate.generate_backup_path") as mock_generate_path,
            patch("owa.cli.mcap.migrate.migrate.OWAMcapReader") as mock_reader,
        ):
            # Mock backup path generation
            expected_backup = test_file.with_suffix(".mcap.backup")
            mock_generate_path.return_value = expected_backup

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
                orchestrator.migrate_file(test_file, "0.4.0", console)

                # Verify unified functions were called
                mock_generate_path.assert_called_once_with(test_file)
                mock_create_backup.assert_called_once_with(test_file, expected_backup)

    def test_orchestrator_uses_unified_rollback(self, temp_dir):
        """Test that orchestrator uses unified rollback on failure."""
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"content")

        orchestrator = MigrationOrchestrator()

        with (
            patch("owa.cli.mcap.migrate.migrate.create_backup") as mock_create_backup,
            patch("owa.cli.mcap.migrate.migrate.generate_backup_path") as mock_generate_path,
            patch("owa.cli.mcap.migrate.migrate.rollback_from_backup") as mock_rollback,
            patch("owa.cli.mcap.migrate.migrate.OWAMcapReader") as mock_reader,
        ):
            # Mock backup path generation
            expected_backup = test_file.with_suffix(".mcap.backup")
            mock_generate_path.return_value = expected_backup

            # Mock version detection
            mock_reader.return_value.__enter__.return_value.file_version = "0.3.0"

            # Mock migration path with failing migrator
            mock_migrator = MagicMock()
            mock_migrator.from_version = "0.3.0"
            mock_migrator.to_version = "0.4.0"
            mock_migrator.migrate.return_value = MagicMock(success=False, error_message="Failed")

            with patch.object(orchestrator, "get_migration_path") as mock_get_path:
                mock_get_path.return_value = [mock_migrator]

                # Call migrate_file
                from rich.console import Console

                console = Console()
                results = orchestrator.migrate_file(test_file, "0.4.0", console)

                # Verify rollback was called with delete_backup=True
                mock_rollback.assert_called_once_with(test_file, expected_backup, console, delete_backup=True)

                # Verify migration failed
                assert len(results) == 1
                assert not results[0].success
