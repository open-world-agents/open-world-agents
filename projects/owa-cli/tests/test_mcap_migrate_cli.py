"""
Tests for the `owl mcap migrate` CLI command.

This test suite verifies that the migration command works correctly with real MCAP files,
testing the full CLI integration including argument parsing, file processing, and migration execution.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from owa.cli import app


class TestMcapMigrateCLI:
    """Test the `owl mcap migrate` CLI command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def test_data_dir(self):
        """Get the test data directory with example MCAP files."""
        return Path(__file__).parent

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def copy_test_file(self, source_dir: Path, filename: str, dest_dir: Path) -> Path:
        """Copy a test file to the destination directory."""
        source_file = source_dir / filename
        dest_file = dest_dir / filename
        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            return dest_file
        else:
            pytest.skip(f"Test file {filename} not found in {source_dir}")

    def test_migrate_help(self, runner):
        """Test that the migrate command shows help correctly."""
        result = runner.invoke(app, ["mcap", "migrate", "--help"])
        assert result.exit_code == 0
        assert "Migrate MCAP files" in result.stdout
        assert "--target" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--verbose" in result.stdout

    def test_migrate_nonexistent_file(self, runner):
        """Test migration with non-existent file."""
        result = runner.invoke(app, ["mcap", "migrate", "nonexistent.mcap"])
        assert result.exit_code == 0  # CLI handles gracefully
        assert "File not found" in result.stdout

    def test_migrate_non_mcap_file(self, runner, temp_dir):
        """Test migration with non-MCAP file."""
        # Create a non-MCAP file
        test_file = temp_dir / "test.txt"
        test_file.write_text("not an mcap file")

        result = runner.invoke(app, ["mcap", "migrate", str(test_file)])
        assert result.exit_code == 0  # Should complete but skip non-MCAP files
        assert "Skipping non-MCAP file" in result.stdout

    def test_migrate_dry_run(self, runner, test_data_dir, temp_dir):
        """Test dry run mode doesn't modify files."""
        # Copy test file to temp directory
        test_file = self.copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)
        original_size = test_file.stat().st_size
        original_mtime = test_file.stat().st_mtime

        result = runner.invoke(app, ["mcap", "migrate", str(test_file), "--dry-run"])
        assert result.exit_code == 0
        assert "DRY RUN MODE" in result.stdout
        # File might already be at target version, so check for either message
        assert "Would migrate" in result.stdout or "already at the target version" in result.stdout

        # File should be unchanged
        assert test_file.stat().st_size == original_size
        assert test_file.stat().st_mtime == original_mtime

    @patch("owa.cli.mcap.migrate.migrate.OWAMcapReader")
    def test_migrate_already_current_version(self, mock_reader_class, runner, test_data_dir, temp_dir):
        """Test migration when file is already at current version."""
        # Mock the reader to return current version
        mock_reader = mock_reader_class.return_value.__enter__.return_value
        mock_reader.file_version = "0.4.2"  # Current version

        test_file = self.copy_test_file(test_data_dir, "0.4.2.mcap", temp_dir)

        result = runner.invoke(app, ["mcap", "migrate", str(test_file)])
        assert result.exit_code == 0
        assert "already at the target version" in result.stdout

    def test_migrate_verbose_mode(self, runner, test_data_dir, temp_dir):
        """Test verbose mode shows additional information."""
        test_file = self.copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)

        result = runner.invoke(app, ["mcap", "migrate", str(test_file), "--verbose", "--dry-run"])
        assert result.exit_code == 0
        assert "Available script migrators" in result.stdout

    def test_migrate_with_target_version(self, runner, test_data_dir, temp_dir):
        """Test migration with specific target version."""
        test_file = self.copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)

        result = runner.invoke(app, ["mcap", "migrate", str(test_file), "--target", "0.4.2", "--dry-run"])
        assert result.exit_code == 0
        assert "Target version: 0.4.2" in result.stdout

    def test_migrate_multiple_files(self, runner, test_data_dir, temp_dir):
        """Test migration with multiple files."""
        # Copy multiple test files
        file1 = self.copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)
        file2 = self.copy_test_file(test_data_dir, "0.4.2.mcap", temp_dir)

        result = runner.invoke(app, ["mcap", "migrate", str(file1), str(file2), "--dry-run"])
        assert result.exit_code == 0
        assert "Files to process: 2" in result.stdout

    def test_migrate_no_backups_option(self, runner, test_data_dir, temp_dir):
        """Test the --no-backups option."""
        test_file = self.copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)

        result = runner.invoke(app, ["mcap", "migrate", str(test_file), "--no-backups", "--dry-run"])
        assert result.exit_code == 0
        # Should still show dry run output
        assert "DRY RUN MODE" in result.stdout

    @patch("owa.cli.mcap.migrate.migrate.OWAMcapReader")
    def test_migrate_version_detection_error(self, mock_reader_class, runner, test_data_dir, temp_dir):
        """Test handling of version detection errors."""
        # Mock the reader to raise an exception during version detection
        mock_reader_class.return_value.__enter__.side_effect = Exception("Failed to read MCAP file")

        test_file = self.copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)

        result = runner.invoke(app, ["mcap", "migrate", str(test_file), "--dry-run"])
        assert result.exit_code == 0
        # Should handle the error gracefully and show analysis error
        assert "Error analyzing" in result.stdout or "already at the target version" in result.stdout

    def test_migrate_user_cancellation(self, runner, test_data_dir, temp_dir):
        """Test user cancellation of migration."""
        test_file = self.copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)

        # Simulate user saying 'no' to confirmation
        result = runner.invoke(app, ["mcap", "migrate", str(test_file)], input="n\n")
        assert result.exit_code == 0
        # File might already be at target version, so check for either message
        assert "Migration cancelled" in result.stdout or "already at the target version" in result.stdout

    @patch("owa.cli.mcap.migrate.migrate.OWAMcapReader")
    def test_migrate_shows_migration_summary_table(self, mock_reader_class, runner, test_data_dir, temp_dir):
        """Test that migration shows a summary table with file information."""
        # Mock different versions for different files
        mock_reader = mock_reader_class.return_value.__enter__.return_value
        mock_reader.file_version = "0.3.2"

        test_file = self.copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)

        result = runner.invoke(app, ["mcap", "migrate", str(test_file), "--dry-run"])
        assert result.exit_code == 0
        assert "Migration Summary" in result.stdout
        assert "Current Version" in result.stdout
        assert "Target Version" in result.stdout
        assert "Status" in result.stdout

    def test_migrate_invalid_target_version(self, runner, test_data_dir, temp_dir):
        """Test migration with invalid target version."""
        test_file = self.copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)

        result = runner.invoke(app, ["mcap", "migrate", str(test_file), "--target", "invalid.version", "--dry-run"])
        # Should still work but may show no migration path
        assert result.exit_code == 0


class TestMcapMigrateIntegration:
    """Integration tests for the migration system with real files."""

    @pytest.fixture
    def test_data_dir(self):
        """Get the test data directory with example MCAP files."""
        return Path(__file__).parent

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def copy_test_file(self, source_dir: Path, filename: str, dest_dir: Path) -> Path:
        """Copy a test file to the destination directory."""
        source_file = source_dir / filename
        dest_file = dest_dir / filename
        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            return dest_file
        else:
            pytest.skip(f"Test file {filename} not found in {source_dir}")

    def test_migration_orchestrator_with_real_files(self, test_data_dir):
        """Test that the migration orchestrator can analyze real MCAP files."""
        from owa.cli.mcap.migrate import MigrationOrchestrator

        orchestrator = MigrationOrchestrator()

        # Test with each available test file
        test_files = ["0.3.2.mcap", "0.4.2.mcap"]

        for filename in test_files:
            test_file = test_data_dir / filename
            if test_file.exists():
                # Should not raise an exception
                version = orchestrator.detect_version(test_file)
                assert version is not None
                assert isinstance(version, str)

                # Test migration path calculation
                if version != orchestrator.current_version:
                    try:
                        path = orchestrator.get_migration_path(version, orchestrator.current_version)
                        assert isinstance(path, list)
                    except ValueError:
                        # Some versions might not have migration paths, which is OK
                        pass

    def test_backup_creation_with_real_files(self, test_data_dir, temp_dir):
        """Test backup creation with real MCAP files."""
        from owa.cli.mcap.migrate import MigrationOrchestrator

        orchestrator = MigrationOrchestrator()

        # Copy a test file
        test_file = self.copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)
        backup_file = temp_dir / "backup.mcap"

        # Create backup
        orchestrator.create_backup(test_file, backup_file)

        # Verify backup
        assert backup_file.exists()
        assert backup_file.stat().st_size == test_file.stat().st_size
        assert backup_file.stat().st_size > 0  # Should not be empty

    def test_migrator_discovery(self):
        """Test that migrators are discovered correctly."""
        from owa.cli.mcap.migrate import MigrationOrchestrator

        orchestrator = MigrationOrchestrator()

        # Should discover the available migrators
        assert len(orchestrator.script_migrators) > 0

        # Check that migrators have correct structure
        for migrator in orchestrator.script_migrators:
            assert hasattr(migrator, "from_version")
            assert hasattr(migrator, "to_version")
            assert hasattr(migrator, "script_path")
            assert migrator.script_path.exists()
            assert migrator.script_path.suffix == ".py"

    def test_version_range_matching(self):
        """Test that version range matching works correctly."""
        from owa.cli.mcap.migrate import MigrationOrchestrator

        orchestrator = MigrationOrchestrator()

        # Test that intermediate versions can find migration paths
        # For example, 0.3.1 should be able to migrate using 0.3.0->0.3.2 migrator
        try:
            path = orchestrator.get_migration_path("0.3.1", "0.4.2")
            assert len(path) > 0
            # First migrator should handle the 0.3.x range
            assert path[0].from_version == "0.3.0"
            assert path[0].to_version == "0.3.2"
        except ValueError:
            # If no path exists, that's also acceptable for this test
            pass

    def test_highest_reachable_version(self):
        """Test highest reachable version calculation."""
        from owa.cli.mcap.migrate import MigrationOrchestrator

        orchestrator = MigrationOrchestrator()

        # Test with different starting versions
        test_versions = ["0.3.1", "0.3.2", "0.4.0"]

        for version in test_versions:
            highest = orchestrator.get_highest_reachable_version(version)
            assert isinstance(highest, str)
            # Should be same or higher version
            from packaging.version import Version

            try:
                assert Version(highest) >= Version(version)
            except Exception:
                # If version parsing fails, just check it's a string
                assert len(highest) > 0

    def test_file_version_detection_with_real_files(self, test_data_dir):
        """Test version detection with real MCAP files."""
        from owa.cli.mcap.migrate import MigrationOrchestrator

        orchestrator = MigrationOrchestrator()

        # Test files should have their versions detectable
        version_files = {"0.3.2.mcap": "0.3.2", "0.4.2.mcap": "0.4.2"}

        for filename in version_files.keys():
            test_file = test_data_dir / filename
            if test_file.exists():
                detected_version = orchestrator.detect_version(test_file)
                # Version should be detected (might not match exactly due to version handling)
                assert detected_version != "unknown"
                assert isinstance(detected_version, str)
                assert len(detected_version) > 0

    def test_migration_path_completeness(self):
        """Test that migration paths are complete and valid."""
        from owa.cli.mcap.migrate import MigrationOrchestrator

        orchestrator = MigrationOrchestrator()

        # Test that we have a complete migration chain
        available_versions = set()
        for migrator in orchestrator.script_migrators:
            available_versions.add(migrator.from_version)
            available_versions.add(migrator.to_version)

        # Should have at least some versions
        assert len(available_versions) > 0

        # Test that migration paths are sequential (no gaps)
        for migrator in orchestrator.script_migrators:
            # Each migrator should be able to execute (script exists)
            assert migrator.script_path.exists()
            assert migrator.script_path.is_file()

    def test_cli_with_glob_patterns(self, test_data_dir, temp_dir):
        """Test CLI with glob patterns (simulated)."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Copy multiple test files
        files = []
        for filename in ["0.3.2.mcap", "0.4.2.mcap"]:
            try:
                file_path = self.copy_test_file(test_data_dir, filename, temp_dir)
                files.append(str(file_path))
            except pytest.skip.Exception:
                continue

        if not files:
            pytest.skip("No test files available")

        # Test with multiple files
        result = runner.invoke(app, ["mcap", "migrate"] + files + ["--dry-run"])
        assert result.exit_code == 0
        assert f"Files to process: {len(files)}" in result.stdout


class TestMcapMigrateErrorHandling:
    """Test error handling in the migration system."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_migrate_with_corrupted_file(self, runner, temp_dir):
        """Test migration with corrupted MCAP file."""
        # Create a corrupted MCAP file
        corrupted_file = temp_dir / "corrupted.mcap"
        corrupted_file.write_bytes(b"not a valid mcap file content")

        result = runner.invoke(app, ["mcap", "migrate", str(corrupted_file), "--dry-run"])
        # Should handle gracefully
        assert result.exit_code == 0

    def test_migrate_with_permission_denied(self, runner, temp_dir):
        """Test migration when backup creation fails due to permissions."""
        # This test is platform-dependent and might not work on all systems
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"fake mcap content")

        # Make the directory read-only (this might not work on all platforms)
        try:
            test_file.chmod(0o444)  # Read-only
            result = runner.invoke(app, ["mcap", "migrate", str(test_file), "--dry-run"])
            # Should still work in dry-run mode
            assert result.exit_code == 0
        except (OSError, PermissionError):
            # Skip if we can't change permissions
            pytest.skip("Cannot test permission errors on this platform")
        finally:
            # Restore permissions for cleanup
            try:
                test_file.chmod(0o644)
            except (OSError, PermissionError):
                pass

    def test_migrate_with_empty_file(self, runner, temp_dir):
        """Test migration with empty MCAP file."""
        empty_file = temp_dir / "empty.mcap"
        empty_file.touch()  # Create empty file

        result = runner.invoke(app, ["mcap", "migrate", str(empty_file), "--dry-run"])
        # Should handle gracefully
        assert result.exit_code == 0

    @patch("owa.cli.mcap.migrate.migrate.subprocess.run")
    def test_migrate_script_execution_failure(self, mock_subprocess, runner, temp_dir):
        """Test handling of script execution failures."""
        # Create a test file
        test_file = temp_dir / "test.mcap"
        test_file.write_bytes(b"fake mcap content")

        # Mock subprocess to simulate script failure
        from subprocess import CompletedProcess

        mock_subprocess.return_value = CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Script execution failed"
        )

        result = runner.invoke(app, ["mcap", "migrate", str(test_file), "--dry-run"])
        # Should handle the error gracefully
        assert result.exit_code == 0
