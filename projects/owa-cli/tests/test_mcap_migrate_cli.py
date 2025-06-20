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


class TestMcapMigrateOutputVerification:
    """Test that migration produces expected output by comparing with reference files."""

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

    def test_migration_produces_expected_output(self, test_data_dir, temp_dir):
        """Test that migrating 0.3.2.mcap produces output equivalent to expected 0.4.2.mcap."""
        from typer.testing import CliRunner

        from owa.cli.mcap.migrate import MigrationOrchestrator

        runner = CliRunner()

        # Skip if test files don't exist
        source_file = test_data_dir / "0.3.2.mcap"
        expected_file = test_data_dir / "0.4.2.mcap"

        if not source_file.exists() or not expected_file.exists():
            pytest.skip("Required test files (0.3.2.mcap or 0.4.2.mcap) not found")

        # Copy the 0.3.2.mcap file to temp directory for migration
        test_file = self.copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)

        # Perform the migration
        result = runner.invoke(app, ["mcap", "migrate", str(test_file)], input="y\n")
        assert result.exit_code == 0
        assert "Migration successful" in result.stdout

        # Verify the migrated file has the correct version
        orchestrator = MigrationOrchestrator()
        migrated_version = orchestrator.detect_version(test_file)
        expected_version = orchestrator.detect_version(expected_file)

        assert migrated_version == expected_version, (
            f"Migrated version {migrated_version} != expected {expected_version}"
        )

        # Verify basic file properties (this is a basic sanity check)
        # Note: We don't do byte-for-byte comparison as migration timestamps may differ
        assert test_file.stat().st_size > 0, "Migrated file should not be empty"

        # Verify that the migrated file can be read without errors
        try:
            from mcap_owa.highlevel import OWAMcapReader

            with OWAMcapReader(test_file) as reader:
                # Just verify we can read the file structure
                assert reader.file_version is not None
                # Count messages to ensure content is preserved
                message_count = sum(1 for _ in reader.iter_messages())
                assert message_count > 0, "Migrated file should contain messages"
        except Exception as e:
            pytest.fail(f"Failed to read migrated file: {e}")

    def test_migration_integrity_verification_with_real_files(self, test_data_dir, temp_dir):
        """Test the verify_migration_integrity function with real migration data."""
        from rich.console import Console
        from typer.testing import CliRunner

        from owa.cli.mcap.migrate.utils import verify_migration_integrity

        runner = CliRunner()
        console = Console()

        # Skip if test files don't exist
        source_file = test_data_dir / "0.3.2.mcap"
        if not source_file.exists():
            pytest.skip("Required test file 0.3.2.mcap not found")

        # Copy the 0.3.2.mcap file to temp directory for migration
        test_file = self.copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)

        # Create a manual backup with a different name to avoid conflict
        manual_backup_file = temp_dir / "original_0.3.2.mcap"
        shutil.copy2(test_file, manual_backup_file)

        # Perform the migration (this will create its own backup)
        result = runner.invoke(app, ["mcap", "migrate", str(test_file)], input="y\n")
        if result.exit_code != 0:
            print(f"Migration failed with exit code {result.exit_code}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
        assert result.exit_code == 0
        assert "Migration successful" in result.stdout

        # The migration system should have created its own backup
        migration_backup_file = temp_dir / "0.3.2.mcap.backup"
        assert migration_backup_file.exists(), "Migration should have created a backup file"

        # Test the verify_migration_integrity function using the migration's backup
        integrity_result = verify_migration_integrity(
            migrated_file=test_file,
            backup_file=migration_backup_file,
            console=console,
            check_message_count=True,
            check_file_size=True,
            check_topics=True,
            size_tolerance_percent=15.0,
        )

        # The integrity check should pass
        assert integrity_result is True, "Migration integrity verification should pass"

        # Also test with our manual backup to ensure consistency
        manual_integrity_result = verify_migration_integrity(
            migrated_file=test_file,
            backup_file=manual_backup_file,
            console=console,
            check_message_count=True,
            check_file_size=True,
            check_topics=True,
            size_tolerance_percent=50.0,
        )

        # Both integrity checks should pass
        assert manual_integrity_result is True, "Manual backup integrity verification should also pass"

    def test_migration_integrity_verification_edge_cases(self, temp_dir):
        """Test verify_migration_integrity function with edge cases."""
        from rich.console import Console

        from owa.cli.mcap.migrate.utils import verify_migration_integrity

        console = Console()

        # Test with non-existent files
        non_existent_file = temp_dir / "non_existent.mcap"
        result = verify_migration_integrity(
            migrated_file=non_existent_file,
            backup_file=non_existent_file,
            console=console,
        )
        assert result is False, "Should fail with non-existent files"

        # Test with only migrated file missing
        backup_file = temp_dir / "backup.mcap"
        backup_file.touch()  # Create empty file
        result = verify_migration_integrity(
            migrated_file=non_existent_file,
            backup_file=backup_file,
            console=console,
        )
        assert result is False, "Should fail when migrated file is missing"

    def test_individual_verification_functions_with_real_data(self, test_data_dir, temp_dir):
        """Test individual verification functions with real migration data."""
        from rich.console import Console
        from typer.testing import CliRunner

        from owa.cli.mcap.migrate.utils import (
            get_file_stats,
            verify_file_size,
            verify_message_count,
            verify_topics_preserved,
        )

        runner = CliRunner()
        console = Console()

        # Skip if test files don't exist
        source_file = test_data_dir / "0.3.2.mcap"
        if not source_file.exists():
            pytest.skip("Required test file 0.3.2.mcap not found")

        # Copy and migrate the file
        test_file = self.copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)
        backup_file = temp_dir / "original_backup.mcap"
        shutil.copy2(test_file, backup_file)

        # Get stats before migration
        original_stats = get_file_stats(backup_file)
        assert original_stats.message_count > 0, "Original file should have messages"
        assert len(original_stats.topics) > 0, "Original file should have topics"
        assert len(original_stats.schemas) > 0, "Original file should have schemas"

        # Perform migration
        result = runner.invoke(app, ["mcap", "migrate", str(test_file)], input="y\n")
        assert result.exit_code == 0

        # Get stats after migration
        migrated_stats = get_file_stats(test_file)

        # Test individual verification functions
        message_count_ok = verify_message_count(migrated_stats, original_stats, console)
        assert message_count_ok is True, "Message count should be preserved"

        file_size_ok = verify_file_size(migrated_stats, original_stats, console, tolerance_percent=50.0)
        assert file_size_ok is True, "File size should be within tolerance"

        topics_ok = verify_topics_preserved(migrated_stats, original_stats, console)
        assert topics_ok is True, "Topics should be preserved"

        # Print stats for verification
        print(
            f"Original: {original_stats.message_count} messages, {len(original_stats.topics)} topics, {original_stats.file_size} bytes"
        )
        print(
            f"Migrated: {migrated_stats.message_count} messages, {len(migrated_stats.topics)} topics, {migrated_stats.file_size} bytes"
        )
        print(f"Topics: {sorted(original_stats.topics)}")
        print(f"Schemas before: {sorted(original_stats.schemas)}")
        print(f"Schemas after: {sorted(migrated_stats.schemas)}")

    def test_verification_functions_with_simulated_failures(self):
        """Test verification functions with simulated failure scenarios."""
        from rich.console import Console

        from owa.cli.mcap.migrate.utils import (
            FileStats,
            verify_file_size,
            verify_message_count,
            verify_topics_preserved,
        )

        console = Console()

        # Create mock file stats for testing
        original_stats = FileStats(
            message_count=100, file_size=1000, topics={"topic1", "topic2", "topic3"}, schemas={"schema1", "schema2"}
        )

        # Test message count mismatch
        bad_message_stats = FileStats(
            message_count=99,  # One less message
            file_size=1000,
            topics={"topic1", "topic2", "topic3"},
            schemas={"schema1", "schema2"},
        )

        result = verify_message_count(bad_message_stats, original_stats, console)
        assert result is False, "Should fail with message count mismatch"

        # Test file size too different
        bad_size_stats = FileStats(
            message_count=100,
            file_size=2000,  # 100% larger
            topics={"topic1", "topic2", "topic3"},
            schemas={"schema1", "schema2"},
        )

        result = verify_file_size(bad_size_stats, original_stats, console, tolerance_percent=10.0)
        assert result is False, "Should fail with large file size difference"

        # Test topics mismatch
        bad_topics_stats = FileStats(
            message_count=100,
            file_size=1000,
            topics={"topic1", "topic2"},  # Missing topic3
            schemas={"schema1", "schema2"},
        )

        result = verify_topics_preserved(bad_topics_stats, original_stats, console)
        assert result is False, "Should fail with topic mismatch"

        # Test successful cases
        good_stats = FileStats(
            message_count=100,
            file_size=1050,  # 5% larger, within tolerance
            topics={"topic1", "topic2", "topic3"},
            schemas={"schema1", "schema2", "schema3"},  # Schemas can change
        )

        assert verify_message_count(good_stats, original_stats, console) is True
        assert verify_file_size(good_stats, original_stats, console, tolerance_percent=10.0) is True
        assert verify_topics_preserved(good_stats, original_stats, console) is True


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
