"""
Tests for the MCAP migration system.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from owa.cli.mcap.migrate import (
    MigrationOrchestrator,
    MigrationResult,
    ScriptMigrator,
    detect_files_needing_migration,
    validate_migration_output,
    validate_verification_output,
)


def test_orchestrator_init():
    """Test orchestrator initialization."""
    orchestrator = MigrationOrchestrator()
    assert len(orchestrator.script_migrators) >= 0

    from mcap_owa import __version__ as mcap_owa_version

    assert orchestrator.current_version == mcap_owa_version


def test_detect_version_fallback():
    """Test version detection returns 'unknown' for non-existent file."""
    orchestrator = MigrationOrchestrator()
    non_existent_file = Path("/non/existent/file.mcap")
    version = orchestrator.detect_version(non_existent_file)
    assert version == "unknown"


# Note: Backup functionality is now handled by BackupContext in backup_utils.py
# These tests have been moved to test_backup_utils.py


def test_backup_naming_scheme():
    """Test that backup files use intuitive naming scheme."""
    test_cases = [
        ("recording.mcap", "recording.mcap.backup"),
        ("data_file.mcap", "data_file.mcap.backup"),
        ("test-file.mcap", "test-file.mcap.backup"),
        ("file.with.dots.mcap", "file.with.dots.mcap.backup"),
    ]

    for original_name, expected_backup_name in test_cases:
        original_path = Path(f"/tmp/{original_name}")
        expected_backup_path = Path(f"/tmp/{expected_backup_name}")
        actual_backup_path = original_path.with_suffix(f"{original_path.suffix}.backup")

        assert actual_backup_path == expected_backup_path
        assert actual_backup_path.name == expected_backup_name


def test_get_migration_path_no_migration_needed():
    """Test migration path when no migration is needed."""
    orchestrator = MigrationOrchestrator()
    path = orchestrator.get_migration_path("0.4.0", "0.4.0")
    assert path == []


def test_get_migration_path_sequential():
    """Test sequential migration path."""
    orchestrator = MigrationOrchestrator()
    path = orchestrator.get_migration_path("0.3.0", "0.4.2")
    assert len(path) == 2
    assert path[0].from_version == "0.3.0"
    assert path[0].to_version == "0.3.2"
    assert path[1].from_version == "0.3.2"
    assert path[1].to_version == "0.4.2"


def test_get_migration_path_from_v032():
    """Test migration path from v0.3.2 to v0.4.2."""
    orchestrator = MigrationOrchestrator()
    path = orchestrator.get_migration_path("0.3.2", "0.4.2")
    assert len(path) == 1
    assert path[0].from_version == "0.3.2"
    assert path[0].to_version == "0.4.2"


def test_get_migration_path_invalid():
    """Test invalid migration path."""
    orchestrator = MigrationOrchestrator()
    with pytest.raises(ValueError, match="No migration path found"):
        orchestrator.get_migration_path("unknown", "0.4.1")


def test_get_migration_path_version_ranges():
    """Test migration path with version ranges."""
    orchestrator = MigrationOrchestrator()

    # Test version 0.3.1 (between 0.3.0 and 0.3.2) should use 0.3.0->0.3.2 migrator
    path = orchestrator.get_migration_path("0.3.1", "0.4.2")
    assert len(path) == 2
    assert path[0].from_version == "0.3.0"
    assert path[0].to_version == "0.3.2"
    assert path[1].from_version == "0.3.2"
    assert path[1].to_version == "0.4.2"


def test_get_highest_reachable_version_ranges():
    """Test highest reachable version with version ranges."""
    orchestrator = MigrationOrchestrator()

    # Get the actual highest reachable version dynamically
    # This should be the highest version that any migrator can reach
    all_target_versions = [m.to_version for m in orchestrator.script_migrators]
    if all_target_versions:
        from packaging.version import Version

        expected_highest = str(max(Version(v) for v in all_target_versions))
    else:
        expected_highest = "0.4.2"  # Fallback if no migrators found

    # Test version 0.3.1 (between 0.3.0 and 0.3.2) should reach the highest available
    highest = orchestrator.get_highest_reachable_version("0.3.1")
    assert highest == expected_highest

    # Test version 0.3.5 (between 0.3.2 and 0.4.2) should reach the highest available
    highest = orchestrator.get_highest_reachable_version("0.3.5")
    assert highest == expected_highest


def test_automatic_migrator_discovery():
    """Test that migrators are automatically discovered from filenames."""
    orchestrator = MigrationOrchestrator()

    # Should have discovered migrators (dynamic count based on actual files)
    assert len(orchestrator.script_migrators) >= 3  # At least the original 3 migrators

    # Verify that all discovered migrators have valid version formats
    for migrator in orchestrator.script_migrators:
        assert migrator.from_version is not None
        assert migrator.to_version is not None
        # Verify version format (should be parseable by packaging.version)
        from packaging.version import Version

        Version(migrator.from_version)  # Should not raise
        Version(migrator.to_version)  # Should not raise

    # Check that the versions were parsed correctly from filenames
    migrator_versions = [(m.from_version, m.to_version) for m in orchestrator.script_migrators]
    expected_versions = [("0.2.0", "0.3.0"), ("0.3.0", "0.3.2"), ("0.3.2", "0.4.2")]

    # Sort both lists to ensure consistent comparison
    migrator_versions.sort()
    expected_versions.sort()

    assert migrator_versions == expected_versions


def test_filename_pattern_parsing():
    """Test that the filename pattern parsing works correctly."""
    import re

    # Test the regex pattern used in the discovery
    version_pattern = re.compile(r"^v(\d+_\d+_\d+)_to_v(\d+_\d+_\d+)\.py$")

    # Test valid patterns
    valid_cases = [
        ("v0_3_0_to_v0_3_2.py", "0.3.0", "0.3.2"),
        ("v0_3_2_to_v0_4_1.py", "0.3.2", "0.4.1"),
        ("v1_0_0_to_v2_0_0.py", "1.0.0", "2.0.0"),
        ("v0_4_1_to_v0_5_0.py", "0.4.1", "0.5.0"),
    ]

    for filename, expected_from, expected_to in valid_cases:
        match = version_pattern.match(filename)
        assert match is not None, f"Pattern should match {filename}"
        from_version = match.group(1).replace("_", ".")
        to_version = match.group(2).replace("_", ".")
        assert from_version == expected_from
        assert to_version == expected_to

    # Test invalid patterns
    invalid_cases = [
        "v0_3_0_to_v0_3_2.txt",  # Wrong extension
        "0_3_0_to_v0_3_2.py",  # Missing 'v' prefix
        "v0_3_to_v0_3_2.py",  # Incomplete version
        "v0_3_0_v0_3_2.py",  # Missing 'to'
        "random_file.py",  # Completely different pattern
    ]

    for filename in invalid_cases:
        match = version_pattern.match(filename)
        assert match is None, f"Pattern should not match {filename}"


# File detection tests
def test_detect_files_no_files():
    """Test detection when no files are provided."""
    console = MagicMock()
    result = detect_files_needing_migration([], console, False, None)
    assert result == []
    console.print.assert_called_with("[yellow]No valid MCAP files found[/yellow]")


def test_detect_files_with_non_existent_file():
    """Test detection with non-existent file."""
    console = MagicMock()
    non_existent_file = Path("/non/existent/file.mcap")

    result = detect_files_needing_migration([non_existent_file], console, False, None)

    assert result == []
    console.print.assert_any_call(f"[red]File not found: {non_existent_file}[/red]")


# Integration tests
def test_migration_orchestrator_with_mocked_reader():
    """Test migration orchestrator with mocked OWAMcapReader."""
    orchestrator = MigrationOrchestrator()

    # Test that the orchestrator can be created and has script migrators
    assert len(orchestrator.script_migrators) >= 3  # At least the original 3 migrators

    # Test migration path calculation
    path_0_3_0_to_0_4_2 = orchestrator.get_migration_path("0.3.0", "0.4.2")
    assert len(path_0_3_0_to_0_4_2) == 2

    path_0_3_2_to_0_4_2 = orchestrator.get_migration_path("0.3.2", "0.4.2")
    assert len(path_0_3_2_to_0_4_2) == 1

    # Test the new migrator path
    path_0_2_0_to_0_4_2 = orchestrator.get_migration_path("0.2.0", "0.4.2")
    assert len(path_0_2_0_to_0_4_2) == 3


@patch("owa.cli.mcap.migrate.migrate.OWAMcapReader")
def test_detect_version_with_mocked_reader(mock_reader_class, temp_dir):
    """Test version detection with mocked reader."""
    # Setup mock
    mock_reader = MagicMock()
    mock_reader.file_version = "0.3.2"
    mock_reader_class.return_value.__enter__.return_value = mock_reader

    orchestrator = MigrationOrchestrator()

    tmp_path = temp_dir / "test.mcap"
    tmp_path.touch()

    version = orchestrator.detect_version(tmp_path)
    assert version == "0.3.2"


@patch("owa.cli.mcap.migrate.migrate.OWAMcapReader")
def test_multi_step_migration_with_script_migrators(mock_reader_class, temp_dir):
    """Test that multi-step migration works with script migrators."""
    orchestrator = MigrationOrchestrator()

    # Mock the reader to simulate version progression
    mock_reader = MagicMock()
    mock_reader.file_version = "0.3.0"  # Initial version
    mock_reader_class.return_value.__enter__.return_value = mock_reader

    # Create mock script migrators
    from owa.cli.mcap.migrate import ScriptMigrator

    mock_migrator_1 = MagicMock(spec=ScriptMigrator)
    mock_migrator_1.from_version = "0.3.0"
    mock_migrator_1.to_version = "0.3.2"
    mock_migrator_1.migrate.return_value = MigrationResult(
        success=True,
        version_from="0.3.0",
        version_to="0.3.2",
        changes_made=1,
    )
    mock_migrator_1.verify_migration.return_value = True

    mock_migrator_2 = MagicMock(spec=ScriptMigrator)
    mock_migrator_2.from_version = "0.3.2"
    mock_migrator_2.to_version = "0.4.2"
    mock_migrator_2.migrate.return_value = MigrationResult(
        success=True,
        version_from="0.3.2",
        version_to="0.4.2",
        changes_made=1,
    )
    mock_migrator_2.verify_migration.return_value = True

    # Replace orchestrator's script migrators with our mocks
    orchestrator.script_migrators = [mock_migrator_1, mock_migrator_2]

    tmp_path = temp_dir / "test.mcap"
    tmp_path.touch()

    # Mock the console for migrate_file
    from rich.console import Console

    console = Console()

    # This should trigger two migrations: 0.3.0 -> 0.3.2 -> 0.4.2
    results = orchestrator.migrate_file(tmp_path, target_version="0.4.2", console=console)

    # Verify migration was successful
    assert len(results) == 2
    assert all(result.success for result in results)

    # Verify that both migrators were called
    mock_migrator_1.migrate.assert_called_once()
    mock_migrator_2.migrate.assert_called_once()


class TestScriptMigratorErrorHandling:
    """Test error handling improvements in ScriptMigrator."""

    def test_migrate_with_json_error_output(self, temp_dir):
        """Test that JSON error output is properly parsed and handled."""
        script_path = temp_dir / "test_migrator.py"
        script_path.write_text("#!/usr/bin/env python3\nprint('test')")

        migrator = ScriptMigrator(script_path, "0.3.0", "0.4.0")

        # Mock subprocess to return JSON error output with non-zero exit code
        error_json = '{"success": false, "changes_made": 0, "error": "Test migration error", "from_version": "0.3.0", "to_version": "0.4.0"}'
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = error_json
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = migrator.migrate(temp_dir / "test.mcap", verbose=False)

        assert not result.success
        assert result.error_message == "Test migration error"
        assert result.changes_made == 0
        assert result.version_from == "0.3.0"
        assert result.version_to == "0.4.0"

    def test_migrate_with_invalid_json_error_fallback(self, temp_dir):
        """Test fallback to stderr/stdout when JSON is invalid."""
        script_path = temp_dir / "test_migrator.py"
        script_path.write_text("#!/usr/bin/env python3\nprint('test')")

        migrator = ScriptMigrator(script_path, "0.3.0", "0.4.0")

        # Mock subprocess to return invalid JSON with non-zero exit code
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "Invalid JSON output"
        mock_result.stderr = "Script execution failed"

        with patch("subprocess.run", return_value=mock_result):
            result = migrator.migrate(temp_dir / "test.mcap", verbose=False)

        assert not result.success
        assert result.error_message == "Script execution failed"
        assert result.changes_made == 0

    def test_migrate_with_json_success_false_on_zero_exit(self, temp_dir):
        """Test handling JSON with success=false even when exit code is 0."""
        script_path = temp_dir / "test_migrator.py"
        script_path.write_text("#!/usr/bin/env python3\nprint('test')")

        migrator = ScriptMigrator(script_path, "0.3.0", "0.4.0")

        # Mock subprocess to return JSON with success=false but exit code 0
        error_json = '{"success": false, "changes_made": 0, "error": "Validation failed", "from_version": "0.3.0", "to_version": "0.4.0"}'
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = error_json
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = migrator.migrate(temp_dir / "test.mcap", verbose=False)

        assert not result.success
        assert result.error_message == "Validation failed"
        assert result.changes_made == 0

    def test_verify_migration_with_json_error_output(self, temp_dir):
        """Test that verification JSON error output is properly handled."""
        script_path = temp_dir / "test_migrator.py"
        script_path.write_text("#!/usr/bin/env python3\nprint('test')")

        migrator = ScriptMigrator(script_path, "0.3.0", "0.4.0")

        # Mock subprocess to return JSON error output
        error_json = '{"success": false, "error": "Legacy structures found"}'
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = error_json
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = migrator.verify_migration(temp_dir / "test.mcap", None, verbose=False)

        assert not result

    def test_verify_migration_with_json_success_output(self, temp_dir):
        """Test that verification JSON success output is properly handled."""
        script_path = temp_dir / "test_migrator.py"
        script_path.write_text("#!/usr/bin/env python3\nprint('test')")

        migrator = ScriptMigrator(script_path, "0.3.0", "0.4.0")

        # Mock subprocess to return JSON success output
        success_json = '{"success": true, "message": "Verification completed successfully"}'
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = success_json
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = migrator.verify_migration(temp_dir / "test.mcap", None, verbose=False)

        assert result


class TestOutputValidation:
    """Test output validation functions."""

    def test_validate_migration_output_success(self):
        """Test validation of successful migration output."""
        valid_success = {"success": True, "changes_made": 5, "from_version": "0.3.0", "to_version": "0.4.0"}
        assert validate_migration_output(valid_success)

    def test_validate_migration_output_failure(self):
        """Test validation of failed migration output."""
        valid_failure = {
            "success": False,
            "changes_made": 0,
            "error": "Migration failed",
            "from_version": "0.3.0",
            "to_version": "0.4.0",
        }
        assert validate_migration_output(valid_failure)

    def test_validate_migration_output_invalid_success(self):
        """Test validation rejects invalid success output."""
        invalid_success = {
            "success": True,
            "changes_made": 5,
            # Missing from_version and to_version
        }
        assert not validate_migration_output(invalid_success)

    def test_validate_migration_output_invalid_failure(self):
        """Test validation rejects invalid failure output."""
        invalid_failure = {
            "success": False,
            "changes_made": 5,  # Should be 0 for failure
            "error": "Migration failed",
        }
        assert not validate_migration_output(invalid_failure)

    def test_validate_verification_output_success(self):
        """Test validation of successful verification output."""
        valid_success = {"success": True, "message": "Verification completed successfully"}
        assert validate_verification_output(valid_success)

    def test_validate_verification_output_failure(self):
        """Test validation of failed verification output."""
        valid_failure = {"success": False, "error": "Legacy structures found"}
        assert validate_verification_output(valid_failure)

    def test_validate_verification_output_invalid(self):
        """Test validation rejects invalid verification output."""
        invalid_output = {
            "success": True,
            # Missing message field
        }
        assert not validate_verification_output(invalid_output)
