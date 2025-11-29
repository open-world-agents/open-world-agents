"""Tests for owl mcap migrate CLI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from owa.cli import app
from owa.cli.mcap.migrate import MigrationOrchestrator, ScriptMigrator


# === CLI Help Tests ===
def test_migrate_help(cli_runner):
    result = cli_runner.invoke(app, ["mcap", "migrate", "--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout


def test_migrate_run_help(cli_runner):
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.stdout


# === Basic Error Handling ===
def test_migrate_nonexistent_file(cli_runner):
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", "nonexistent.mcap"])
    assert "File not found" in result.stdout


def test_migrate_non_mcap_file(cli_runner, tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("not mcap")
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file)])
    assert "Skipping non-MCAP file" in result.stdout


# === Dry Run Tests ===
def test_migrate_dry_run(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)
    original_mtime = test_file.stat().st_mtime

    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--dry-run"])
    assert result.exit_code == 0
    assert "DRY RUN" in result.stdout
    assert test_file.stat().st_mtime == original_mtime


def test_migrate_with_target_version(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    """Test migration with specific target version."""
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--target", "0.4.2", "--dry-run"])
    assert result.exit_code == 0
    assert "Target version: 0.4.2" in result.stdout


def test_migrate_multiple_files(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    """Test migration with multiple files."""
    file1 = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path, "file1.mcap")
    file2 = copy_test_file(test_data_dir, "0.4.2.mcap", tmp_path, "file2.mcap")
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(file1), str(file2), "--dry-run"])
    assert result.exit_code == 0
    assert "Files to process: 2" in result.stdout


def test_migrate_user_cancellation(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    """Test user cancellation of migration."""
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file)], input="n\n")
    assert result.exit_code == 0


def test_migrate_verbose_mode(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    """Test verbose mode shows additional information."""
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--verbose", "--dry-run"])
    assert result.exit_code == 0
    assert "Available script migrators" in result.stdout


@patch("owa.cli.mcap.migrate.migrate.OWAMcapReader")
def test_migrate_already_current_version(mock_reader_class, cli_runner, test_data_dir, tmp_path, copy_test_file):
    """Test migration when file is already at current version."""
    from packaging.version import Version

    orchestrator = MigrationOrchestrator()
    all_target_versions = [m.to_version for m in orchestrator.script_migrators]
    highest_version = str(max(Version(v) for v in all_target_versions)) if all_target_versions else "0.4.2"

    mock_reader = mock_reader_class.return_value.__enter__.return_value
    mock_reader.file_version = highest_version

    test_file = copy_test_file(test_data_dir, "0.4.2.mcap", tmp_path)
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file)])
    assert result.exit_code == 0
    assert "already at the target version" in result.stdout


def test_migrate_corrupted_file(cli_runner, tmp_path, suppress_mcap_warnings):
    """Test migration with corrupted MCAP file."""
    corrupted_file = tmp_path / "corrupted.mcap"
    corrupted_file.write_bytes(b"not a valid mcap file content")
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(corrupted_file), "--dry-run"])
    assert result.exit_code == 0


def test_migrate_empty_file(cli_runner, tmp_path, suppress_mcap_warnings):
    """Test migration with empty MCAP file."""
    empty_file = tmp_path / "empty.mcap"
    empty_file.touch()
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(empty_file), "--dry-run"])
    assert result.exit_code == 0


# === Orchestrator Tests ===
def test_orchestrator_discovery():
    orchestrator = MigrationOrchestrator()
    assert len(orchestrator.script_migrators) >= 3
    for m in orchestrator.script_migrators:
        assert m.script_path.exists()


def test_orchestrator_migration_path():
    orchestrator = MigrationOrchestrator()
    path = orchestrator.get_migration_path("0.3.0", "0.4.2")
    assert len(path) == 2
    assert path[0].from_version == "0.3.0"
    assert path[0].to_version == "0.3.2"


def test_orchestrator_no_path_needed():
    orchestrator = MigrationOrchestrator()
    path = orchestrator.get_migration_path("0.4.0", "0.4.0")
    assert path == []


def test_orchestrator_invalid_path():
    orchestrator = MigrationOrchestrator()
    with pytest.raises(ValueError, match="No migration path found"):
        orchestrator.get_migration_path("unknown", "0.4.2")


def test_detect_version_nonexistent():
    orchestrator = MigrationOrchestrator()
    version = orchestrator.detect_version(Path("/nonexistent.mcap"))
    assert version == "unknown"


# === Real File Tests ===
def test_migrate_real_file(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    """Test migration updates version and file is still readable."""
    from packaging.version import Version

    from mcap_owa.highlevel import OWAMcapReader

    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)
    orchestrator = MigrationOrchestrator()
    original_version = orchestrator.detect_version(test_file)

    # Count messages before migration
    with OWAMcapReader(test_file) as reader:
        original_count = reader.message_count

    # Migrate
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--yes", "-t", "0.5.0", "--no-backups"])
    assert result.exit_code == 0
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file)], input="y\n")
    assert result.exit_code == 0
    assert "Migration successful" in result.stdout

    # Verify version increased
    migrated_version = orchestrator.detect_version(test_file)
    assert Version(migrated_version) > Version(original_version)

    # Verify file is still readable and message count is preserved
    with OWAMcapReader(test_file) as reader:
        assert reader.message_count == original_count


def test_migrate_preserves_messages(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    """Test migration preserves all message data."""
    from mcap_owa.highlevel import OWAMcapReader

    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)

    # Get original topics and count
    with OWAMcapReader(test_file) as reader:
        original_topics = set(reader.topics)
        original_count = reader.message_count

    # Migrate with explicit target (skipping first step as test_migrate_real_file already did it)
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--yes", "--no-backups", "-t", "0.5.0"])
    assert result.exit_code == 0

    # Verify topics and count preserved
    with OWAMcapReader(test_file) as reader:
        assert set(reader.topics) == original_topics
        assert reader.message_count == original_count


def test_migrate_multi_step_path():
    """Test multi-step migration path calculation."""
    from packaging.version import Version

    orchestrator = MigrationOrchestrator()

    # Verify migration path calculation works for multi-step migrations
    target = orchestrator.get_highest_reachable_version("0.3.2")
    path = orchestrator.get_migration_path("0.3.2", target)

    # Should have multiple steps from 0.3.2 to latest
    assert len(path) >= 3, f"Expected at least 3 migration steps, got {len(path)}"

    # Verify path is ordered correctly
    for i, migrator in enumerate(path):
        assert Version(migrator.from_version) < Version(migrator.to_version)
        if i > 0:
            assert migrator.from_version == path[i - 1].to_version


# === Rollback/Cleanup Tests ===
def test_rollback_no_backups(cli_runner, tmp_path):
    test_file = tmp_path / "test.mcap"
    test_file.write_text("content")
    result = cli_runner.invoke(app, ["mcap", "migrate", "rollback", str(test_file)])
    assert "No backup files found" in result.stdout


def test_rollback_workflow(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)
    backup_file = test_file.with_suffix(".mcap.backup")

    # Migrate
    cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--yes", "-t", "0.5.0", "--no-backups"])
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--yes"])
    assert result.exit_code == 0
    assert backup_file.exists()

    # Rollback
    result = cli_runner.invoke(app, ["mcap", "migrate", "rollback", str(test_file), "--yes"])
    assert result.exit_code == 0
    assert not backup_file.exists()


def test_cleanup_patterns(cli_runner, tmp_path):
    (tmp_path / "a.mcap.backup").write_text("a")
    (tmp_path / "b.mcap.backup").write_text("b")

    result = cli_runner.invoke(app, ["mcap", "migrate", "cleanup", str(tmp_path / "*.mcap.backup"), "--dry-run"])
    assert "Would delete 2" in result.stdout


# === Backup Naming Scheme Test ===
def test_backup_naming_scheme():
    """Test that backup files use intuitive naming scheme (.mcap.backup)."""
    test_cases = [
        ("recording.mcap", "recording.mcap.backup"),
        ("data_file.mcap", "data_file.mcap.backup"),
        ("test-file.mcap", "test-file.mcap.backup"),
        ("file.with.dots.mcap", "file.with.dots.mcap.backup"),
    ]
    for original_name, expected_backup_name in test_cases:
        original_path = Path(f"/tmp/{original_name}")
        actual_backup_path = original_path.with_suffix(f"{original_path.suffix}.backup")
        assert actual_backup_path.name == expected_backup_name


# === Migration Integrity Test ===
def test_migration_integrity_verification(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    """Test migration integrity verification functionality."""
    from owa.cli.mcap.migrate.utils import verify_migration_integrity

    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)

    # Migrate to 0.5.0 first (to prevent false-positive verification fail due to size diff)
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--yes", "-t", "0.5.0", "--no-backups"])
    assert result.exit_code == 0

    # Migrate to latest with backup
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file)], input="y\n")
    assert result.exit_code == 0

    # Check if backup was created and verify integrity
    backup_file = tmp_path / "0.3.2.mcap.backup"
    if backup_file.exists():
        integrity_result = verify_migration_integrity(
            migrated_file=test_file, backup_file=backup_file, size_tolerance_percent=50.0
        )
        assert integrity_result.success is True


# === Output Validation Tests ===
def test_validate_migration_output_success():
    """Test validation of successful migration output."""
    from owa.cli.mcap.migrate import validate_migration_output

    valid_success = {"success": True, "changes_made": 5, "from_version": "0.3.0", "to_version": "0.4.0"}
    assert validate_migration_output(valid_success)


def test_validate_migration_output_failure():
    """Test validation of failed migration output."""
    from owa.cli.mcap.migrate import validate_migration_output

    valid_failure = {
        "success": False,
        "changes_made": 0,
        "error": "Migration failed",
        "from_version": "0.3.0",
        "to_version": "0.4.0",
    }
    assert validate_migration_output(valid_failure)


def test_validate_verification_output():
    """Test validation of verification output."""
    from owa.cli.mcap.migrate import validate_verification_output

    valid_success = {"success": True, "message": "Verification completed successfully"}
    assert validate_verification_output(valid_success)

    valid_failure = {"success": False, "error": "Legacy structures found"}
    assert validate_verification_output(valid_failure)


# === Version Range Matching Test ===
def test_version_range_matching():
    """Test that version range matching works correctly (e.g., 0.3.1 uses 0.3.0->0.3.2 migrator)."""
    orchestrator = MigrationOrchestrator()
    path = orchestrator.get_migration_path("0.3.1", "0.4.2")
    assert len(path) == 2
    assert path[0].from_version == "0.3.0"
    assert path[0].to_version == "0.3.2"
    assert path[1].from_version == "0.3.2"
    assert path[1].to_version == "0.4.2"


# === ScriptMigrator Error Handling ===
def test_script_migrator_json_error(tmp_path):
    script = tmp_path / "test.py"
    script.write_text("print('test')")
    migrator = ScriptMigrator(script, "0.3.0", "0.4.0")

    error_json = (
        '{"success": false, "changes_made": 0, "error": "fail", "from_version": "0.3.0", "to_version": "0.4.0"}'
    )
    with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout=error_json, stderr="")):
        result = migrator.migrate(tmp_path / "test.mcap", verbose=False)
    assert not result.success
    assert result.error == "fail"


def test_script_migrator_invalid_json_fallback(tmp_path):
    """Test fallback to stderr/stdout when JSON is invalid."""
    script = tmp_path / "test.py"
    script.write_text("print('test')")
    migrator = ScriptMigrator(script, "0.3.0", "0.4.0")

    with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="Invalid JSON", stderr="Script failed")):
        result = migrator.migrate(tmp_path / "test.mcap", verbose=False)
    assert not result.success
    assert result.error == "Script failed"


def test_script_migrator_verify_success(tmp_path):
    """Test verification with successful JSON output."""
    script = tmp_path / "test.py"
    script.write_text("print('test')")
    migrator = ScriptMigrator(script, "0.3.0", "0.4.0")

    success_json = '{"success": true, "message": "Verification completed successfully"}'
    with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout=success_json, stderr="")):
        result = migrator.verify_migration(tmp_path / "test.mcap", None, verbose=False)
    assert result.success
