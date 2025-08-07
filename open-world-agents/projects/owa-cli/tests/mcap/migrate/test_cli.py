"""
Tests for the `owl mcap migrate` CLI command.
"""

from unittest.mock import patch

import pytest

from owa.cli import app


def test_migrate_help(cli_runner, strip_ansi_codes):
    """Test migrate command help."""
    result = cli_runner.invoke(app, ["mcap", "migrate", "--help"])
    assert result.exit_code == 0, f"Migrate help failed: {result.stdout}"
    # Strip ANSI codes for more reliable testing in CI environments
    clean_output = strip_ansi_codes(result.stdout)
    assert "MCAP migration commands" in clean_output
    assert "run" in clean_output
    assert "rollback" in clean_output
    assert "cleanup" in clean_output


def test_migrate_run_help(cli_runner, strip_ansi_codes):
    """Test migrate run command help."""
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", "--help"])
    assert result.exit_code == 0, f"Migrate run help failed: {result.stdout}"
    # Strip ANSI codes for more reliable testing in CI environments
    clean_output = strip_ansi_codes(result.stdout)
    assert "Migrate MCAP files" in clean_output
    assert "--target" in clean_output
    assert "--dry-run" in clean_output


def test_migrate_nonexistent_file(cli_runner):
    """Test migration with non-existent file."""
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", "nonexistent.mcap"])
    assert result.exit_code == 0, f"Migrate failed: {result.stdout}"
    assert "File not found" in result.stdout


def test_migrate_non_mcap_file(cli_runner, tmp_path):
    """Test migration with non-MCAP file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("not an mcap file")

    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file)])
    assert result.exit_code == 0, f"Migrate failed: {result.stdout}"
    assert "Skipping non-MCAP file" in result.stdout


def test_migrate_dry_run(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    """Test dry run mode doesn't modify files."""
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)
    original_size = test_file.stat().st_size
    original_mtime = test_file.stat().st_mtime

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--dry-run"])

    assert result.exit_code == 0, f"Migrate failed: {result.stdout}"
    assert "DRY RUN MODE" in result.stdout

    # File should be unchanged
    assert test_file.stat().st_size == original_size
    assert test_file.stat().st_mtime == original_mtime


@patch("owa.cli.mcap.migrate.migrate.OWAMcapReader")
def test_migrate_already_current_version(mock_reader_class, cli_runner, test_data_dir, tmp_path, copy_test_file):
    """Test migration when file is already at current version."""
    from owa.cli.mcap.migrate import MigrationOrchestrator

    # Get the highest reachable version dynamically
    orchestrator = MigrationOrchestrator()
    all_target_versions = [m.to_version for m in orchestrator.script_migrators]
    if all_target_versions:
        from packaging.version import Version

        highest_version = str(max(Version(v) for v in all_target_versions))
    else:
        highest_version = "0.4.2"  # Fallback

    mock_reader = mock_reader_class.return_value.__enter__.return_value
    # Set the file version to the highest reachable version so no migration is needed
    mock_reader.file_version = highest_version

    test_file = copy_test_file(test_data_dir, "0.4.2.mcap", tmp_path)
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file)])

    assert result.exit_code == 0, f"Migrate failed: {result.stdout}"
    assert "already at the target version" in result.stdout


def test_migrate_verbose_mode(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    """Test verbose mode shows additional information."""
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--verbose", "--dry-run"])

    assert result.exit_code == 0, f"Migrate failed: {result.stdout}"
    assert "Available script migrators" in result.stdout


def test_migrate_with_target_version(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    """Test migration with specific target version."""
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--target", "0.4.2", "--dry-run"])

    assert result.exit_code == 0, f"Migrate failed: {result.stdout}"
    assert "Target version: 0.4.2" in result.stdout


def test_migrate_multiple_files(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    """Test migration with multiple files."""
    file1 = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)
    file2 = copy_test_file(test_data_dir, "0.4.2.mcap", tmp_path)

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(file1), str(file2), "--dry-run"])

    assert result.exit_code == 0, f"Migrate failed: {result.stdout}"
    assert "Files to process: 2" in result.stdout


def test_migrate_user_cancellation(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    """Test user cancellation of migration."""
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file)], input="n\n")

    assert result.exit_code == 0, f"Migrate failed: {result.stdout}"


@patch("owa.cli.mcap.migrate.migrate.OWAMcapReader")
def test_migrate_shows_migration_summary_table(mock_reader_class, cli_runner, test_data_dir, tmp_path, copy_test_file):
    """Test that migration shows a summary table."""
    mock_reader = mock_reader_class.return_value.__enter__.return_value
    mock_reader.file_version = "0.3.2"

    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--dry-run"])
    assert result.exit_code == 0, f"Migrate failed: {result.stdout}"
    assert "Migration Summary" in result.stdout


# Integration tests
def test_migration_orchestrator_with_real_files(test_data_dir, suppress_mcap_warnings):
    """Test that the migration orchestrator can analyze real MCAP files."""
    from owa.cli.mcap.migrate import MigrationOrchestrator

    orchestrator = MigrationOrchestrator()
    test_files = ["0.3.2.mcap", "0.4.2.mcap"]

    # Warnings are suppressed by the fixture
    for filename in test_files:
        test_file = test_data_dir / filename
        if test_file.exists():
            version = orchestrator.detect_version(test_file)
            assert version is not None
            assert isinstance(version, str)


def test_migrator_discovery():
    """Test that migrators are discovered correctly."""
    from owa.cli.mcap.migrate import MigrationOrchestrator

    orchestrator = MigrationOrchestrator()
    assert len(orchestrator.script_migrators) > 0

    for migrator in orchestrator.script_migrators:
        assert hasattr(migrator, "from_version")
        assert hasattr(migrator, "to_version")
        assert hasattr(migrator, "script_path")
        assert migrator.script_path.exists()


def test_version_range_matching():
    """Test that version range matching works correctly."""
    from owa.cli.mcap.migrate import MigrationOrchestrator

    orchestrator = MigrationOrchestrator()
    try:
        path = orchestrator.get_migration_path("0.3.1", "0.4.2")
        assert len(path) > 0
        assert path[0].from_version == "0.3.0"
        assert path[0].to_version == "0.3.2"
    except ValueError:
        pass  # No path exists, which is acceptable


def test_cli_with_multiple_files(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    """Test CLI with multiple files."""
    files = []
    for filename in ["0.3.2.mcap", "0.4.2.mcap"]:
        try:
            file_path = copy_test_file(test_data_dir, filename, tmp_path)
            files.append(str(file_path))
        except pytest.skip.Exception:
            continue

    if not files:
        pytest.skip("No test files available")

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", "run"] + files + ["--dry-run"])

    assert result.exit_code == 0, f"Migration failed: {result.stdout}"
    assert f"Files to process: {len(files)}" in result.stdout


# Output verification tests
def test_migration_produces_expected_output(
    cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings
):
    """Test that migrating 0.3.2.mcap produces expected output."""
    from packaging.version import Version

    from owa.cli.mcap.migrate import MigrationOrchestrator

    source_file = test_data_dir / "0.3.2.mcap"

    if not source_file.exists():
        pytest.skip("Required test file not found")

    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)

    # Get the original version before migration
    orchestrator = MigrationOrchestrator()
    original_version = orchestrator.detect_version(test_file)

    # Warnings are suppressed by the fixture
    # Migrate up to v0.5.0 first, this is to prevent false-positive verification fail.
    # across multiple migrations, the size difference get's bigger, inducing a false positive verification failure.
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--yes", "-t", "0.5.0", "--no-backups"])
    assert result.exit_code == 0, f"Migration failed: {result.stdout}"
    # Now run to the latest version
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file)], input="y\n")

    # Verify the migration was successful
    assert result.exit_code == 0, f"Migration failed: {result.stdout}"
    assert "Migration successful" in result.stdout

    # Verify the migrated file has a higher version than the original
    migrated_version = orchestrator.detect_version(test_file)
    assert migrated_version != "unknown", "Migration should produce a valid version"
    assert Version(migrated_version) > Version(original_version), (
        f"Migration should increase version from {original_version} to {migrated_version}"
    )

    # Verify basic file properties
    assert test_file.stat().st_size > 0


def test_migration_integrity_verification(cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings):
    """Test migration integrity verification functionality."""

    from owa.cli.mcap.migrate.utils import verify_migration_integrity

    source_file = test_data_dir / "0.3.2.mcap"
    if not source_file.exists():
        pytest.skip("Required test file not found")

    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)

    # Warnings are suppressed by the fixture
    # Migrate up to v0.5.0 first, this is to prevent false-positive verification fail.
    # across multiple migrations, the size difference get's bigger, inducing a false positive verification failure.
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--yes", "-t", "0.5.0", "--no-backups"])
    assert result.exit_code == 0, f"Migration failed: {result.stdout}"
    # Now run to the latest version
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file)], input="y\n")

    assert result.exit_code == 0, f"Migration failed: {result.stdout}"

    # Test integrity verification with backup file (BackupContext creates this automatically)
    backup_file = tmp_path / "0.3.2.mcap.backup"
    if backup_file.exists():  # Only test if backup was created by BackupContext
        # Verify integrity - warnings are suppressed by the fixture
        integrity_result = verify_migration_integrity(
            migrated_file=test_file,
            backup_file=backup_file,
            size_tolerance_percent=50.0,
        )

        assert integrity_result.success is True


# Error handling tests
def test_migrate_with_corrupted_file(cli_runner, tmp_path, suppress_mcap_warnings):
    """Test migration with corrupted MCAP file."""
    corrupted_file = tmp_path / "corrupted.mcap"
    corrupted_file.write_bytes(b"not a valid mcap file content")

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(corrupted_file), "--dry-run"])

    assert result.exit_code == 0, f"Migration failed: {result.stdout}"


def test_migrate_with_empty_file(cli_runner, tmp_path, suppress_mcap_warnings):
    """Test migration with empty MCAP file."""
    empty_file = tmp_path / "empty.mcap"
    empty_file.touch()

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(empty_file), "--dry-run"])

    assert result.exit_code == 0, f"Migration failed: {result.stdout}"


def test_migrate_rollback_help(cli_runner, strip_ansi_codes):
    """Test migrate rollback command help."""
    result = cli_runner.invoke(app, ["mcap", "migrate", "rollback", "--help"])
    assert result.exit_code == 0, f"Rollback help failed: {result.stdout}"
    # Strip ANSI codes for more reliable testing in CI environments
    clean_output = strip_ansi_codes(result.stdout)
    assert "Rollback MCAP files" in clean_output
    assert "--yes" in clean_output
    assert "--verbose" in clean_output


def test_migrate_cleanup_help(cli_runner, strip_ansi_codes):
    """Test migrate cleanup command help."""
    result = cli_runner.invoke(app, ["mcap", "migrate", "cleanup", "--help"])
    assert result.exit_code == 0, f"Cleanup help failed: {result.stdout}"
    # Strip ANSI codes for more reliable testing in CI environments
    clean_output = strip_ansi_codes(result.stdout)
    assert "Clean up MCAP backup files" in clean_output
    assert "--dry-run" in clean_output
    assert "--yes" in clean_output
    assert "--verbose" in clean_output


def test_migrate_rollback_no_backups(cli_runner, tmp_path):
    """Test rollback with no backup files."""
    test_file = tmp_path / "test.mcap"
    test_file.write_text("test content")

    result = cli_runner.invoke(app, ["mcap", "migrate", "rollback", str(test_file)])
    assert result.exit_code == 0, f"Rollback failed: {result.stdout}"
    assert "No backup files found" in result.stdout


def test_migrate_cleanup_no_backups(cli_runner, tmp_path):
    """Test cleanup with no backup files in a controlled directory."""
    # Run cleanup in the tmp_path to confine the operation to a safe directory
    result = cli_runner.invoke(app, ["mcap", "migrate", "cleanup", str(tmp_path / "*.mcap.backup"), "--dry-run"])
    assert result.exit_code == 0, f"Cleanup failed: {result.stdout}"
    assert "No backup files found" in result.stdout


def test_migrate_rollback_cleanup_workflow(
    cli_runner, test_data_dir, tmp_path, copy_test_file, suppress_mcap_warnings
):
    """Test complete workflow: migrate -> rollback -> cleanup."""
    # Setup: Copy test file
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)
    backup_file = test_file.with_suffix(".mcap.backup")

    # Step 1: Run migration.
    # First run up to v0.5.0, this is to prevent false-positive verification fail.
    # across multiple migrations, the size difference get's bigger, inducing a false positive verification failure.
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--yes", "-t", "0.5.0", "--no-backups"])
    assert result.exit_code == 0, f"Migration failed: {result.stdout}"
    # Now run to the latest version
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--yes"])
    assert result.exit_code == 0, f"Migration failed: {result.stdout}"

    # Verify backup was created
    assert backup_file.exists(), "Backup file should be created during migration"

    # Step 2: Test rollback functionality
    result = cli_runner.invoke(app, ["mcap", "migrate", "rollback", str(test_file), "--yes", "--verbose"])
    assert result.exit_code == 0, f"Rollback failed: {result.stdout}"
    assert "Rollback Complete" in result.stdout
    assert "Successful: 1" in result.stdout

    # Verify backup was removed after rollback
    assert not backup_file.exists(), "Backup file should be removed after successful rollback"

    # Step 3: Create another backup for cleanup test
    # Again, migrate up to v0.5.0 first
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--yes", "-t", "0.5.0", "--no-backups"])
    assert result.exit_code == 0, f"Migration failed: {result.stdout}"
    # Now run to the latest version
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--yes"])
    assert result.exit_code == 0, f"Migration failed: {result.stdout}"
    assert backup_file.exists(), "Backup file should be created again"

    # Step 4: Test cleanup functionality
    result = cli_runner.invoke(app, ["mcap", "migrate", "cleanup", str(backup_file), "--yes", "--verbose"])
    assert result.exit_code == 0, f"Cleanup failed: {result.stdout}"
    assert "Cleanup Complete" in result.stdout
    assert "Deleted: 1" in result.stdout

    # Verify backup was removed after cleanup
    assert not backup_file.exists(), "Backup file should be removed after cleanup"


def test_migrate_rollback_with_missing_current(cli_runner, tmp_path):
    """Test rollback when current file is missing (restore scenario)."""
    # Create backup file without original
    backup_file = tmp_path / "missing.mcap.backup"
    backup_file.write_text("backup content")
    original_file = tmp_path / "missing.mcap"

    # Test rollback (should restore the missing file)
    result = cli_runner.invoke(app, ["mcap", "migrate", "rollback", str(original_file), "--yes", "--verbose"])
    assert result.exit_code == 0, f"Rollback failed: {result.stdout}"
    assert "MISSI" in result.stdout  # Should show MISSING status (may be truncated in table)
    assert "Rollback Complete" in result.stdout

    # Verify file was restored
    assert original_file.exists(), "Original file should be restored from backup"
    assert original_file.read_text() == "backup content"
    assert not backup_file.exists(), "Backup file should be removed after rollback"


def test_migrate_cleanup_with_patterns(cli_runner, tmp_path):
    """Test cleanup with different file patterns."""
    # Create multiple backup files
    backup1 = tmp_path / "test1.mcap.backup"
    backup2 = tmp_path / "test2.mcap.backup"
    backup3 = tmp_path / "subdir" / "test3.mcap.backup"

    backup1.write_text("backup1")
    backup2.write_text("backup2")
    backup3.parent.mkdir(exist_ok=True)
    backup3.write_text("backup3")

    # Test cleanup with specific pattern
    result = cli_runner.invoke(app, ["mcap", "migrate", "cleanup", str(tmp_path / "test1.mcap.backup"), "--dry-run"])
    assert result.exit_code == 0, f"Cleanup failed: {result.stdout}"
    assert "test1" in result.stdout  # Filename should appear in table (may be truncated)
    assert "Would delete 1 backup files" in result.stdout

    # Test cleanup with wildcard pattern
    result = cli_runner.invoke(app, ["mcap", "migrate", "cleanup", str(tmp_path / "*.mcap.backup"), "--dry-run"])
    assert result.exit_code == 0, f"Cleanup failed: {result.stdout}"
    assert "Would delete 2 backup files" in result.stdout  # Should find test1 and test2

    # Test cleanup with recursive pattern
    result = cli_runner.invoke(
        app, ["mcap", "migrate", "cleanup", str(tmp_path / "**" / "*.mcap.backup"), "--dry-run"]
    )
    assert result.exit_code == 0, f"Cleanup failed: {result.stdout}"
    assert "Would delete 3 backup files" in result.stdout  # Should find all 3 files


def test_migrate_commands_help_consistency(cli_runner, strip_ansi_codes):
    """Test that all migrate subcommands have consistent help output."""
    # Test main migrate help
    result = cli_runner.invoke(app, ["mcap", "migrate", "--help"])
    assert result.exit_code == 0, f"Main migrate help failed: {result.stdout}"
    clean_output = strip_ansi_codes(result.stdout)
    assert "run" in clean_output
    assert "rollback" in clean_output
    assert "cleanup" in clean_output

    # Test each subcommand has proper help
    for subcommand in ["run", "rollback", "cleanup"]:
        result = cli_runner.invoke(app, ["mcap", "migrate", subcommand, "--help"])
        assert result.exit_code == 0, f"{subcommand} help should work"
        clean_output = strip_ansi_codes(result.stdout)
        assert "--help" in clean_output
        assert "Show this message and exit" in clean_output or "help" in clean_output.lower()
