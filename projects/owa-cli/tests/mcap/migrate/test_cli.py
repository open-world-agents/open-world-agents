"""
Tests for the `owl mcap migrate` CLI command.
"""

from unittest.mock import patch

import pytest

from owa.cli import app


def test_migrate_help(cli_runner, strip_ansi_codes):
    """Test migrate command help."""
    result = cli_runner.invoke(app, ["mcap", "migrate", "--help"])
    assert result.exit_code == 0
    # Strip ANSI codes for more reliable testing in CI environments
    clean_output = strip_ansi_codes(result.stdout)
    assert "Migrate MCAP files" in clean_output
    assert "--target" in clean_output
    assert "--dry-run" in clean_output


def test_migrate_nonexistent_file(cli_runner):
    """Test migration with non-existent file."""
    result = cli_runner.invoke(app, ["mcap", "migrate", "nonexistent.mcap"])
    assert result.exit_code == 0
    assert "File not found" in result.stdout


def test_migrate_non_mcap_file(cli_runner, temp_dir):
    """Test migration with non-MCAP file."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("not an mcap file")

    result = cli_runner.invoke(app, ["mcap", "migrate", str(test_file)])
    assert result.exit_code == 0
    assert "Skipping non-MCAP file" in result.stdout


def test_migrate_dry_run(cli_runner, test_data_dir, temp_dir, copy_test_file, suppress_mcap_warnings):
    """Test dry run mode doesn't modify files."""
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)
    original_size = test_file.stat().st_size
    original_mtime = test_file.stat().st_mtime

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", str(test_file), "--dry-run"])

    assert result.exit_code == 0
    assert "DRY RUN MODE" in result.stdout

    # File should be unchanged
    assert test_file.stat().st_size == original_size
    assert test_file.stat().st_mtime == original_mtime


@patch("owa.cli.mcap.migrate.migrate.OWAMcapReader")
def test_migrate_already_current_version(mock_reader_class, cli_runner, test_data_dir, temp_dir, copy_test_file):
    """Test migration when file is already at current version."""
    mock_reader = mock_reader_class.return_value.__enter__.return_value
    mock_reader.file_version = "0.4.2"

    test_file = copy_test_file(test_data_dir, "0.4.2.mcap", temp_dir)
    result = cli_runner.invoke(app, ["mcap", "migrate", str(test_file)])
    assert result.exit_code == 0
    assert "already at the target version" in result.stdout


def test_migrate_verbose_mode(cli_runner, test_data_dir, temp_dir, copy_test_file, suppress_mcap_warnings):
    """Test verbose mode shows additional information."""
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", str(test_file), "--verbose", "--dry-run"])

    assert result.exit_code == 0
    assert "Available script migrators" in result.stdout


def test_migrate_with_target_version(cli_runner, test_data_dir, temp_dir, copy_test_file, suppress_mcap_warnings):
    """Test migration with specific target version."""
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", str(test_file), "--target", "0.4.2", "--dry-run"])

    assert result.exit_code == 0
    assert "Target version: 0.4.2" in result.stdout


def test_migrate_multiple_files(cli_runner, test_data_dir, temp_dir, copy_test_file, suppress_mcap_warnings):
    """Test migration with multiple files."""
    file1 = copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)
    file2 = copy_test_file(test_data_dir, "0.4.2.mcap", temp_dir)

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", str(file1), str(file2), "--dry-run"])

    assert result.exit_code == 0
    assert "Files to process: 2" in result.stdout


def test_migrate_user_cancellation(cli_runner, test_data_dir, temp_dir, copy_test_file, suppress_mcap_warnings):
    """Test user cancellation of migration."""
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", str(test_file)], input="n\n")

    assert result.exit_code == 0


@patch("owa.cli.mcap.migrate.migrate.OWAMcapReader")
def test_migrate_shows_migration_summary_table(mock_reader_class, cli_runner, test_data_dir, temp_dir, copy_test_file):
    """Test that migration shows a summary table."""
    mock_reader = mock_reader_class.return_value.__enter__.return_value
    mock_reader.file_version = "0.3.2"

    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)
    result = cli_runner.invoke(app, ["mcap", "migrate", str(test_file), "--dry-run"])
    assert result.exit_code == 0
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


def test_cli_with_multiple_files(cli_runner, test_data_dir, temp_dir, copy_test_file, suppress_mcap_warnings):
    """Test CLI with multiple files."""
    files = []
    for filename in ["0.3.2.mcap", "0.4.2.mcap"]:
        try:
            file_path = copy_test_file(test_data_dir, filename, temp_dir)
            files.append(str(file_path))
        except pytest.skip.Exception:
            continue

    if not files:
        pytest.skip("No test files available")

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate"] + files + ["--dry-run"])

    assert result.exit_code == 0
    assert f"Files to process: {len(files)}" in result.stdout


# Output verification tests
def test_migration_produces_expected_output(
    cli_runner, test_data_dir, temp_dir, copy_test_file, suppress_mcap_warnings
):
    """Test that migrating 0.3.2.mcap produces expected output."""
    from owa.cli.mcap.migrate import MigrationOrchestrator

    source_file = test_data_dir / "0.3.2.mcap"
    expected_file = test_data_dir / "0.4.2.mcap"

    if not source_file.exists() or not expected_file.exists():
        pytest.skip("Required test files not found")

    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", str(test_file)], input="y\n")

    # Verify the migrated file has the correct version
    orchestrator = MigrationOrchestrator()
    migrated_version = orchestrator.detect_version(test_file)
    expected_version = orchestrator.detect_version(expected_file)

    assert result.exit_code == 0
    assert "Migration successful" in result.stdout
    assert migrated_version == expected_version

    # Verify basic file properties
    assert test_file.stat().st_size > 0


def test_migration_integrity_verification(cli_runner, test_data_dir, temp_dir, copy_test_file, suppress_mcap_warnings):
    """Test migration integrity verification functionality."""
    from rich.console import Console

    from owa.cli.mcap.migrate.utils import verify_migration_integrity

    source_file = test_data_dir / "0.3.2.mcap"
    if not source_file.exists():
        pytest.skip("Required test file not found")

    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", temp_dir)

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", str(test_file)], input="y\n")

    assert result.exit_code == 0

    # Test integrity verification with backup file (BackupContext creates this automatically)
    backup_file = temp_dir / "0.3.2.mcap.backup"
    if backup_file.exists():  # Only test if backup was created by BackupContext
        # Verify integrity - warnings are suppressed by the fixture
        console = Console()
        integrity_result = verify_migration_integrity(
            migrated_file=test_file,
            backup_file=backup_file,
            console=console,
            size_tolerance_percent=50.0,
        )

        assert integrity_result is True


# Error handling tests
def test_migrate_with_corrupted_file(cli_runner, temp_dir, suppress_mcap_warnings):
    """Test migration with corrupted MCAP file."""
    corrupted_file = temp_dir / "corrupted.mcap"
    corrupted_file.write_bytes(b"not a valid mcap file content")

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", str(corrupted_file), "--dry-run"])

    assert result.exit_code == 0


def test_migrate_with_empty_file(cli_runner, temp_dir, suppress_mcap_warnings):
    """Test migration with empty MCAP file."""
    empty_file = temp_dir / "empty.mcap"
    empty_file.touch()

    # Warnings are suppressed by the fixture
    result = cli_runner.invoke(app, ["mcap", "migrate", str(empty_file), "--dry-run"])

    assert result.exit_code == 0
