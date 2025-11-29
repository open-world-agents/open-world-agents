"""Tests for owl mcap migrate CLI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from owa.cli import app
from owa.cli.mcap.migrate import MigrationOrchestrator, MigrationResult, ScriptMigrator


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
    test_file = copy_test_file(test_data_dir, "0.3.2.mcap", tmp_path)
    orchestrator = MigrationOrchestrator()
    original = orchestrator.detect_version(test_file)

    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file), "--yes", "-t", "0.5.0", "--no-backups"])
    assert result.exit_code == 0
    result = cli_runner.invoke(app, ["mcap", "migrate", "run", str(test_file)], input="y\n")
    assert result.exit_code == 0
    assert "Migration successful" in result.stdout

    from packaging.version import Version

    migrated = orchestrator.detect_version(test_file)
    assert Version(migrated) > Version(original)


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
