"""
Tests for the MCAP migration system.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from owa.cli.mcap.migrate import (
    MigrationOrchestrator,
    MigrationResult,
    detect_files_needing_migration,
)


class TestMigrationOrchestrator:
    """Test the migration orchestrator."""

    def test_init(self):
        """Test orchestrator initialization."""
        orchestrator = MigrationOrchestrator()
        # Should discover script migrators
        assert len(orchestrator.script_migrators) >= 0  # May be 0 if scripts not found
        # Current version should match the actual mcap-owa-support library version
        from mcap_owa import __version__ as mcap_owa_version

        assert orchestrator.current_version == mcap_owa_version

    def test_detect_version_fallback_to_unknown(self):
        """Test version detection returns 'unknown' for non-existent file."""
        orchestrator = MigrationOrchestrator()
        non_existent_file = Path("/non/existent/file.mcap")
        version = orchestrator.detect_version(non_existent_file)
        assert version == "unknown"

    def test_create_backup_success(self):
        """Test successful backup creation."""
        orchestrator = MigrationOrchestrator()

        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            source_path = Path(tmp_file.name)

        try:
            backup_path = source_path.with_suffix(".backup.mcap")
            # Should not raise any exception
            orchestrator.create_backup(source_path, backup_path)

            assert backup_path.exists()
            assert backup_path.stat().st_size == source_path.stat().st_size

        finally:
            # Cleanup
            if source_path.exists():
                source_path.unlink()
            if backup_path.exists():
                backup_path.unlink()

    def test_create_backup_nonexistent_source(self):
        """Test backup creation with non-existent source file."""
        orchestrator = MigrationOrchestrator()

        source_path = Path("/non/existent/file.mcap")
        backup_path = Path("/tmp/backup.mcap")

        with pytest.raises(FileNotFoundError, match="Source file not found"):
            orchestrator.create_backup(source_path, backup_path)

    def test_create_backup_existing_backup(self):
        """Test backup creation when backup file already exists."""
        orchestrator = MigrationOrchestrator()

        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            source_path = Path(tmp_file.name)

        with tempfile.NamedTemporaryFile(suffix=".backup.mcap", delete=False) as backup_file:
            backup_path = Path(backup_file.name)

        try:
            with pytest.raises(FileExistsError, match="Backup file already exists"):
                orchestrator.create_backup(source_path, backup_path)

        finally:
            # Cleanup
            if source_path.exists():
                source_path.unlink()
            if backup_path.exists():
                backup_path.unlink()

    def test_backup_naming_scheme(self):
        """Test that backup files use intuitive naming scheme."""
        # Test with different file names and extensions
        test_cases = [
            ("recording.mcap", "recording.mcap.backup"),
            ("data_file.mcap", "data_file.mcap.backup"),
            ("test-file.mcap", "test-file.mcap.backup"),
            ("file.with.dots.mcap", "file.with.dots.mcap.backup"),
        ]

        for original_name, expected_backup_name in test_cases:
            original_path = Path(f"/tmp/{original_name}")
            expected_backup_path = Path(f"/tmp/{expected_backup_name}")

            # The backup path should be generated as: original.extension.backup
            actual_backup_path = original_path.with_suffix(f"{original_path.suffix}.backup")

            assert actual_backup_path == expected_backup_path
            assert actual_backup_path.name == expected_backup_name

    def test_get_migration_path_no_migration_needed(self):
        """Test migration path when no migration is needed."""
        orchestrator = MigrationOrchestrator()
        path = orchestrator.get_migration_path("0.4.0", "0.4.0")
        assert path == []

    def test_get_migration_path_sequential(self):
        """Test sequential migration path."""
        orchestrator = MigrationOrchestrator()
        path = orchestrator.get_migration_path("0.3.0", "0.4.1")
        assert len(path) == 2
        assert path[0].from_version == "0.3.0"
        assert path[0].to_version == "0.3.2"
        assert path[1].from_version == "0.3.2"
        assert path[1].to_version == "0.4.1"

    def test_get_migration_path_from_v032(self):
        """Test migration path from v0.3.2 to v0.4.1."""
        orchestrator = MigrationOrchestrator()
        path = orchestrator.get_migration_path("0.3.2", "0.4.1")
        assert len(path) == 1
        assert path[0].from_version == "0.3.2"
        assert path[0].to_version == "0.4.1"

    def test_get_migration_path_invalid(self):
        """Test invalid migration path."""
        orchestrator = MigrationOrchestrator()
        with pytest.raises(ValueError, match="No migration path found"):
            orchestrator.get_migration_path("unknown", "0.4.1")

    def test_get_migration_path_version_ranges(self):
        """Test migration path with version ranges - versions between migrator ranges."""
        orchestrator = MigrationOrchestrator()

        # Test version 0.3.1 (between 0.3.0 and 0.3.2) should use 0.3.0->0.3.2 migrator
        path = orchestrator.get_migration_path("0.3.1", "0.4.1")
        assert len(path) == 2
        assert path[0].from_version == "0.3.0"
        assert path[0].to_version == "0.3.2"
        assert path[1].from_version == "0.3.2"
        assert path[1].to_version == "0.4.1"

    def test_get_migration_path_version_ranges_single_step(self):
        """Test migration path with version ranges - single step."""
        orchestrator = MigrationOrchestrator()

        # Test version 0.3.1 (between 0.3.0 and 0.3.2) should use 0.3.0->0.3.2 migrator
        path = orchestrator.get_migration_path("0.3.1", "0.3.2")
        assert len(path) == 1
        assert path[0].from_version == "0.3.0"
        assert path[0].to_version == "0.3.2"

    def test_get_migration_path_version_ranges_second_range(self):
        """Test migration path with version ranges - version in second range."""
        orchestrator = MigrationOrchestrator()

        # Test version 0.3.5 (between 0.3.2 and 0.4.1) should use 0.3.2->0.4.1 migrator
        path = orchestrator.get_migration_path("0.3.5", "0.4.1")
        assert len(path) == 1
        assert path[0].from_version == "0.3.2"
        assert path[0].to_version == "0.4.1"

    def test_get_highest_reachable_version_ranges(self):
        """Test highest reachable version with version ranges."""
        orchestrator = MigrationOrchestrator()

        # Test version 0.3.1 (between 0.3.0 and 0.3.2) should reach 0.4.1
        highest = orchestrator.get_highest_reachable_version("0.3.1")
        assert highest == "0.4.1"

        # Test version 0.3.5 (between 0.3.2 and 0.4.1) should reach 0.4.1
        highest = orchestrator.get_highest_reachable_version("0.3.5")
        assert highest == "0.4.1"

    def test_automatic_migrator_discovery(self):
        """Test that migrators are automatically discovered from filenames."""
        orchestrator = MigrationOrchestrator()

        # Should have discovered the existing migrators
        assert len(orchestrator.script_migrators) == 2

        # Check that the versions were parsed correctly from filenames
        migrator_versions = [(m.from_version, m.to_version) for m in orchestrator.script_migrators]
        expected_versions = [("0.3.0", "0.3.2"), ("0.3.2", "0.4.1")]

        # Sort both lists to ensure consistent comparison
        migrator_versions.sort()
        expected_versions.sort()

        assert migrator_versions == expected_versions

    def test_filename_pattern_parsing(self):
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


class TestDetectFilesNeedingMigration:
    """Test file detection functionality."""

    def test_detect_files_no_files(self):
        """Test detection when no files are provided."""
        console = MagicMock()
        result = detect_files_needing_migration([], console, False, None)
        assert result == []
        console.print.assert_called_with("[yellow]No valid MCAP files found[/yellow]")

    def test_detect_files_with_non_existent_file(self):
        """Test detection with non-existent file."""
        console = MagicMock()
        non_existent_file = Path("/non/existent/file.mcap")

        result = detect_files_needing_migration([non_existent_file], console, False, None)

        assert result == []
        console.print.assert_any_call(f"[red]File not found: {non_existent_file}[/red]")


class TestMigrationIntegration:
    """Integration tests for the migration system."""

    def test_migration_orchestrator_with_mocked_reader(self):
        """Test migration orchestrator with mocked OWAMcapReader."""
        orchestrator = MigrationOrchestrator()

        # Test that the orchestrator can be created and has the expected script migrators
        assert len(orchestrator.script_migrators) == 2

        # Test migration path calculation
        path_0_3_0_to_0_4_1 = orchestrator.get_migration_path("0.3.0", "0.4.1")
        assert len(path_0_3_0_to_0_4_1) == 2

        path_0_3_2_to_0_4_1 = orchestrator.get_migration_path("0.3.2", "0.4.1")
        assert len(path_0_3_2_to_0_4_1) == 1

    @patch("owa.cli.mcap.migrate.migrate.OWAMcapReader")
    def test_detect_version_with_mocked_reader(self, mock_reader_class):
        """Test version detection with mocked reader."""
        # Setup mock
        mock_reader = MagicMock()
        mock_reader.file_version = "0.3.2"
        mock_reader_class.return_value.__enter__.return_value = mock_reader

        orchestrator = MigrationOrchestrator()

        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            version = orchestrator.detect_version(tmp_path)
            assert version == "0.3.2"
        finally:
            # Clean up the temporary file
            if tmp_path.exists():
                tmp_path.unlink()

    @patch("owa.cli.mcap.migrate.migrate.OWAMcapReader")
    def test_multi_step_migration_with_script_migrators(self, mock_reader_class):
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
            backup_path=None,
        )
        mock_migrator_1.verify_migration.return_value = True

        mock_migrator_2 = MagicMock(spec=ScriptMigrator)
        mock_migrator_2.from_version = "0.3.2"
        mock_migrator_2.to_version = "0.4.1"
        mock_migrator_2.migrate.return_value = MigrationResult(
            success=True,
            version_from="0.3.2",
            version_to="0.4.1",
            changes_made=1,
            backup_path=None,
        )
        mock_migrator_2.verify_migration.return_value = True

        # Replace orchestrator's script migrators with our mocks
        orchestrator.script_migrators = [mock_migrator_1, mock_migrator_2]

        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Mock the console for migrate_file
            from rich.console import Console

            console = Console()

            # This should trigger two migrations: 0.3.0 -> 0.3.2 -> 0.4.1
            results = orchestrator.migrate_file(tmp_path, target_version="0.4.1", console=console)

            # Verify migration was successful
            assert len(results) == 2
            assert all(result.success for result in results)

            # Verify that both migrators were called
            mock_migrator_1.migrate.assert_called_once()
            mock_migrator_2.migrate.assert_called_once()
        finally:
            # Clean up the temporary file
            if tmp_path.exists():
                tmp_path.unlink()
