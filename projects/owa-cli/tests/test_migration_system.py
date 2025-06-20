"""
Tests for the MCAP migration system.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from owa.cli.mcap.migrate import (
    MigrationOrchestrator,
    detect_files_needing_migration,
)
from owa.cli.mcap.migrators import MigrationResult, discover_migrators


class TestMigrationOrchestrator:
    """Test the migration orchestrator."""

    def test_init(self):
        """Test orchestrator initialization."""
        orchestrator = MigrationOrchestrator()
        assert len(orchestrator.migrators) == 2
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
            result = orchestrator.create_backup(source_path, backup_path)

            assert result is True
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

        result = orchestrator.create_backup(source_path, backup_path)
        assert result is False

    def test_override_library_version(self):
        """Test that library version override works correctly."""
        orchestrator = MigrationOrchestrator()

        # Import the writer module to check the function
        import mcap_owa.writer

        # Store original function
        original_func = mcap_owa.writer._library_identifier
        original_result = original_func()

        # Test that override works
        target_version = "0.3.2"
        with orchestrator._override_library_version(target_version):
            overridden_result = mcap_owa.writer._library_identifier()
            assert target_version in overridden_result
            assert "mcap-owa-support 0.3.2" in overridden_result

        # Test that function is restored after context
        restored_result = mcap_owa.writer._library_identifier()
        assert restored_result == original_result

    def test_override_uses_patch_correctly(self):
        """Test that version override uses the imported patch correctly."""
        orchestrator = MigrationOrchestrator()

        # Verify that the context manager works without manual patching
        target_version = "0.3.2"
        with orchestrator._override_library_version(target_version):
            import mcap_owa.writer

            result = mcap_owa.writer._library_identifier()
            assert target_version in result
            assert "mcap-owa-support 0.3.2" in result

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


class TestMigrators:
    """Test individual migrators."""

    def test_discovered_migrators_have_expected_versions(self):
        """Test that discovered migrators have expected version transitions."""
        migrator_classes = discover_migrators()
        migrators = [cls() for cls in migrator_classes]

        # Should have at least 2 migrators
        assert len(migrators) >= 2

        # Check for expected version transitions
        version_transitions = {(m.from_version, m.to_version) for m in migrators}
        assert ("0.3.0", "0.3.2") in version_transitions
        assert ("0.3.2", "0.4.1") in version_transitions

    def test_migrators_have_required_properties(self):
        """Test that all discovered migrators have required properties."""
        migrator_classes = discover_migrators()
        migrators = [cls() for cls in migrator_classes]

        for migrator in migrators:
            assert hasattr(migrator, "from_version")
            assert hasattr(migrator, "to_version")
            assert hasattr(migrator, "migrate")
            assert hasattr(migrator, "verify_migration")
            assert migrator.from_version != migrator.to_version


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

        # Test that the orchestrator can be created and has the expected migrators
        assert len(orchestrator.migrators) == 2

        # Test migration path calculation
        path_0_3_0_to_0_4_1 = orchestrator.get_migration_path("0.3.0", "0.4.1")
        assert len(path_0_3_0_to_0_4_1) == 2

        path_0_3_2_to_0_4_1 = orchestrator.get_migration_path("0.3.2", "0.4.1")
        assert len(path_0_3_2_to_0_4_1) == 1

    @patch("owa.cli.mcap.migrate.OWAMcapReader")
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

    @patch("owa.cli.mcap.migrate.OWAMcapReader")
    def test_multi_step_migration_version_override(self, mock_reader_class):
        """Test that multi-step migration correctly writes intermediate versions."""
        orchestrator = MigrationOrchestrator()

        # Mock the reader to simulate version progression
        mock_reader = MagicMock()

        # First call: detect initial version as 0.3.0
        # Second call: after first migration, should detect 0.3.2 (not current version)
        # Third call: after second migration, should detect 0.4.1
        mock_reader.file_version = "0.3.0"  # Initial version
        mock_reader_class.return_value.__enter__.return_value = mock_reader

        # Create a mock migrator that we can track
        mock_migrator_1 = MagicMock()
        mock_migrator_1.from_version = "0.3.0"
        mock_migrator_1.to_version = "0.3.2"
        mock_migrator_1.migrate.return_value = MigrationResult(
            success=True,
            version_from="0.3.0",
            version_to="0.3.2",
            changes_made=1,
            backup_path=Path("/tmp/backup1.mcap"),
        )

        mock_migrator_2 = MagicMock()
        mock_migrator_2.from_version = "0.3.2"
        mock_migrator_2.to_version = "0.4.1"
        mock_migrator_2.migrate.return_value = MigrationResult(
            success=True,
            version_from="0.3.2",
            version_to="0.4.1",
            changes_made=1,
            backup_path=Path("/tmp/backup2.mcap"),
        )

        # Replace orchestrator's migrators with our mocks
        orchestrator.migrators = [mock_migrator_1, mock_migrator_2]

        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Test that version override context manager is used
            with patch.object(orchestrator, "_override_library_version") as mock_override:
                mock_override.return_value.__enter__ = MagicMock()
                mock_override.return_value.__exit__ = MagicMock()

                # This should trigger two migrations: 0.3.0 -> 0.3.2 -> 0.4.1
                results = orchestrator.migrate_file(tmp_path, target_version="0.4.1")

                # Verify migration was successful
                assert len(results) == 2
                assert all(result.success for result in results)

                # Verify that version override was called for each migration step
                assert mock_override.call_count == 2
                mock_override.assert_any_call("0.3.2")  # First migration target
                mock_override.assert_any_call("0.4.1")  # Second migration target
        finally:
            # Clean up the temporary file
            if tmp_path.exists():
                tmp_path.unlink()
