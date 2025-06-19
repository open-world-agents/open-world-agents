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
from owa.cli.mcap.migrators import discover_migrators


class TestMigrationOrchestrator:
    """Test the migration orchestrator."""

    def test_init(self):
        """Test orchestrator initialization."""
        orchestrator = MigrationOrchestrator()
        assert len(orchestrator.migrators) == 2
        # Current version should match the actual mcap-owa-support library version
        from mcap_owa import __version__ as mcap_owa_version

        assert orchestrator.current_version == mcap_owa_version

    def test_detect_version_fallback_to_current(self):
        """Test version detection falls back to current version for non-existent file."""
        orchestrator = MigrationOrchestrator()
        non_existent_file = Path("/non/existent/file.mcap")
        version = orchestrator.detect_version(non_existent_file)
        assert version == orchestrator.current_version

    def test_get_migration_path_no_migration_needed(self):
        """Test migration path when no migration is needed."""
        orchestrator = MigrationOrchestrator()
        path = orchestrator.get_migration_path("0.4.0", "0.4.0")
        assert path == []

    def test_get_migration_path_sequential(self):
        """Test sequential migration path."""
        orchestrator = MigrationOrchestrator()
        path = orchestrator.get_migration_path("0.3.0", "0.4.0")
        assert len(path) == 2
        assert path[0].from_version == "0.3.0"
        assert path[0].to_version == "0.3.2"
        assert path[1].from_version == "0.3.2"
        assert path[1].to_version == "0.4.0"

    def test_get_migration_path_from_v032(self):
        """Test migration path from v0.3.2 to v0.4.0."""
        orchestrator = MigrationOrchestrator()
        path = orchestrator.get_migration_path("0.3.2", "0.4.0")
        assert len(path) == 1
        assert path[0].from_version == "0.3.2"
        assert path[0].to_version == "0.4.0"

    def test_get_migration_path_invalid(self):
        """Test invalid migration path."""
        orchestrator = MigrationOrchestrator()
        with pytest.raises(ValueError, match="No migration path found"):
            orchestrator.get_migration_path("unknown", "0.4.0")


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
        assert ("0.3.2", "0.4.0") in version_transitions

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
        result = detect_files_needing_migration([], console)
        assert result == []
        console.print.assert_called_with("[yellow]No valid MCAP files found[/yellow]")

    def test_detect_files_with_non_existent_file(self):
        """Test detection with non-existent file."""
        console = MagicMock()
        non_existent_file = Path("/non/existent/file.mcap")

        result = detect_files_needing_migration([non_existent_file], console)

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
        path_0_3_0_to_0_4_0 = orchestrator.get_migration_path("0.3.0", "0.4.0")
        assert len(path_0_3_0_to_0_4_0) == 2

        path_0_3_2_to_0_4_0 = orchestrator.get_migration_path("0.3.2", "0.4.0")
        assert len(path_0_3_2_to_0_4_0) == 1

    @patch("owa.cli.mcap.migrate.OWAMcapReader")
    def test_detect_version_with_mocked_reader(self, mock_reader_class):
        """Test version detection with mocked reader."""
        # Setup mock
        mock_reader = MagicMock()
        mock_reader.file_version = "0.3.2"
        mock_reader_class.return_value.__enter__.return_value = mock_reader

        orchestrator = MigrationOrchestrator()

        with tempfile.NamedTemporaryFile(suffix=".mcap") as tmp_file:
            tmp_path = Path(tmp_file.name)
            version = orchestrator.detect_version(tmp_path)
            assert version == "0.3.2"
