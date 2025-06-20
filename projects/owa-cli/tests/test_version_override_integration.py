"""
Integration test to verify that version override works correctly during multi-step migration.

This test demonstrates that the fix for the version override issue works correctly,
ensuring that multi-step migrations (e.g., 0.3.0 -> 0.3.2 -> 0.4.1) proceed correctly
without being interrupted by incorrect version detection.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from owa.cli.mcap.migrate import MigrationOrchestrator
from owa.cli.mcap.migrators import MigrationResult


class MockMigrator:
    """Mock migrator for testing version override functionality."""

    def __init__(self, from_version: str, to_version: str):
        self.from_version = from_version
        self.to_version = to_version
        self.migrate_called = False
        self.version_written = None

    def migrate(self, file_path: Path, console, verbose: bool) -> MigrationResult:
        """Mock migration that captures the version being written."""
        self.migrate_called = True

        # Capture what version would be written to the file
        import mcap_owa.writer

        version_string = mcap_owa.writer._library_identifier()
        # Extract version from string like "mcap-owa-support 0.3.2; mcap 1.0.0"
        self.version_written = version_string.split()[1].rstrip(";")

        return MigrationResult(
            success=True,
            version_from=self.from_version,
            version_to=self.to_version,
            changes_made=1,
        )

    def verify_migration(self, file_path: Path, backup_path, console) -> bool:
        return True


class TestVersionOverrideIntegration:
    """Integration tests for version override functionality."""

    def test_multi_step_migration_writes_correct_versions(self):
        """Test that multi-step migration writes correct intermediate versions."""
        orchestrator = MigrationOrchestrator()

        # Create mock migrators that track what version they write
        migrator_1 = MockMigrator("0.3.0", "0.3.2")
        migrator_2 = MockMigrator("0.3.2", "0.4.1")

        # Replace orchestrator's migrators with our mocks
        orchestrator.migrators = [migrator_1, migrator_2]

        # Mock the file version detection to simulate the progression
        with patch("owa.cli.mcap.migrate.OWAMcapReader") as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.file_version = "0.3.0"  # Initial version
            mock_reader_class.return_value.__enter__.return_value = mock_reader

            with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            try:
                # Perform migration
                results = orchestrator.migrate_file(tmp_path, target_version="0.4.1")

                # Verify both migrations were called
                assert migrator_1.migrate_called
                assert migrator_2.migrate_called

                # Verify correct versions were written
                assert migrator_1.version_written == "0.3.2"  # First migration writes 0.3.2
                assert migrator_2.version_written == "0.4.1"  # Second migration writes 0.4.1

                # Verify migration results
                assert len(results) == 2
                assert all(result.success for result in results)
            finally:
                # Clean up the temporary file
                if tmp_path.exists():
                    tmp_path.unlink()

    def test_single_step_migration_writes_correct_version(self):
        """Test that single-step migration writes correct target version."""
        orchestrator = MigrationOrchestrator()

        # Create mock migrator
        migrator = MockMigrator("0.3.2", "0.4.1")
        orchestrator.migrators = [migrator]

        # Mock the file version detection
        with patch("owa.cli.mcap.migrate.OWAMcapReader") as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.file_version = "0.3.2"
            mock_reader_class.return_value.__enter__.return_value = mock_reader

            with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            try:
                # Perform migration
                results = orchestrator.migrate_file(tmp_path, target_version="0.4.1")

                # Verify migration was called
                assert migrator.migrate_called

                # Verify correct version was written
                assert migrator.version_written == "0.4.1"

                # Verify migration result
                assert len(results) == 1
                assert results[0].success
            finally:
                # Clean up the temporary file
                if tmp_path.exists():
                    tmp_path.unlink()

    def test_version_override_restores_original_function(self):
        """Test that version override properly restores the original function."""
        orchestrator = MigrationOrchestrator()

        import mcap_owa.writer

        # Store original function and its result
        original_func = mcap_owa.writer._library_identifier
        original_result = original_func()

        # Use the context manager
        with orchestrator._override_library_version("0.3.2"):
            # Inside context, should return overridden version
            overridden_result = mcap_owa.writer._library_identifier()
            assert "0.3.2" in overridden_result

        # After context, should be restored
        restored_result = mcap_owa.writer._library_identifier()
        assert restored_result == original_result

        # Function object should be the same
        assert mcap_owa.writer._library_identifier is original_func
