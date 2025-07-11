"""
Pytest configuration for MCAP tests.

This module provides fixtures and utilities specific to MCAP testing,
including test file management and common MCAP testing utilities.
"""

import re
import shutil
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def strip_ansi_codes():
    """Fixture that provides a function to strip ANSI escape codes from text."""

    def _strip_ansi_codes(text: str) -> str:
        """Strip ANSI escape codes from text for more reliable testing."""
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    return _strip_ansi_codes


@pytest.fixture
def copy_test_file():
    """Fixture that provides a function to copy test files."""

    def _copy_test_file(source_dir: Path, filename: str, dest_dir: Path) -> Path:
        """Copy a test file to the destination directory."""
        source_file = source_dir / filename
        dest_file = dest_dir / filename
        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            return dest_file
        else:
            pytest.skip(f"Test file {filename} not found in {source_dir}")

    return _copy_test_file


@pytest.fixture
def suppress_mcap_warnings():
    """Context manager to suppress MCAP version warnings during tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Reader version.*may not be compatible with writer version.*", UserWarning)
        warnings.filterwarnings("ignore", "unclosed file.*", ResourceWarning)
        yield


@pytest.fixture
def mock_migration_orchestrator():
    """Fixture that provides a mocked MigrationOrchestrator."""
    with patch("owa.cli.mcap.migrate.migrate.MigrationOrchestrator") as mock_class:
        mock_orchestrator = MagicMock()
        mock_class.return_value = mock_orchestrator
        yield mock_orchestrator


@pytest.fixture
def mock_mcap_reader():
    """Fixture that provides a mocked OWAMcapReader."""
    with patch("owa.cli.mcap.migrate.migrate.OWAMcapReader") as mock_reader:
        yield mock_reader


@pytest.fixture
def mock_file_detection():
    """Fixture that provides mocked file detection functionality."""
    with patch("owa.cli.mcap.migrate.migrate.detect_files_needing_migration") as mock_detect:
        yield mock_detect


@pytest.fixture
def test_data_dir():
    """Get the test data directory with example MCAP files."""
    return Path(__file__).parent / "data"
