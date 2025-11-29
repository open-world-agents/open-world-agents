"""
Pytest configuration for owa-cli tests.

This module provides global fixtures and configurations for testing owa-cli.
Environment variables are set to disable console styling and version checks
during test execution.
"""

import os
import shutil
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner


def pytest_configure(config):
    # Disable console styling to enable string comparison
    os.environ["OWA_DISABLE_CONSOLE_STYLING"] = "1"
    # Disable version checks to prevent GitHub API calls
    os.environ["OWA_DISABLE_VERSION_CHECK"] = "1"


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing.

    NOTE: Environment variables are also needed here for GitHub Actions
    (not just pytest_configure).
    """
    return CliRunner(
        charset="utf-8",
        env={"NO_COLOR": "1", "TERM": "dumb", "TTY_COMPATIBLE": "1", "TTY_INTERACTIVE": "0"},
    )


@pytest.fixture
def test_data_dir():
    """Get the test data directory with example MCAP files."""
    return Path(__file__).parent / "data"


@pytest.fixture
def copy_test_file():
    """Fixture that provides a function to copy test files."""

    def _copy(source_dir: Path, filename: str, dest_dir: Path, dest_filename: str | None = None) -> Path:
        """Copy a test file to the destination directory."""
        source = source_dir / filename
        dest = dest_dir / (dest_filename or filename)
        if source.exists():
            shutil.copy2(source, dest)
            return dest
        pytest.skip(f"Test file {filename} not found")

    return _copy


@pytest.fixture
def suppress_mcap_warnings():
    """Context manager to suppress MCAP version warnings during tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Reader version.*", UserWarning)
        yield
