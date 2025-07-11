"""
Pytest configuration for MCAP migration tests.

This module provides fixtures specific to migration testing.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_migration_result():
    """Fixture that provides a mock migration result."""
    result = MagicMock()
    result.success = True
    result.changes_made = 1
    result.error_message = None
    return result


@pytest.fixture
def mock_file_info():
    """Fixture that provides a mock file info object."""

    def _create_file_info(file_path, detected_version="0.3.0", needs_migration=True, target_version="0.4.0"):
        info = MagicMock()
        info.file_path = file_path
        info.detected_version = detected_version
        info.needs_migration = needs_migration
        info.target_version = target_version
        return info

    return _create_file_info
