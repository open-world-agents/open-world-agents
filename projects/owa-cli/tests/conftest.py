"""
Pytest configuration for owa-cli tests.

This module provides global fixtures and configurations for testing owa-cli,
including mocking of GitHub API calls to prevent rate limiting during test execution.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner


@pytest.fixture(scope="session", autouse=True)
def mock_github_api_calls():
    """
    Automatically mock all GitHub API calls during test execution.

    This fixture runs for the entire test session and prevents any real
    GitHub API calls that could cause rate limiting issues.
    """
    # Mock the main OWA version check API call
    with patch("owa.cli.utils.requests.get") as mock_get:
        # Set up mock response for OWA version check
        mock_response = MagicMock()
        mock_response.json.return_value = {"tag_name": "v0.4.2"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Also mock the mcap CLI version check API call
        with patch("owa.cli.mcap.info.requests.get") as mock_mcap_get:
            # Set up mock response for mcap CLI version check
            mock_mcap_response = MagicMock()
            mock_mcap_response.json.return_value = [
                {"tag_name": "releases/mcap-cli/v0.0.54"},
                {"tag_name": "releases/rust/v0.19.0"},
                {"tag_name": "releases/mcap-cli/v0.0.53"},
            ]
            mock_mcap_response.raise_for_status.return_value = None
            mock_mcap_get.return_value = mock_mcap_response

            yield


@pytest.fixture(scope="session", autouse=True)
def disable_version_checks():
    """
    Disable version checks during testing by setting environment variable.

    This provides an additional layer of protection against GitHub API calls
    by setting a flag that can be checked by the application code.
    """
    original_value = os.environ.get("OWA_DISABLE_VERSION_CHECK")
    os.environ["OWA_DISABLE_VERSION_CHECK"] = "1"
    yield
    if original_value is None:
        os.environ.pop("OWA_DISABLE_VERSION_CHECK", None)
    else:
        os.environ["OWA_DISABLE_VERSION_CHECK"] = original_value


@pytest.fixture
def mock_mcap_version_functions():
    """
    Provide mocked versions of mcap version checking functions.

    This fixture can be used by individual tests that need specific
    version checking behavior.
    """
    with (
        patch("owa.cli.mcap.info.get_latest_mcap_cli_version") as mock_latest,
        patch("owa.cli.mcap.info.get_local_mcap_version") as mock_local,
    ):
        mock_latest.return_value = "v0.0.54"
        mock_local.return_value = "v0.0.53"

        yield {
            "get_latest_mcap_cli_version": mock_latest,
            "get_local_mcap_version": mock_local,
        }


@pytest.fixture
def mock_owa_version_functions():
    """
    Provide mocked versions of OWA version checking functions.

    This fixture can be used by individual tests that need specific
    version checking behavior.
    """
    with (
        patch("owa.cli.utils.get_latest_release") as mock_latest,
        patch("owa.cli.utils.get_local_version") as mock_local,
    ):
        mock_latest.return_value = "0.4.2"
        mock_local.return_value = "0.4.1"

        yield {
            "get_latest_release": mock_latest,
            "get_local_version": mock_local,
        }


@pytest.fixture
def no_version_check():
    """
    Completely disable version checking for tests that don't need it.

    This fixture mocks the check_for_update function to do nothing.
    """
    with patch("owa.cli.utils.check_for_update") as mock_check:
        mock_check.return_value = True
        yield mock_check


# Common fixtures for all CLI tests
@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner(charset="utf-8")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
