"""
Pytest configuration for owa-cli tests.

This module provides global fixtures and configurations for testing owa-cli,
including mocking of GitHub API calls to prevent rate limiting during test execution.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner


def pytest_configure(config):
    # Disable console styling to enable string comparison
    os.environ["OWA_DISABLE_CONSOLE_STYLING"] = "1"
    # Disable version checks to prevent GitHub API calls
    os.environ["OWA_DISABLE_VERSION_CHECK"] = "1"


@pytest.fixture(scope="session", autouse=True)
def mock_github_api_calls():
    """
    Automatically mock GitHub API calls during test execution.

    This fixture runs for the entire test session and prevents any real
    GitHub API calls that could cause rate limiting issues. It only mocks
    specific GitHub API endpoints to avoid interfering with other HTTP requests.
    """
    # Store the original requests.get function before any patching
    import requests

    original_requests_get = requests.get

    def mock_requests_get(url, *args, **kwargs):
        """Mock requests.get but only for GitHub API calls."""
        # Mock OWA version check API call
        if url == "https://api.github.com/repos/open-world-agents/open-world-agents/releases/latest":
            mock_response = MagicMock()
            mock_response.json.return_value = {"tag_name": "v0.4.2"}
            mock_response.raise_for_status.return_value = None
            return mock_response

        # Mock mcap CLI version check API call
        elif url == "https://api.github.com/repos/foxglove/mcap/releases":
            mock_response = MagicMock()
            mock_response.json.return_value = [
                {"tag_name": "releases/mcap-cli/v0.0.54"},
                {"tag_name": "releases/rust/v0.19.0"},
                {"tag_name": "releases/mcap-cli/v0.0.53"},
            ]
            mock_response.raise_for_status.return_value = None
            return mock_response

        # For any other URL, make the actual request using the original function
        return original_requests_get(url, *args, **kwargs)

    # Mock both modules' requests.get calls
    with patch("owa.cli.utils.requests.get", side_effect=mock_requests_get):
        with patch("owa.cli.mcap.info.requests.get", side_effect=mock_requests_get):
            yield


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
    # NOTE: OWA_DISABLE_CONSOLE_STYLING or equivalent configuration ALSO needed here to disable console styling.
    # Only matters in Github Action, not local, can't guess why
    return CliRunner(
        charset="utf-8", env={"NO_COLOR": "1", "TERM": "dumb", "TTY_COMPATIBLE": "1", "TTY_INTERACTIVE": "0"}
    )
