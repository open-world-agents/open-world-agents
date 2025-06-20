"""
Test to verify that no GitHub API calls are made during normal test execution.
"""

from unittest.mock import patch

import pytest


def test_no_github_api_calls_during_import():
    """Test that importing owa.cli modules doesn't trigger GitHub API calls."""
    with patch("requests.get") as mock_get:
        # Import the main CLI module
        import owa.cli  # noqa: F401
        import owa.cli.mcap.info  # noqa: F401
        import owa.cli.utils  # noqa: F401

        # Verify no API calls were made during import
        mock_get.assert_not_called()


def test_no_github_api_calls_during_version_check():
    """Test that version checking functions respect the environment variable."""
    from owa.cli.mcap.info import get_latest_mcap_cli_version
    from owa.cli.utils import check_for_update

    with patch("requests.get") as mock_get:
        # These should not make API calls due to environment variable
        check_for_update()
        get_latest_mcap_cli_version()

        # Verify no API calls were made
        mock_get.assert_not_called()


def test_main_cli_entry_point_no_api_calls():
    """Test that the main CLI entry point doesn't make API calls."""
    from owa.cli import main

    with patch("requests.get") as mock_get:
        # This should not make API calls due to environment variable
        main()

        # Verify no API calls were made
        mock_get.assert_not_called()


def test_environment_variable_is_set():
    """Test that the OWA_DISABLE_VERSION_CHECK environment variable is set during tests."""
    import os

    # This should be set by conftest.py
    assert os.environ.get("OWA_DISABLE_VERSION_CHECK") == "1"


@pytest.mark.parametrize(
    "module_name",
    [
        "owa.cli.utils",
        "owa.cli.mcap.info",
    ],
)
def test_version_functions_respect_env_var(module_name):
    """Test that version checking functions in various modules respect the environment variable."""
    import importlib
    import os

    module = importlib.import_module(module_name)

    # Verify environment variable is set
    assert os.environ.get("OWA_DISABLE_VERSION_CHECK") == "1"

    with patch("requests.get") as mock_get:
        # Try to call version checking functions if they exist
        if hasattr(module, "get_latest_release"):
            module.get_latest_release()
        if hasattr(module, "get_latest_mcap_cli_version"):
            module.get_latest_mcap_cli_version()
        if hasattr(module, "check_for_update"):
            module.check_for_update()

        # Verify no API calls were made
        mock_get.assert_not_called()
