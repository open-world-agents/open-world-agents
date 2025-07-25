import os

import pytest


def pytest_runtest_setup(item):
    """
    Skip tests based on platform and environment conditions.
    This will mark tests as "skipped" rather than ignoring them completely.
    """
    test_path = item.path.as_posix()

    # On Github Actions, skip tests that require physical display
    if os.environ.get("GITHUB_ACTIONS") == "true" and "projects/owa-env-gst" in test_path:
        pytest.skip("Skipping tests that require physical display on GitHub Actions")

    # Skip Windows-specific tests on non-Windows platforms
    if os.name != "nt" and (
        "projects/owa-env-gst" in test_path or "projects/owa-env-desktop" in test_path or "projects/ocap" in test_path
    ):
        pytest.skip("Skipping Windows-specific tests on non-Windows platform")
