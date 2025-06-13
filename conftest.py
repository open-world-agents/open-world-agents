import os
from pathlib import Path


def pytest_ignore_collect(path, config):
    """
    Return True to prevent pytest from collecting tests from the specified path.
    """
    str_path = str(path)

    # Check if running in GitHub Actions and if path contains projects/owa-env-gst
    if os.environ.get("GITHUB_ACTIONS") == "true" and "projects/owa-env-gst" in str_path:
        return True

    return False
