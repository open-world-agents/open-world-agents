"""
This script is used to restore the ocap command after the release process.
This process is needed to prevent `Failed to canonicalize script path` error.
The cause is `python`'s path is changed after the release process.
In principal, `conda-unpack` must restore the original path, but it does not work as expected.
"""

import os
import subprocess
import sys
import tempfile


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create pyproject.toml file
        with open(os.path.join(temp_dir, "pyproject.toml"), "w") as f:
            f.write("""
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
bypass-selection = true

[project]
name = "ocap-wrapper"
version = "0.1.0"
description = "Simple wrapper to restore ocap command"

[project.scripts]
owl = "owa.cli:app"
ocap = "owa.ocap.record:main"
""")

        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", temp_dir], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except Exception as e:
            print(f"Error restoring ocap command: {e}")


if __name__ == "__main__":
    main()
