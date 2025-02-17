"""
This `uv` wrapper script is needed to support automatic virtual environment support in `uv`.
Related issue: https://github.com/astral-sh/uv/issues/11315
"""

import argparse
import os
import subprocess


def main():
    """Parse all arguments and pass them to the `uv` command."""

    # Parse all arguments
    parser = argparse.ArgumentParser(description="Run the `uv` command.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments to pass to the `uv` command.")
    args = parser.parse_args()

    # Setup environment variables
    env = os.environ.copy()
    if "CONDA_DEFAULT_ENV" in env and env["CONDA_DEFAULT_ENV"] != "base":
        # to support `project` feature: https://docs.astral.sh/uv/configuration/environment/#uv_project_environment
        env["UV_PROJECT_ENVIRONMENT"] = env["CONDA_PREFIX"]
        # to support `--active` option: https://github.com/astral-sh/uv/pull/11189
        # this is not mandatory
        env["VIRTUAL_ENV"] = env["CONDA_PREFIX"]
    elif "VIRTUAL_ENV" in env:
        # to support `project` feature: https://docs.astral.sh/uv/configuration/environment/#uv_project_environment
        env["UV_PROJECT_ENVIRONMENT"] = env["VIRTUAL_ENV"]

    # Run the `uv` command
    subprocess.run(["uv"] + args.args, env=env)


if __name__ == "__main__":
    main()
