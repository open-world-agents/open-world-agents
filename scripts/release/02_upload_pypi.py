#!/usr/bin/env python3
"""
Script to build and publish packages to PyPI.
Finds packages in the projects/ directory and publishes them using uv.
"""

import os
import subprocess
from pathlib import Path


def list_subrepos() -> list[str]:
    """List all subrepositories in the projects directory."""
    projects_dir = Path("projects")
    return [d for d in projects_dir.iterdir() if d.is_dir() and d.name.startswith("owa")]


def run_command(command, cwd=None):
    """Run a shell command and return the output."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}")
        raise RuntimeError(f"Command failed: {result.stderr}")
    return result.stdout.strip()


def main():
    # Check if PyPI token is set
    if "PYPI_TOKEN" not in os.environ:
        print("PYPI_TOKEN environment variable is not set.")
        print("Please set it before running this script:")
        print("  export PYPI_TOKEN=your_token_here")
        exit(1)

    print("Building and publishing packages to PyPI...")

    # Find all project directories
    package_dirs = list_subrepos()

    # Process each package
    for package_dir in package_dirs:
        print("=======================")
        print(f"Processing package in {package_dir}")

        # Check if package directory has required files
        pyproject_exists = (package_dir / "pyproject.toml").exists()
        setup_exists = (package_dir / "setup.py").exists()

        if pyproject_exists or setup_exists:
            print(f"Building and publishing package in {package_dir}")
            try:
                run_command(["uv", "build"], cwd=package_dir)
                run_command(["uv", "publish"], cwd=package_dir)
                print(f"✓ Published {package_dir.name} successfully")
            except RuntimeError as e:
                print(f"! Failed to publish {package_dir.name}: {e}")
        else:
            print(f"! Skipping {package_dir.name} - No pyproject.toml or setup.py found")

        print("=======================")

    print("All packages have been built and published!")


if __name__ == "__main__":
    main()
