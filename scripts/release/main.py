#!/usr/bin/env python3
"""
OWA Release Manager - A comprehensive CLI tool for managing OWA package releases.

This tool provides functionality for:
- Updating package versions across multiple projects
- Publishing packages to PyPI
- Upgrading dependency lock files
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List

import typer

app = typer.Typer(help="OWA Release Manager - A tool for managing OWA package releases")

# Common project paths used across commands
PROJECTS = [
    ".",
    "projects/mcap-owa-support",
    "projects/owa-cli",
    "projects/owa-core",
    "projects/owa-env-desktop",
    "projects/owa-env-gst",
]


def get_package_dirs() -> List[Path]:
    """List all subrepositories in the projects directory."""
    return [Path(p) for p in PROJECTS]


def run_git_command(command: List[str]) -> str:
    """Run a git command and handle errors."""
    print(f"Running: git {' '.join(command)}")
    result = subprocess.run(["git"] + command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing git command: {result.stderr}")
        raise RuntimeError(f"Git command failed: {result.stderr}")
    return result.stdout.strip()


def run_command(command: List[str], cwd=None) -> str:
    """Run a shell command and return the output."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}")
        raise RuntimeError(f"Command failed: {result.stderr}")
    return result.stdout.strip()


def update_version_in_pyproject(pyproject_file: Path, version: str) -> bool:
    """Update version in pyproject.toml file."""
    with open(pyproject_file, "r") as f:
        content = f.read()

    # Update the version
    new_content = re.sub(r'version\s*=\s*"[^"]*"', f'version = "{version}"', content)

    # Only write if content changed
    if new_content != content:
        with open(pyproject_file, "w") as f:
            f.write(new_content)
        return True
    return False


def update_uv_lock(package_dir: Path) -> bool:
    """Run `uv lock --upgrade` in the given directory to update uv.lock."""
    print(f"Running `uv lock --upgrade` in {package_dir}")
    try:
        run_command(["uv", "lock", "--upgrade"], cwd=package_dir)
        print(f"✓ Updated uv.lock in {package_dir}")
        return True
    except RuntimeError as e:
        print(f"! Warning: {e}")
        return False


@app.command()
def new_version(
    version: str = typer.Argument(..., help="Version to set for all packages (e.g., 1.0.0)"),
    lock: bool = typer.Option(True, help="Update uv.lock files after changing versions"),
    push: bool = typer.Option(False, help="Push changes to git remote after committing"),
):
    """
    Update package versions in pyproject.toml files and create a git tag.

    This command updates the version in all project pyproject.toml files,
    optionally updates the lockfiles, commits the changes, and creates a git tag.
    """
    print(f"Setting all package versions to: {version}")

    # Check if the version tag already exists
    tag_name = f"v{version}"
    existing_tags = run_git_command(["tag"]).splitlines()
    if tag_name in existing_tags:
        print(f"! Error: Tag '{tag_name}' already exists. Aborting version update.")
        raise typer.Exit(code=1)

    # Find all project directories
    package_dirs = get_package_dirs()
    modified_files = []

    # Process each package
    for package_dir in package_dirs:
        print("=======================")
        print(f"Processing package in {package_dir}")

        # For all projects, check and update pyproject.toml
        pyproject_file = package_dir / "pyproject.toml"
        if pyproject_file.exists():
            print(f"Updating version in {pyproject_file}")
            if update_version_in_pyproject(pyproject_file, version):
                modified_files.append(pyproject_file)
                print(f"✓ Updated pyproject.toml version to {version}")
        else:
            print(f"! Warning: pyproject.toml not found in {package_dir}")

        print("=======================")

    # Update lock files if requested
    modified_lock_files = []
    if lock:
        print("Updating uv.lock files...")
        for package_dir in package_dirs:
            lock_file = package_dir / "uv.lock"
            if update_uv_lock(package_dir) and lock_file.exists():
                modified_lock_files.append(lock_file)

    # Commit changes if any
    if modified_files or modified_lock_files:
        print("Committing version changes...")

        # Add modified pyproject.toml files
        for file in modified_files:
            run_git_command(["add", str(file)])

        # Add modified lock files
        for file in modified_lock_files:
            run_git_command(["add", str(file)])

        tag_name = f"v{version}"
        run_git_command(["commit", "-m", f"{tag_name}"])
        run_git_command(["tag", tag_name])

        print(f"✓ Version updates committed and tagged as {tag_name}.")

        # Push changes if requested
        if push:
            print("Pushing changes to remote repository...")
            run_git_command(["push", "origin", "main"])
            run_git_command(["push", "origin", tag_name])
            print("✓ Changes pushed to remote repository.")
        else:
            print("")
            print("To push changes and tag to remote repository:")
            print(f"  git push origin main && git push origin {tag_name}")
    else:
        print("No files were modified. Nothing to commit.")

    print(f"All packages have been updated to version {version}!")


@app.command()
def publish():
    """
    Build and publish packages to PyPI.

    This command finds packages in the projects directory and publishes them using uv.
    A PyPI token must be set in the PYPI_TOKEN environment variable.
    """
    # Check if PyPI token is set
    if "PYPI_TOKEN" not in os.environ:
        print("PYPI_TOKEN environment variable is not set.")
        print("Please set it before running this script:")
        print("  export PYPI_TOKEN=your_token_here")
        raise typer.Exit(code=1)

    # https://docs.astral.sh/uv/guides/package/#publishing-your-package
    os.environ["UV_PUBLISH_TOKEN"] = os.environ["PYPI_TOKEN"]

    print("Building and publishing packages to PyPI...")

    # Find all project directories
    package_dirs = get_package_dirs()

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


@app.command()
def upgrade_all(
    commit: bool = typer.Option(False, help="Commit changes to git after updating uv.lock files"),
    push: bool = typer.Option(False, help="Push changes to remote after committing (requires --commit)"),
):
    """
    Update all uv.lock files and optionally commit changes.

    This command runs `uv lock --upgrade` for all projects to update dependency lock files.
    """
    print("Updating uv.lock files...")

    # Find all project directories
    package_dirs = get_package_dirs()
    modified_dirs = []

    # Process each package
    for package_dir in package_dirs:
        print("=======================")
        print(f"Processing package in {package_dir}")

        # Update uv.lock file
        if update_uv_lock(package_dir):
            modified_dirs.append(package_dir)

        print("=======================")

    # Commit changes if requested
    if commit and modified_dirs:
        print("Committing uv.lock updates...")
        for package_dir in modified_dirs:
            uv_lock_file = package_dir / "uv.lock"
            run_git_command(["add", str(uv_lock_file)])

        run_git_command(["commit", "-m", "build: updated `uv.lock`"])
        print("✓ uv.lock updates committed.")

        if push:
            print("Pushing changes to remote repository...")
            run_git_command(["push", "origin", "main"])
            print("✓ Changes pushed to remote repository.")
        else:
            print("")
            print("To push changes to the remote repository:")
            print("  git push origin main")
    elif not commit:
        print("Skipping git commit as per the CLI argument.")
    else:
        print("No uv.lock files were modified. Nothing to commit.")

    print("All uv.lock files have been updated!")


if __name__ == "__main__":
    app()
