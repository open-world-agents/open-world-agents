#!/usr/bin/env python3
"""
Script to update package versions and tag repository.
Updates versions in projects and creates a git tag.
"""

import argparse
import re
import subprocess
from pathlib import Path


def list_subrepos() -> list[str]:
    """List all subrepositories in the projects directory."""
    projects = [Path(".")]
    for d in Path("projects").iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith("owa"):
            continue
        if d.name == "owa-env-example":
            continue
        projects.append(d)
    return projects


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


def run_git_command(command: list[str]) -> None:
    """Run a git command."""
    print(f"Running: git {' '.join(command)}")
    result = subprocess.run(["git"] + command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing git command: {result.stderr}")
        raise RuntimeError(f"Git command failed: {result.stderr}")
    return result.stdout.strip()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Update package versions and tag repository")
    parser.add_argument("version", help="Version to set for all packages (e.g., 1.0.0)")
    args = parser.parse_args()

    version = args.version
    print(f"Setting all package versions to: {version}")

    # Find all project directories
    package_dirs = list_subrepos()
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

    # Commit changes if any
    if modified_files:
        print("Committing version changes...")
        for file in modified_files:
            run_git_command(["add", str(file)])

        run_git_command(["commit", "-m", f"chore: update package versions to {version}"])
        tag_name = f"v{version}"
        run_git_command(["tag", tag_name])

        print(f"✓ Version updates committed and tagged as {tag_name}.")
        print("")
        print("To push changes and tag to remote repository:")
        print(f"  git push origin main && git push origin {tag_name}")
    else:
        print("No files were modified. Nothing to commit.")

    print(f"All packages have been updated to version {version}!")


if __name__ == "__main__":
    main()
