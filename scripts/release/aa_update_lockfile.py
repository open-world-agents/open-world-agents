#!/usr/bin/env python3
"""
Script to update uv.lock files and commit changes.
Updates lock files in projects and commits the changes.
"""

import argparse
import subprocess
from pathlib import Path

PROJECTS = [
    ".",
    "projects/mcap-owa-support",
    "projects/owa-cli",
    "projects/owa-core",
    "projects/owa-env-desktop",
    "projects/owa-env-gst",
]


def packages_to_be_released() -> list[Path]:
    """List all subrepositories in the projects directory."""
    return [Path(p) for p in PROJECTS]


def update_uv_lock(package_dir: Path) -> bool:
    """Run `uv lock --upgrade` in the given directory to update uv.lock."""
    print(f"Running `uv lock --upgrade` in {package_dir}")
    result = subprocess.run(
        ["uv", "lock", "--upgrade"],
        cwd=package_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error updating uv.lock in {package_dir}: {result.stderr}")
        raise RuntimeError(f"`uv lock --upgrade` failed in {package_dir}: {result.stderr}")
    print(f"✓ Updated uv.lock in {package_dir}")
    return True


def run_git_command(command: list[str]) -> None:
    """Run a git command."""
    print(f"Running: git {' '.join(command)}")
    result = subprocess.run(["git"] + command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing git command: {result.stderr}")
        raise RuntimeError(f"Git command failed: {result.stderr}")
    return result.stdout.strip()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Update uv.lock files and optionally commit changes.")
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit changes to git after updating uv.lock files.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    print("Updating uv.lock files...")

    # Find all project directories
    package_dirs = packages_to_be_released()
    modified_dirs = []

    # Process each package
    for package_dir in package_dirs:
        print("=======================")
        print(f"Processing package in {package_dir}")

        # Update uv.lock file
        try:
            if update_uv_lock(package_dir):
                modified_dirs.append(package_dir)
        except RuntimeError as e:
            print(f"! Warning: {e}")

        print("=======================")

    # Commit changes if any
    if args.commit and modified_dirs:
        print("Committing uv.lock updates...")
        for package_dir in modified_dirs:
            uv_lock_file = package_dir / "uv.lock"
            run_git_command(["add", str(uv_lock_file)])

        run_git_command(["commit", "-m", "build: updated `uv.lock`"])

        print("✓ uv.lock updates committed.")
        print("")
        print("To push changes to the remote repository:")
        print("  git push origin main")
    elif not args.commit:
        print("Skipping git commit as per the CLI argument.")
    else:
        print("No uv.lock files were modified. Nothing to commit.")

    print("All uv.lock files have been updated!")


if __name__ == "__main__":
    main()
