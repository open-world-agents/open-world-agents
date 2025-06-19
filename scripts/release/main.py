#!/usr/bin/env python3
"""
OWA Release Manager - CLI tool for managing OWA package releases.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Set

import typer

# Use tomllib for Python 3.11+, tomli for older versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("tomli is required for Python < 3.11. Install with: pip install tomli")

try:
    from packaging.requirements import Requirement
except ImportError:
    raise ImportError("packaging is required. Install with: pip install packaging")

app = typer.Typer(help="OWA Release Manager - A tool for managing OWA package releases")

# Project paths and first-party packages
PROJECTS = [
    ".",
    "projects/mcap-owa-support",
    "projects/ocap",
    "projects/owa-cli",
    "projects/owa-core",
    "projects/owa-env-desktop",
    "projects/owa-env-gst",
    "projects/owa-msgs",
]

FIRST_PARTY_PACKAGES = {
    "owa",
    "mcap-owa-support",
    "ocap",
    "owa-cli",
    "owa-core",
    "owa-env-desktop",
    "owa-env-gst",
    "owa-msgs",
}


def get_package_dirs() -> List[Path]:
    """List all project directories."""
    return [Path(p) for p in PROJECTS]


def get_package_name(package_dir: Path) -> str:
    """Get package name from pyproject.toml."""
    pyproject_file = package_dir / "pyproject.toml"
    if not pyproject_file.exists():
        return ""

    raw_toml = pyproject_file.read_text(encoding="utf-8")
    data = tomllib.loads(raw_toml)
    return data.get("project", {}).get("name", "")


def get_first_party_dependencies(package_dir: Path) -> Set[str]:
    """Get first-party dependencies from pyproject.toml."""
    pyproject_file = package_dir / "pyproject.toml"
    if not pyproject_file.exists():
        return set()

    raw_toml = pyproject_file.read_text(encoding="utf-8")
    data = tomllib.loads(raw_toml)

    dependencies = set()
    raw_deps = data.get("project", {}).get("dependencies", [])

    for dep_str in raw_deps:
        req = Requirement(dep_str)
        if req.name in FIRST_PARTY_PACKAGES:
            dependencies.add(req.name)

    return dependencies


def run_git_command(command: List[str]) -> str:
    """Run a git command."""
    print(f"Running: git {' '.join(command)}")
    result = subprocess.run(["git"] + command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"Git command failed: {result.stderr}")
    return result.stdout.strip()


def run_command(command: List[str], cwd: Path = None) -> str:
    """Run a shell command."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {result.stderr}")
    return result.stdout.strip()


@app.command()
def version(
    value: str = typer.Argument(..., help="Version to set for all packages (e.g., 1.0.0)"),
    lock: bool = typer.Option(True, "--lock", help="Update uv.lock files after changing versions"),
    tag: bool = typer.Option(True, "--tag/--no-tag", help="Create git tag and commit changes"),
    push: bool = typer.Option(False, "--push", help="Push changes to git remote after committing"),
):
    """
    Update package versions using vuv and hatch version management.

    This command:
    1. Detects first-party dependencies for each package
    2. Updates dependencies using 'vuv add x==v --frozen'
    3. Updates package version using 'vuv version v' or 'hatch version v'
    4. Optionally runs lock command if --lock is specified
    5. Optionally commits and tags changes if --tag is specified
    6. Optionally pushes changes if --push is specified
    """
    if value.startswith("v"):
        value = value[1:]

    print(f"Setting all package versions to: {value}")

    # Check if tag already exists when tagging is enabled
    if tag:
        tag_name = f"v{value}"
        existing_tags = run_git_command(["tag"]).splitlines()
        if tag_name in existing_tags:
            print(f"! Error: Tag '{tag_name}' already exists. Aborting version update.")
            raise typer.Exit(code=1)

    # Process each package
    package_dirs = get_package_dirs()
    packages_updated = 0

    for package_dir in package_dirs:
        print("=======================")
        print(f"Processing package in {package_dir}")

        package_name = get_package_name(package_dir)
        if not package_name:
            print(f"! Warning: Could not determine package name for {package_dir}")
            continue

        first_party_deps = get_first_party_dependencies(package_dir)
        print(f"Package: {package_name}")
        print(f"First-party dependencies: {first_party_deps}")

        # Step 1: Detect first-party dependencies and update them
        for dep in first_party_deps:
            print(f"Updating dependency {dep} to version {value}")
            run_command(["vuv", "add", f"{dep}=={value}", "--frozen"], cwd=package_dir)
            print(f"✓ Updated {dep} dependency to {value}")

        # Step 2: Update package version
        print(f"Updating {package_name} version to {value}")
        try:
            run_command(["vuv", "version", value], cwd=package_dir)
            print(f"✓ Updated {package_name} version to {value} using vuv")
        except RuntimeError:
            run_command(["hatch", "version", value], cwd=package_dir)
            print(f"✓ Updated {package_name} version to {value} using hatch")
        packages_updated += 1

        print("=======================")

    # Step 3: Run lock command if requested
    if lock:
        print("Running lock command...")
        for package_dir in package_dirs:
            run_command(["vuv", "lock"], cwd=package_dir)
            print(f"✓ Successfully ran 'vuv lock' in {package_dir}")

    # Step 4: Commit and tag changes if requested
    if tag:
        print("Committing version changes...")
        files_added = False

        for package_dir in package_dirs:
            pyproject_file = package_dir / "pyproject.toml"
            uv_lock_file = package_dir / "uv.lock"

            if pyproject_file.exists():
                run_git_command(["add", str(pyproject_file)])
                files_added = True
            if uv_lock_file.exists():
                run_git_command(["add", str(uv_lock_file)])
                files_added = True

        if files_added:
            tag_name = f"v{value}"
            run_git_command(["commit", "-m", f"{tag_name}"])
            run_git_command(["tag", tag_name])
            print(f"✓ Version updates committed and tagged as {tag_name}.")

            # Step 5: Push changes if requested
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

    print(f"All packages have been updated to version {value}! ({packages_updated} packages processed)")


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

    # Process each package
    for package_dir in get_package_dirs():
        print("=======================")
        print(f"Processing package in {package_dir}")

        # Check if package directory has required files
        pyproject_exists = (package_dir / "pyproject.toml").exists()
        setup_exists = (package_dir / "setup.py").exists()

        if pyproject_exists or setup_exists:
            print(f"Building and publishing package in {package_dir}")
            run_command(["uv", "build"], cwd=package_dir)
            run_command(["uv", "publish"], cwd=package_dir)
            print(f"✓ Published {package_dir.name} successfully")
        else:
            print(f"! Skipping {package_dir.name} - No pyproject.toml or setup.py found")

        print("=======================")

    print("All packages have been built and published!")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def lock(ctx: typer.Context):
    """
    Run 'vuv lock ARGS' in all first-party repositories.

    This command runs 'vuv lock' with the provided arguments in all project directories.
    Common usage: 'lock --upgrade' to upgrade all dependencies.
    """
    args = ctx.params.get("args", []) or ctx.args

    print(f"Running 'vuv lock {' '.join(args)}' in all repositories...")

    for package_dir in get_package_dirs():
        print("=======================")
        print(f"Processing package in {package_dir}")

        run_command(["vuv", "lock"] + args, cwd=package_dir)
        print(f"✓ Successfully ran 'vuv lock {' '.join(args)}' in {package_dir}")

        print("=======================")

    print("Lock command completed for all repositories!")


if __name__ == "__main__":
    app()
