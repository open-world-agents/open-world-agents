#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "hatch",
#     "packaging",
#     "rich",
#     "typer",
# ]
# ///
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

try:
    from rich.console import Console
    from rich.panel import Panel
except ImportError:
    raise ImportError("rich is required. Install with: pip install rich")

app = typer.Typer(help="OWA Release Manager - A tool for managing OWA package releases")
console = Console()

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


def run_git_command(command: List[str], verbose: bool = False) -> str:
    """Run a git command."""
    if verbose:
        console.print(f"[dim]$ git {' '.join(command)}[/dim]")
    result = subprocess.run(["git"] + command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"Git command failed: {result.stderr}")
    return result.stdout.strip()


def run_command(command: List[str], cwd: Path = None, verbose: bool = False) -> str:
    """Run a shell command."""
    if verbose:
        console.print(f"[dim]$ {' '.join(command)}[/dim]")
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {result.stderr}")
    return result.stdout.strip()


@app.command()
def version(
    value: str = typer.Argument(..., help="Version to set for all packages (e.g., 1.0.0)"),
    lock: bool = typer.Option(True, "--lock", help="Update uv.lock files after changing versions"),
    commit: bool = typer.Option(True, "--commit/--no-commit", help="Commit changes to git"),
    tag: bool = typer.Option(False, "--tag/--no-tag", help="Create git tag (requires --commit)"),
    push: bool = typer.Option(False, "--push", help="Push changes to git remote after committing"),
):
    """
    Update package versions using vuv and hatch version management.

    This command:
    1. Detects first-party dependencies for each package
    2. Updates dependencies using 'vuv add x==v --frozen'
    3. Updates package version using 'vuv version v' or 'hatch version v'
    4. Optionally runs lock command if --lock is specified
    5. Optionally commits changes if --commit is specified
    6. Optionally creates git tag if --tag is specified (requires --commit)
    7. Optionally pushes changes if --push is specified
    """
    if value.startswith("v"):
        value = value[1:]

    # Validate arguments
    if tag and not commit:
        console.print("[bold red]✗ Error: --tag requires --commit to be enabled.[/bold red]")
        raise typer.Exit(code=1)

    console.print(Panel(f"[bold blue]Setting all package versions to: {value}[/bold blue]", title="Version Update"))

    # Check if tag already exists when tagging is enabled
    if tag:
        tag_name = f"v{value}"
        existing_tags = run_git_command(["tag"]).splitlines()
        if tag_name in existing_tags:
            console.print(f"[bold red]✗ Error: Tag '{tag_name}' already exists. Aborting version update.[/bold red]")
            raise typer.Exit(code=1)

    # Process each package
    package_dirs = get_package_dirs()
    packages_updated = 0

    for package_dir in package_dirs:
        package_name = get_package_name(package_dir)
        if not package_name:
            console.print(f"[yellow]⚠ Warning: Could not determine package name for {package_dir}[/yellow]")
            continue

        first_party_deps = get_first_party_dependencies(package_dir)

        # Create a clean package info display
        console.print(f"\n[bold cyan]📦 {package_name}[/bold cyan] [dim]({package_dir})[/dim]")

        if first_party_deps:
            deps_str = ", ".join(sorted(first_party_deps))
            console.print(f"   Dependencies: [dim]{deps_str}[/dim]")

        # Step 1: Update first-party dependencies
        for dep in first_party_deps:
            with console.status(f"Updating {dep} dependency..."):
                run_command(["vuv", "add", f"{dep}=={value}", "--frozen"], cwd=package_dir)
            console.print(f"   [green]✓[/green] Updated {dep} → {value}")

        # Step 2: Update package version
        with console.status(f"Updating {package_name} version..."):
            # TODO?: we may adopt hatch-vcs, a plugin for Hatch that uses Git to determine project versions.
            try:
                run_command(["vuv", "version", value], cwd=package_dir)
                version_tool = "vuv"
            except RuntimeError:
                run_command(["hatch", "version", value], cwd=package_dir)
                version_tool = "hatch"

        console.print(f"   [green]✓[/green] Updated version → {value} [dim]({version_tool})[/dim]")
        packages_updated += 1

    # Step 3: Run lock command if requested
    if lock:
        console.print("\n[bold yellow]🔒 Updating lock files...[/bold yellow]")
        for package_dir in package_dirs:
            with console.status(f"Locking {package_dir.name}..."):
                run_command(["vuv", "lock"], cwd=package_dir)
            console.print(f"   [green]✓[/green] {package_dir.name}")

    # Step 4: Commit changes if requested
    if commit:
        console.print("\n[bold magenta]📝 Committing changes...[/bold magenta]")

        # Check if there are any modified files to commit
        git_status = run_git_command(["status", "--porcelain"])

        if git_status.strip():
            # Add all modified files (handles dynamic versioning and any other changes)
            run_git_command(["add", "-A"])
            files_added = True
        else:
            files_added = False

        if files_added:
            commit_message = f"v{value}"
            run_git_command(["commit", "-m", commit_message])
            console.print(f"   [green]✓[/green] Committed changes with message: [bold]{commit_message}[/bold]")

            # Step 5: Create tag if requested
            if tag:
                tag_name = f"v{value}"
                run_git_command(["tag", tag_name])
                console.print(f"   [green]✓[/green] Created tag: [bold]{tag_name}[/bold]")

            # Step 6: Push changes if requested
            if push:
                console.print("\n[bold blue]🚀 Pushing to remote...[/bold blue]")
                run_git_command(["push", "origin", "main"])
                console.print("   [green]✓[/green] Pushed changes to remote")

                if tag:
                    run_git_command(["push", "origin", tag_name])
                    console.print(f"   [green]✓[/green] Pushed tag {tag_name} to remote")
            else:
                push_commands = ["git push origin main"]
                if tag:
                    push_commands.append(f"git push origin {tag_name}")

                console.print("\n[blue]To push changes to remote:[/blue]")
                console.print(f"[cyan]  {' && '.join(push_commands)}[/cyan]")
        else:
            console.print("   [yellow]⚠[/yellow] No files were modified. Nothing to commit.")
            if tag:
                console.print("   [yellow]⚠[/yellow] Cannot create tag without committing changes.")

    console.print(
        f"\n[bold green]🎉 Success![/bold green] Updated {packages_updated} packages to version [bold]{value}[/bold]"
    )


@app.command()
def publish():
    """
    Build and publish packages to PyPI.

    This command finds packages in the projects directory and publishes them using uv.
    A PyPI token must be set in the PYPI_TOKEN environment variable.
    """
    # Check if PyPI token is set
    if "PYPI_TOKEN" not in os.environ:
        console.print("[bold red]✗ PYPI_TOKEN environment variable is not set.[/bold red]")
        console.print("Please set it before running this script:")
        console.print("[cyan]  export PYPI_TOKEN=your_token_here[/cyan]")
        raise typer.Exit(code=1)

    # https://docs.astral.sh/uv/guides/package/#publishing-your-package
    os.environ["UV_PUBLISH_TOKEN"] = os.environ["PYPI_TOKEN"]

    console.print(Panel("[bold blue]Building and publishing packages to PyPI[/bold blue]", title="Publish"))

    published_count = 0
    skipped_count = 0

    # Process each package
    for package_dir in get_package_dirs():
        package_name = get_package_name(package_dir) or package_dir.name

        # Check if package directory has required files
        pyproject_exists = (package_dir / "pyproject.toml").exists()
        setup_exists = (package_dir / "setup.py").exists()

        if pyproject_exists or setup_exists:
            console.print(f"\n[bold cyan]📦 {package_name}[/bold cyan] [dim]({package_dir})[/dim]")

            with console.status("Building package..."):
                run_command(["uv", "build"], cwd=package_dir)
            console.print("   [green]✓[/green] Built package")

            with console.status("Publishing to PyPI..."):
                run_command(["uv", "publish"], cwd=package_dir)
            console.print("   [green]✓[/green] Published to PyPI")
            published_count += 1
        else:
            console.print(f"\n[yellow]⚠ {package_name}[/yellow] [dim]({package_dir})[/dim]")
            console.print("   [dim]Skipped - No pyproject.toml or setup.py found[/dim]")
            skipped_count += 1

    console.print(f"\n[bold green]🎉 Success![/bold green] Published {published_count} packages")
    if skipped_count > 0:
        console.print(f"[dim]Skipped {skipped_count} packages[/dim]")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def lock(ctx: typer.Context):
    """
    Run 'vuv lock ARGS' in all first-party repositories.

    This command runs 'vuv lock' with the provided arguments in all project directories.
    Common usage: 'lock --upgrade' to upgrade all dependencies.
    """
    args = ctx.params.get("args", []) or ctx.args
    args_str = " ".join(args) if args else ""

    title = f"Lock Dependencies{' ' + args_str if args_str else ''}"
    command_desc = f"vuv lock{' ' + args_str if args_str else ''}"
    console.print(Panel(f"[bold blue]Running '{command_desc}' in all repositories[/bold blue]", title=title))

    for package_dir in get_package_dirs():
        package_name = get_package_name(package_dir) or package_dir.name
        console.print(f"\n[bold cyan]📦 {package_name}[/bold cyan] [dim]({package_dir})[/dim]")

        with console.status(f"Running vuv lock{' ' + args_str if args_str else ''}..."):
            run_command(["vuv", "lock"] + args, cwd=package_dir)
        console.print("   [green]✓[/green] Lock completed")

    console.print("\n[bold green]🎉 Success![/bold green] Lock command completed for all repositories")


if __name__ == "__main__":
    app()
