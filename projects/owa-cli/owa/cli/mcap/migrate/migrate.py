"""
MCAP file migration system with automatic version detection and sequential migration.
"""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import typer
from packaging.version import Version
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from mcap_owa import __version__ as mcap_owa_version
from mcap_owa.highlevel import OWAMcapReader


def _get_subprocess_env():
    """Get environment variables for subprocess calls with proper encoding."""
    import os

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    success: bool
    version_from: str
    version_to: str
    # TODO: define structured output format and add `--output-format` argument to migrators
    changes_made: int
    error_message: str = ""


@dataclass
class ScriptMigrator:
    """Represents a standalone migration script."""

    script_path: Path
    from_version: str
    to_version: str

    def migrate(self, file_path: Path, verbose: bool) -> MigrationResult:
        """Execute the standalone migration script."""
        cmd = ["uv", "run", str(self.script_path), "migrate", str(file_path)]
        if verbose:
            cmd.append("--verbose")

        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=_get_subprocess_env())

        if result.returncode == 0:
            changes_made = 1 if result.stdout and "changes made" in result.stdout else 0
            return MigrationResult(True, self.from_version, self.to_version, changes_made)

        error_msg = (
            result.stderr.strip() if result.stderr else result.stdout.strip() if result.stdout else "Migration failed"
        )
        return MigrationResult(False, self.from_version, self.to_version, 0, error_msg)

    def verify_migration(self, file_path: Path, backup_path: Optional[Path]) -> bool:
        """Verify migration by running the script with --verify flag."""
        cmd = ["uv", "run", str(self.script_path), "verify", str(file_path)]
        if backup_path:
            cmd.extend(["--backup-path", str(backup_path)])

        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=_get_subprocess_env())
        return result.returncode == 0


@dataclass
class FileVersionInfo:
    """Information about an MCAP file's version."""

    file_path: Path
    detected_version: str
    needs_migration: bool
    target_version: str


class MigrationOrchestrator:
    """Orchestrates sequential migrations for MCAP files using script migrators."""

    def __init__(self) -> None:
        """Initialize the orchestrator."""
        self.script_migrators: List[ScriptMigrator] = []
        self.current_version = mcap_owa_version
        self._discover_script_migrators()

    def _discover_script_migrators(self) -> None:
        """Discover standalone migration scripts automatically from filename patterns."""
        migrators_dir = Path(__file__).parent / "migrators"
        if not migrators_dir.exists():
            return

        version_pattern = re.compile(r"^v(\d+_\d+_\d+)_to_v(\d+_\d+_\d+)\.py$")

        for script_path in migrators_dir.glob("*.py"):
            match = version_pattern.match(script_path.name)
            if match:
                from_version = match.group(1).replace("_", ".")
                to_version = match.group(2).replace("_", ".")
                self.script_migrators.append(ScriptMigrator(script_path, from_version, to_version))

    def detect_version(self, file_path: Path) -> str:
        """Detect the version of an MCAP file using mcap-owa-support's stored version."""
        try:
            with OWAMcapReader(file_path) as reader:
                file_version = reader.file_version
                return file_version if file_version and file_version != "unknown" else self.current_version
        except Exception:
            return "unknown"

    def create_backup(self, file_path: Path, backup_path: Path) -> None:
        """Create a backup of the file with high reliability."""
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        if backup_path.exists():
            raise FileExistsError(f"Backup file already exists: {backup_path}")

        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)

        if not backup_path.exists():
            raise OSError(f"Backup creation failed: {backup_path}")

        if backup_path.stat().st_size != file_path.stat().st_size:
            raise OSError(f"Backup verification failed: size mismatch for {backup_path}")

    def get_migration_path(self, from_version: str, to_version: str) -> List[ScriptMigrator]:
        """Get the sequence of migrators needed to go from one version to another."""
        if from_version == to_version:
            return []

        try:
            from_ver = Version(from_version)
            to_ver = Version(to_version)

            # Build migration ranges
            migration_ranges = []
            for migrator in self.script_migrators:
                try:
                    range_start = Version(migrator.from_version)
                    range_end = Version(migrator.to_version)
                    migration_ranges.append((range_start, range_end, migrator))
                except Exception:
                    continue

            migration_ranges.sort(key=lambda x: x[0])

            # Find path from source to target
            path = []
            current_ver = from_ver

            while current_ver < to_ver:
                found_migrator = None
                for range_start, range_end, migrator in migration_ranges:
                    if range_start <= current_ver < range_end:
                        found_migrator = migrator
                        break

                if found_migrator is None:
                    raise ValueError(f"No migration path found from {from_version} to {to_version}")

                path.append(found_migrator)
                current_ver = Version(found_migrator.to_version)

                if len(path) > 10:  # Prevent infinite loops
                    raise ValueError(f"Migration path too long from {from_version} to {to_version}")

            return path

        except Exception:
            # Fallback to exact matching
            return self._get_migration_path_exact(from_version, to_version)

    def _get_migration_path_exact(self, from_version: str, to_version: str) -> List[ScriptMigrator]:
        """Fallback method for exact version matching when version parsing fails."""
        migration_graph = {migrator.from_version: migrator for migrator in self.script_migrators}

        path = []
        current = from_version

        while current != to_version:
            if current not in migration_graph:
                raise ValueError(f"No migration path found from {from_version} to {to_version}")

            migrator = migration_graph[current]
            path.append(migrator)
            current = migrator.to_version

            if len(path) > 10:  # Prevent infinite loops
                raise ValueError(f"Migration path too long from {from_version} to {to_version}")

        return path

    def get_highest_reachable_version(self, from_version: str) -> str:
        """Get the highest version reachable from the given version using available migrators."""
        if not self.script_migrators:
            return from_version

        try:
            from_ver = Version(from_version)

            # Build migration ranges
            migration_ranges = []
            for migrator in self.script_migrators:
                try:
                    range_start = Version(migrator.from_version)
                    range_end = Version(migrator.to_version)
                    migration_ranges.append((range_start, range_end, migrator))
                except Exception:
                    continue

            migration_ranges.sort(key=lambda x: x[0])

            # Follow the migration chain as far as possible
            current_ver = from_ver
            for _ in range(10):  # Prevent infinite loops
                found_migrator = None
                for range_start, range_end, migrator in migration_ranges:
                    if range_start <= current_ver < range_end:
                        found_migrator = migrator
                        break

                if found_migrator is None:
                    break

                current_ver = Version(found_migrator.to_version)

            return str(current_ver)

        except Exception:
            # Fallback to exact matching
            migration_graph = {migrator.from_version: migrator for migrator in self.script_migrators}
            current = from_version
            while current in migration_graph:
                migrator = migration_graph[current]
                current = migrator.to_version
            return current

    def migrate_file(
        self, file_path: Path, target_version: Optional[str] = None, console: Console = None, verbose: bool = False
    ) -> List[MigrationResult]:
        """Migrate a single file through all necessary steps."""
        if console is None:
            console = Console()

        if target_version is None:
            target_version = self.current_version

        current_version = self.detect_version(file_path)
        console.print(f"Detected version: {current_version}")

        if current_version == target_version:
            console.print("[green]File is already at target version[/green]")
            return []

        try:
            migration_path = self.get_migration_path(current_version, target_version)
        except ValueError as e:
            console.print(f"[red]Migration error: {e}[/red]")
            return []

        if not migration_path:
            console.print("[green]No migration needed[/green]")
            return []

        console.print(f"Migration path: {' → '.join([m.from_version for m in migration_path] + [target_version])}")

        # Create backup before starting migration
        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
        self.create_backup(file_path, backup_path)

        results = []
        for i, migrator in enumerate(migration_path):
            console.print(
                f"\n[bold]Step {i + 1}/{len(migration_path)}: {migrator.from_version} → {migrator.to_version}[/bold]"
            )

            if verbose:
                console.print("Using script migrator")

            # Perform migration
            result = migrator.migrate(file_path, verbose)
            results.append(result)

            if not result.success:
                console.print(f"[red]Migration failed: {result.error_message}[/red]")
                self._rollback_migrations(file_path, backup_path, console)
                return results

            # Verify migration
            if not migrator.verify_migration(file_path, backup_path):
                console.print("[red]Migration verification failed[/red]")
                result.success = False
                result.error_message = "Verification failed"
                self._rollback_migrations(file_path, backup_path, console)
                return results

            console.print(f"[green]Migration successful ({result.changes_made} changes)[/green]")

        console.print("\n[green]All migrations completed successfully![/green]")
        return results

    def _rollback_migrations(self, file_path: Path, backup_path: Path, console: Console) -> None:
        """Rollback migrations by restoring from the backup."""
        console.print("[yellow]Rolling back migrations...[/yellow]")

        if backup_path.exists():
            shutil.copy2(backup_path, file_path)
            backup_path.unlink()
            console.print(f"[green]Restored from backup: {backup_path}[/green]")
        else:
            console.print("[red]No valid backup found for rollback[/red]")


def detect_files_needing_migration(
    file_paths: List[Path], console: Console, target_version_explicit: bool, target_version: Optional[str]
) -> List[FileVersionInfo]:
    """Detect all files that need migration."""
    orchestrator = MigrationOrchestrator()

    # Filter for valid MCAP files
    valid_file_paths = [fp for fp in file_paths if fp.is_file() and fp.suffix == ".mcap"]

    for file_path in file_paths:
        if not file_path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
        elif file_path.suffix != ".mcap":
            console.print(f"[yellow]Skipping non-MCAP file: {file_path}[/yellow]")

    if not valid_file_paths:
        console.print("[yellow]No valid MCAP files found[/yellow]")
        return []

    console.print(f"Found {len(valid_file_paths)} MCAP files")

    file_infos = []
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Analyzing files...", total=len(valid_file_paths))

        for file_path in valid_file_paths:
            try:
                detected_version = orchestrator.detect_version(file_path)

                if target_version_explicit:
                    file_target_version = target_version
                    needs_migration = detected_version != target_version
                else:
                    file_target_version = orchestrator.get_highest_reachable_version(detected_version)
                    needs_migration = detected_version != file_target_version

                file_infos.append(FileVersionInfo(file_path, detected_version, needs_migration, file_target_version))

            except Exception as e:
                console.print(f"[red]Error analyzing {file_path}: {e}[/red]")

            progress.update(task, advance=1)

    return file_infos


def _check_uv_available() -> bool:
    """Check if uv is available in the system."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def migrate(
    files: List[Path] = typer.Argument(..., help="MCAP files to migrate"),
    target_version: Optional[str] = typer.Option(
        None, "--target", "-t", help="Target version (default: highest reachable)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be migrated without making changes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed migration information"),
    keep_backups: bool = typer.Option(True, "--keep-backups/--no-backups", help="Keep backup files after migration"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """
    Migrate MCAP files to the highest reachable version with automatic version detection.
    """
    console = Console()

    # Check if uv is available before proceeding
    if not _check_uv_available():
        console.print("[red]Error: 'uv' is not installed or not available in PATH[/red]")
        console.print(
            "[yellow]Please install uv first: https://docs.astral.sh/uv/getting-started/installation/[/yellow]"
        )
        raise typer.Exit(1)

    orchestrator = MigrationOrchestrator()

    if verbose:
        console.print(f"[dim]Available script migrators: {len(orchestrator.script_migrators)}[/dim]\n")

    target_version_explicit = target_version is not None

    console.print("[bold blue]MCAP Migration Tool[/bold blue]")
    console.print(f"Target version: {target_version or 'highest reachable'}")
    console.print(f"Files to process: {len(files)}")

    if dry_run:
        console.print("[yellow]DRY RUN MODE - No files will be modified[/yellow]")

    # Detect files needing migration
    file_infos = detect_files_needing_migration(files, console, target_version_explicit, target_version)

    if not file_infos:
        console.print("[yellow]No files found matching the pattern[/yellow]")
        return

    files_needing_migration = [info for info in file_infos if info.needs_migration]

    if not files_needing_migration:
        console.print("[green]All files are already at the target version[/green]")
        return

    # Show summary table
    console.print("\n[bold]Migration Summary[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("Current Version", style="yellow")
    table.add_column("Target Version", style="green")
    table.add_column("Status", justify="center")

    for info in file_infos:
        status = "MIGRATE" if info.needs_migration else "OK"
        table.add_row(str(info.file_path.name), info.detected_version, info.target_version, status)

    console.print(table)

    if dry_run:
        console.print(f"\n[yellow]Would migrate {len(files_needing_migration)} files[/yellow]")
        return

    if not yes and not typer.confirm(f"\nProceed with migrating {len(files_needing_migration)} files?", default=True):
        console.print("Migration cancelled.")
        return

    # Perform migrations
    console.print(f"\n[bold]Starting migration of {len(files_needing_migration)} files...[/bold]")

    successful_migrations = 0
    failed_migrations = 0
    backup_paths = []

    for i, info in enumerate(files_needing_migration, 1):
        console.print(f"\n[bold cyan]File {i}/{len(files_needing_migration)}: {info.file_path}[/bold cyan]")

        try:
            results = orchestrator.migrate_file(info.file_path, info.target_version, console, verbose)

            if results and all(r.success for r in results):
                successful_migrations += 1
                backup_path = info.file_path.with_suffix(f"{info.file_path.suffix}.backup")
                if backup_path.exists():
                    backup_paths.append(backup_path)
            else:
                failed_migrations += 1

        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            failed_migrations += 1

    # Final summary
    console.print("\n[bold]Migration Complete[/bold]")
    console.print(f"[green]Successful: {successful_migrations}[/green]")
    console.print(f"[red]Failed: {failed_migrations}[/red]")

    # Handle backups
    if backup_paths and not keep_backups:
        console.print(f"\n[yellow]Cleaning up {len(backup_paths)} backup files...[/yellow]")
        for backup_path in backup_paths:
            try:
                if backup_path.exists():
                    backup_path.unlink()
            except Exception as e:
                console.print(f"[red]Warning: Could not delete backup {backup_path}: {e}[/red]")
    elif backup_paths and keep_backups:
        console.print("\n[blue]Backup files saved with .mcap.backup extension[/blue]")

    if failed_migrations > 0:
        raise typer.Exit(1)
