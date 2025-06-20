"""
MCAP file migration system with automatic version detection and sequential migration.

This module provides a comprehensive migration system using standalone script migrators that can:
1. Automatically detect the version of MCAP files
2. Apply sequential migrations to bring files up to the latest version
3. Verify migration success and rollback on failure
4. Handle multiple files with shell glob expansion
"""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from mcap_owa import __version__ as mcap_owa_version
from mcap_owa.highlevel import OWAMcapReader


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    success: bool
    version_from: str
    version_to: str
    changes_made: int
    error_message: Optional[str] = None
    backup_path: Optional[Path] = None


@dataclass
class ScriptMigrator:
    """Represents a standalone migration script."""

    script_path: Path
    from_version: str
    to_version: str

    def migrate(self, file_path: Path, console: Console, verbose: bool) -> MigrationResult:
        """Execute the standalone migration script."""
        try:
            # Build command for migrate subcommand
            cmd = ["uv", "run", str(self.script_path), "migrate", str(file_path)]

            if verbose:
                cmd.append("--verbose")

            # Execute script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.script_path.parent.parent.parent.parent,  # Run from owa-cli directory
            )

            if result.returncode == 0:
                # Count changes from output (simple heuristic) TODO
                changes_made = 1 if "changes made" in result.stdout else 0
                return MigrationResult(
                    success=True, version_from=self.from_version, version_to=self.to_version, changes_made=changes_made
                )
            else:
                return MigrationResult(
                    success=False,
                    version_from=self.from_version,
                    version_to=self.to_version,
                    changes_made=0,
                    error_message=result.stderr.strip() or result.stdout.strip(),
                )

        except Exception as e:
            return MigrationResult(
                success=False,
                version_from=self.from_version,
                version_to=self.to_version,
                changes_made=0,
                error_message=str(e),
            )

    def verify_migration(self, file_path: Path, backup_path: Optional[Path], console: Console) -> bool:
        """Verify migration by running the script with --verify flag."""
        try:
            # Build verification command
            cmd = ["uv", "run", str(self.script_path), "verify", str(file_path)]

            # Add backup path if provided
            if backup_path:
                cmd.extend(["--backup-path", str(backup_path)])

            # Execute verification
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.script_path.parent.parent.parent.parent
            )

            return result.returncode == 0

        except Exception:
            return False


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
        self.current_version = mcap_owa_version  # Use actual library version

        # Discover available script migrators
        self._discover_script_migrators()

    def _discover_script_migrators(self) -> None:
        """Discover standalone migration scripts automatically from filename patterns."""
        try:
            # Get the migrators directory relative to this file
            # migrate.py is in owa/cli/mcap/, migrators are in owa/cli/mcap/migrators/
            current_file = Path(__file__)
            migrators_dir = current_file.parent / "migrators"

            if not migrators_dir.exists():
                return

            # Automatically discover all Python files matching the pattern v{from}_to_v{to}.py
            import re

            version_pattern = re.compile(r"^v(\d+_\d+_\d+)_to_v(\d+_\d+_\d+)\.py$")

            for script_path in migrators_dir.glob("*.py"):
                match = version_pattern.match(script_path.name)
                if match:
                    # Convert underscores back to dots for version strings
                    from_version = match.group(1).replace("_", ".")
                    to_version = match.group(2).replace("_", ".")

                    self.script_migrators.append(
                        ScriptMigrator(script_path=script_path, from_version=from_version, to_version=to_version)
                    )

        except Exception as e:
            # If script discovery fails, log the error and continue
            typer.echo(f"Error discovering script migrators: {e}", err=True)

    def detect_version(self, file_path: Path) -> str:
        """Detect the version of an MCAP file using mcap-owa-support's stored version."""
        try:
            with OWAMcapReader(file_path) as reader:
                file_version = reader.file_version
                if file_version and file_version != "unknown":
                    return file_version
                # If version is unknown, default to current version
                return self.current_version
        except Exception:
            return "unknown"

    def create_backup(self, file_path: Path, backup_path: Path) -> None:
        """
        Create a backup of the file with high reliability.

        Args:
            file_path: Source file to backup
            backup_path: Destination backup path

        Raises:
            FileNotFoundError: If source file doesn't exist
            FileExistsError: If backup path already exists
            OSError: If backup creation or verification fails
        """
        # Ensure source file exists and is readable
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        # Check if backup already exists
        if backup_path.exists():
            raise FileExistsError(f"Backup file already exists: {backup_path}")

        # Ensure backup directory exists
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Create backup with metadata preservation
        shutil.copy2(file_path, backup_path)

        # Verify backup was created successfully
        if not backup_path.exists():
            raise OSError(f"Backup creation failed: {backup_path}")

        # Verify backup size matches original
        if backup_path.stat().st_size != file_path.stat().st_size:
            raise OSError(f"Backup verification failed: size mismatch for {backup_path}")

    def get_migration_path(self, from_version: str, to_version: str) -> List[ScriptMigrator]:
        """Get the sequence of migrators needed to go from one version to another."""
        if from_version == to_version:
            return []

        # Parse versions for comparison
        from packaging.version import Version

        try:
            from_ver = Version(from_version)
            to_ver = Version(to_version)
        except Exception:
            # If version parsing fails, fall back to exact matching
            return self._get_migration_path_exact(from_version, to_version)

        # Build migration ranges - each migrator covers a version range
        migration_ranges = []
        for migrator in self.script_migrators:
            try:
                range_start = Version(migrator.from_version)
                range_end = Version(migrator.to_version)
                migration_ranges.append((range_start, range_end, migrator))
            except Exception:
                # Skip migrators with unparseable versions
                continue

        # Sort migration ranges by start version
        migration_ranges.sort(key=lambda x: x[0])

        # Find path from source to target using version ranges
        path = []
        current_ver = from_ver

        while current_ver < to_ver:
            # Find a migrator that can handle the current version
            found_migrator = None
            for range_start, range_end, migrator in migration_ranges:
                # Check if current version is in the range [range_start, range_end)
                if range_start <= current_ver < range_end:
                    found_migrator = migrator
                    break

            if found_migrator is None:
                raise ValueError(f"No migration path found from {from_version} to {to_version}")

            path.append(found_migrator)
            current_ver = Version(found_migrator.to_version)

            # Prevent infinite loops
            if len(path) > 10:
                raise ValueError(f"Migration path too long from {from_version} to {to_version}")

        return path

    def _get_migration_path_exact(self, from_version: str, to_version: str) -> List[ScriptMigrator]:
        """Fallback method for exact version matching when version parsing fails."""
        # Build migration graph using script migrators
        migration_graph = {}
        for migrator in self.script_migrators:
            migration_graph[migrator.from_version] = migrator

        # Find path from source to target
        path = []
        current = from_version

        while current != to_version:
            if current not in migration_graph:
                raise ValueError(f"No migration path found from {from_version} to {to_version}")

            migrator = migration_graph[current]
            path.append(migrator)
            current = migrator.to_version

            # Prevent infinite loops
            if len(path) > 10:
                raise ValueError(f"Migration path too long from {from_version} to {to_version}")

        return path

    def get_highest_reachable_version(self, from_version: str) -> str:
        """Get the highest version reachable from the given version using available migrators."""
        if not self.script_migrators:
            return from_version

        # Parse version for comparison
        from packaging.version import Version

        try:
            from_ver = Version(from_version)
        except Exception:
            # If version parsing fails, fall back to exact matching
            return self._get_highest_reachable_version_exact(from_version)

        # Build migration ranges
        migration_ranges = []
        for migrator in self.script_migrators:
            try:
                range_start = Version(migrator.from_version)
                range_end = Version(migrator.to_version)
                migration_ranges.append((range_start, range_end, migrator))
            except Exception:
                continue

        # Sort migration ranges by start version
        migration_ranges.sort(key=lambda x: x[0])

        # Follow the migration chain as far as possible using version ranges
        current_ver = from_ver
        max_iterations = 10  # Prevent infinite loops

        for _ in range(max_iterations):
            # Find a migrator that can handle the current version
            found_migrator = None
            for range_start, range_end, migrator in migration_ranges:
                if range_start <= current_ver < range_end:
                    found_migrator = migrator
                    break

            if found_migrator is None:
                # No more migrations possible
                break

            current_ver = Version(found_migrator.to_version)

        return str(current_ver)

    def _get_highest_reachable_version_exact(self, from_version: str) -> str:
        """Fallback method for exact version matching when version parsing fails."""
        # Build migration graph using script migrators
        migration_graph = {}
        for migrator in self.script_migrators:
            migration_graph[migrator.from_version] = migrator

        # Follow the migration chain as far as possible
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
            console.print("[green]✓ File is already at target version[/green]")
            return []

        try:
            migration_path = self.get_migration_path(current_version, target_version)
        except ValueError as e:
            console.print(f"[red]✗ {e}[/red]")
            return []

        if not migration_path:
            console.print("[green]✓ No migration needed[/green]")
            return []

        console.print(f"Migration path: {' → '.join([m.from_version for m in migration_path] + [target_version])}")

        results = []

        # Create a single backup before starting migration (only for the first step)
        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")

        # Create backup with high reliability before any migration
        try:
            self.create_backup(file_path, backup_path)
        except Exception as e:
            console.print(f"[red]✗ Failed to create backup at {backup_path}: {e}[/red]")
            return results

        for i, migrator in enumerate(migration_path):
            console.print(
                f"\n[bold]Step {i + 1}/{len(migration_path)}: {migrator.from_version} → {migrator.to_version}[/bold]"
            )

            if verbose:
                console.print("Using script migrator")

            # Perform migration using script migrator
            result = migrator.migrate(file_path, console, verbose)
            result.backup_path = backup_path  # FIXME: better flow
            results.append(result)

            if not result.success:
                console.print(f"[red]✗ Migration failed: {result.error_message}[/red]")
                # Rollback all previous migrations
                self._rollback_migrations(file_path, backup_path, console)
                return results

            # Verify migration
            if not migrator.verify_migration(file_path, backup_path, console):
                console.print("[red]✗ Migration verification failed[/red]")
                result.success = False
                result.error_message = "Verification failed"
                # Rollback all previous migrations
                self._rollback_migrations(file_path, backup_path, console)
                return results

            console.print(f"[green]✓ Migration successful ({result.changes_made} changes)[/green]")

        console.print("\n[green]✓ All migrations completed successfully![/green]")
        return results

    def _rollback_migrations(self, file_path: Path, backup_path: Path, console: Console) -> None:
        """Rollback migrations by restoring from the backup."""
        console.print("[yellow]Rolling back migrations...[/yellow]")

        if backup_path and backup_path.exists():
            try:
                shutil.copy2(backup_path, file_path)
                console.print(f"[green]✓ Restored from backup: {backup_path}[/green]")
                return
            except Exception as e:
                console.print(f"[red]✗ Failed to restore backup: {e}[/red]")

        console.print("[red]✗ No valid backup found for rollback[/red]")


def detect_files_needing_migration(
    file_paths: List[Path], console: Console, target_version_explicit: bool, target_version: Optional[str]
) -> List[FileVersionInfo]:
    """Detect all files that need migration."""
    orchestrator = MigrationOrchestrator()
    file_infos = []

    # Filter for valid MCAP files
    valid_file_paths = []
    for file_path in file_paths:
        if file_path.is_file() and file_path.suffix == ".mcap":
            valid_file_paths.append(file_path)
        elif not file_path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
        elif not file_path.suffix == ".mcap":
            console.print(f"[yellow]Skipping non-MCAP file: {file_path}[/yellow]")

    if not valid_file_paths:
        console.print("[yellow]No valid MCAP files found[/yellow]")
        return []

    console.print(f"Found {len(valid_file_paths)} MCAP files")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing files...", total=len(valid_file_paths))

        for file_path in valid_file_paths:
            try:
                detected_version = orchestrator.detect_version(file_path)

                # Determine target version for this file
                if target_version_explicit:
                    # Use explicit target version
                    file_target_version = target_version
                    needs_migration = detected_version != target_version
                else:
                    # Use highest reachable version from detected version
                    file_target_version = orchestrator.get_highest_reachable_version(detected_version)
                    needs_migration = detected_version != file_target_version

                file_infos.append(
                    FileVersionInfo(
                        file_path=file_path,
                        detected_version=detected_version,
                        needs_migration=needs_migration,
                        target_version=file_target_version,
                    )
                )

            except Exception as e:
                console.print(f"[red]Error analyzing {file_path}: {e}[/red]")

            progress.update(task, advance=1)

    return file_infos


def migrate(
    files: List[Path] = typer.Argument(..., help="MCAP files to migrate"),
    target_version: Optional[str] = typer.Option(
        None, "--target", "-t", help="Target version (default: highest reachable)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be migrated without making changes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed migration information"),
    keep_backups: bool = typer.Option(True, "--keep-backups/--no-backups", help="Keep backup files after migration"),
) -> None:
    """
    Migrate MCAP files to the highest reachable version with automatic version detection.

    This command automatically detects the version of MCAP files and applies
    sequential migrations to bring them up to the highest reachable version using
    standalone script migrators. Each migration step is verified and can be rolled back on failure.

    Examples:
        owl mcap migrate *.mcap                      # Migrate all MCAP files (shell expands glob)
        owl mcap migrate data/**/*.mcap              # Migrate all MCAP files recursively
        owl mcap migrate recording.mcap --dry-run    # Preview migration for single file
        owl mcap migrate *.mcap --target 0.3.2       # Migrate to specific version
    """
    console = Console()
    orchestrator = MigrationOrchestrator()

    # Show migrator information if verbose
    if verbose:
        script_count = len(orchestrator.script_migrators)
        console.print(f"[dim]Available script migrators: {script_count}[/dim]")
        console.print()

    # Determine target version strategy
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

    # Filter files that need migration
    files_needing_migration = [info for info in file_infos if info.needs_migration]

    if not files_needing_migration:
        console.print("[green]✓ All files are already at the target version[/green]")
        return

    # Show summary table
    console.print("\n[bold]Migration Summary[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("Current Version", style="yellow")
    table.add_column("Target Version", style="green")
    table.add_column("Status", justify="center")

    for info in file_infos:
        status = "🔄" if info.needs_migration else "✅"
        table.add_row(str(info.file_path.name), info.detected_version, info.target_version, status)

    console.print(table)

    if dry_run:
        console.print(f"\n[yellow]Would migrate {len(files_needing_migration)} files[/yellow]")
        return

    # Confirm migration
    if not typer.confirm(f"\nProceed with migrating {len(files_needing_migration)} files?", default=True):
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
            # Use the file's specific target version (which may be different from global target)
            results = orchestrator.migrate_file(info.file_path, info.target_version, console, verbose)

            if results and all(r.success for r in results):
                successful_migrations += 1
                # Collect backup path (now only one per file)
                if results and results[0].backup_path:
                    backup_paths.append(results[0].backup_path)
            else:
                failed_migrations += 1

        except Exception as e:
            console.print(f"[red]✗ Unexpected error: {e}[/red]")
            failed_migrations += 1

    # Final summary
    console.print("\n[bold]Migration Complete[/bold]")
    console.print(f"✅ Successful: {successful_migrations}")
    console.print(f"❌ Failed: {failed_migrations}")

    # Handle backups
    if backup_paths and not keep_backups:
        console.print(f"\n[yellow]Cleaning up {len(backup_paths)} backup files...[/yellow]")
        for backup_path in backup_paths:
            try:
                if backup_path and backup_path.exists():
                    backup_path.unlink()
            except Exception as e:
                console.print(f"[red]Warning: Could not delete backup {backup_path}: {e}[/red]")
    elif backup_paths and keep_backups:
        console.print("\n[blue]Backup files saved with .mcap.backup extension[/blue]")

    if failed_migrations > 0:
        raise typer.Exit(1)
