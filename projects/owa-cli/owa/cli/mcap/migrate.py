"""
MCAP file migration system with automatic version detection and sequential migration.

This module provides a comprehensive migration system that can:
1. Automatically detect the version of MCAP files
2. Apply sequential migrations to bring files up to the latest version
3. Verify migration success and rollback on failure
4. Handle multiple files with shell glob expansion
"""

import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

try:
    from mcap_owa import __version__ as mcap_owa_version
    from mcap_owa.highlevel import OWAMcapReader
except ImportError as e:
    typer.echo(f"Error: Required packages not available: {e}", err=True)
    typer.echo("Please install: pip install mcap-owa-support", err=True)
    raise typer.Exit(1)

from dataclasses import dataclass

from .migrators import BaseMigrator, MigrationResult, get_all_migrators


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
                # Count changes from output (simple heuristic)
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
    """Orchestrates sequential migrations for MCAP files."""

    def __init__(self, use_scripts: bool = True):
        """
        Initialize the orchestrator.

        Args:
            use_scripts: If True, prefer standalone scripts over class-based migrators
        """
        self.use_scripts = use_scripts
        self.migrators: List[BaseMigrator] = []
        self.script_migrators: List[ScriptMigrator] = []
        self.current_version = mcap_owa_version  # Use actual library version

        # Discover available migrators
        if use_scripts:
            self._discover_script_migrators()

        # Always discover class-based migrators as fallback
        self.migrators = get_all_migrators()

    def _discover_script_migrators(self):
        """Discover standalone migration scripts."""
        try:
            # Get the migrators directory relative to this file
            # migrate.py is in owa/cli/mcap/, migrators are in owa/cli/mcap/migrators/
            current_file = Path(__file__)
            migrators_dir = current_file.parent / "migrators"

            if not migrators_dir.exists():
                return

            # Look for migration scripts - check if they're uv scripts
            script_patterns = [
                ("v0_3_0_to_v0_3_2.py", "0.3.0", "0.3.2"),
                ("v0_3_2_to_v0_4_1.py", "0.3.2", "0.4.1"),
            ]

            for script_name, from_version, to_version in script_patterns:
                script_path = migrators_dir / script_name
                if script_path.exists():
                    # Check if it's a uv script by looking for the shebang
                    try:
                        with open(script_path, "r") as f:
                            first_line = f.readline().strip()
                            if first_line.startswith("#!/usr/bin/env -S uv run --script"):
                                self.script_migrators.append(
                                    ScriptMigrator(
                                        script_path=script_path, from_version=from_version, to_version=to_version
                                    )
                                )
                    except Exception:
                        pass

        except Exception:
            # If script discovery fails, fall back to class-based migrators
            pass

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

    def create_backup(self, file_path: Path, backup_path: Path) -> bool:
        """
        Create a backup of the file with high reliability.

        Args:
            file_path: Source file to backup
            backup_path: Destination backup path

        Returns:
            True if backup was successful, False otherwise
        """
        try:
            # Ensure source file exists and is readable
            if not file_path.exists():
                return False

            # Ensure backup directory exists
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup with metadata preservation
            shutil.copy2(file_path, backup_path)

            # Verify backup was created successfully
            if not backup_path.exists():
                return False

            # Verify backup size matches original
            if backup_path.stat().st_size != file_path.stat().st_size:
                return False

            return True
        except Exception:
            return False

    @contextmanager
    def _override_library_version(self, target_version: str):
        """
        Context manager to temporarily override the mcap-owa-support version
        written to MCAP files during migration.

        This ensures that migrated files have the correct target version
        in their headers instead of the current library version.
        """

        # Create patched function that returns target version
        def patched_library_identifier():
            import mcap

            mcap_version = getattr(mcap, "__version__", "<=0.0.10")
            return f"mcap-owa-support {target_version}; mcap {mcap_version}"

        # Use the imported patch decorator
        with patch("mcap_owa.writer._library_identifier", patched_library_identifier):
            yield

    def get_migration_path(self, from_version: str, to_version: str):
        """Get the sequence of migrators needed to go from one version to another."""
        if from_version == to_version:
            return []

        # Build migration graph - prefer scripts over class-based migrators
        migration_graph = {}

        # Add script migrators first (higher priority)
        if self.use_scripts:
            for migrator in self.script_migrators:
                migration_graph[migrator.from_version] = migrator

        # Add class-based migrators (lower priority, won't override scripts)
        for migrator in self.migrators:
            if migrator.from_version not in migration_graph:
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
        if not self.migrators and not self.script_migrators:
            return from_version

        # Build migration graph - prefer scripts over class-based migrators
        migration_graph = {}

        # Add script migrators first (higher priority)
        if self.use_scripts:
            for migrator in self.script_migrators:
                migration_graph[migrator.from_version] = migrator

        # Add class-based migrators (lower priority)
        for migrator in self.migrators:
            if migrator.from_version not in migration_graph:
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
        if not self.create_backup(file_path, backup_path):
            console.print(f"[red]✗ Failed to create backup at {backup_path}[/red]")
            return results

        for i, migrator in enumerate(migration_path):
            console.print(
                f"\n[bold]Step {i + 1}/{len(migration_path)}: {migrator.from_version} → {migrator.to_version}[/bold]"
            )

            # Show which type of migrator is being used
            migrator_type = "script" if isinstance(migrator, ScriptMigrator) else "class-based"
            if verbose:
                console.print(f"Using {migrator_type} migrator")

            # Perform migration
            if isinstance(migrator, ScriptMigrator):
                # Script migrators handle their own backup and don't need version override
                result = migrator.migrate(file_path, console, verbose)
            else:
                # Class-based migrators need version override
                with self._override_library_version(migrator.to_version):
                    result = migrator.migrate(file_path, console, verbose)

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

    def _rollback_migrations(self, file_path: Path, backup_path: Path, console: Console):
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
    use_scripts: bool = typer.Option(True, "--scripts/--no-scripts", help="Use standalone uv scripts (default: True)"),
) -> None:
    """
    Migrate MCAP files to the highest reachable version with automatic version detection.

    This command automatically detects the version of MCAP files and applies
    sequential migrations to bring them up to the highest reachable version using
    available migrators. Each migration step is verified and can be rolled back on failure.

    Examples:
        owl mcap migrate *.mcap                      # Migrate all MCAP files (shell expands glob)
        owl mcap migrate data/**/*.mcap              # Migrate all MCAP files recursively
        owl mcap migrate recording.mcap --dry-run    # Preview migration for single file
        owl mcap migrate *.mcap --target 0.3.2       # Migrate to specific version
    """
    console = Console()
    orchestrator = MigrationOrchestrator(use_scripts=use_scripts)

    # Show migrator information if verbose
    if verbose:
        script_count = len(orchestrator.script_migrators)
        class_count = len(orchestrator.migrators)
        console.print(f"[dim]Available migrators: {script_count} scripts, {class_count} class-based[/dim]")
        if use_scripts and script_count > 0:
            console.print("[dim]Using standalone uv scripts (preferred)[/dim]")
        elif class_count > 0:
            console.print("[dim]Using class-based migrators[/dim]")
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
