"""
MCAP file migration system with automatic version detection and sequential migration.

This module provides a comprehensive migration system that can:
1. Automatically detect the version of MCAP files
2. Apply sequential migrations to bring files up to the latest version
3. Verify migration success and rollback on failure
4. Handle multiple files with shell glob expansion
"""

import shutil
from pathlib import Path
from typing import List, Optional

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
class FileVersionInfo:
    """Information about an MCAP file's version."""

    file_path: Path
    detected_version: str
    needs_migration: bool
    target_version: str


class MigrationOrchestrator:
    """Orchestrates sequential migrations for MCAP files."""

    def __init__(self):
        self.migrators: List[BaseMigrator] = get_all_migrators()
        self.current_version = mcap_owa_version  # Use actual library version

    def detect_version(self, file_path: Path) -> str:
        """Detect the version of an MCAP file."""
        try:
            # Try to read version from OWAMcapReader first (most reliable)
            try:
                with OWAMcapReader(file_path) as reader:
                    file_version = reader.file_version
                    if file_version and file_version != "unknown":
                        return file_version
            except Exception:
                pass

            # Check for legacy format (old schema names) - treat as 0.3.2 since they need schema migration
            try:
                with OWAMcapReader(file_path) as reader:
                    for schema in reader.schemas.values():
                        if schema.name.startswith("owa.env."):
                            return "0.3.2"
            except Exception:
                pass

            # Check for v0.3.0 format (keyboard state field changes)
            try:
                with OWAMcapReader(file_path) as reader:
                    for msg in reader.iter_messages(topics=["keyboard/state"]):
                        if hasattr(msg.decoded, "pressed_vk_list"):
                            return "0.3.0"
                        break
            except Exception:
                pass

            # Default to current version if we can't detect
            return self.current_version

        except Exception:
            return "unknown"

    def get_migration_path(self, from_version: str, to_version: str) -> List[BaseMigrator]:
        """Get the sequence of migrators needed to go from one version to another."""
        if from_version == to_version:
            return []

        # Build migration graph
        migration_graph = {}
        for migrator in self.migrators:
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
        backup_dir = file_path.parent / ".mcap_migration_backups"
        backup_dir.mkdir(exist_ok=True)

        for i, migrator in enumerate(migration_path):
            console.print(
                f"\n[bold]Step {i + 1}/{len(migration_path)}: {migrator.from_version} → {migrator.to_version}[/bold]"
            )

            # Create backup for this step
            backup_path = (
                backup_dir
                / f"{file_path.stem}_backup_{migrator.from_version}_to_{migrator.to_version}{file_path.suffix}"
            )

            # Perform migration
            result = migrator.migrate(file_path, backup_path, console, verbose)
            results.append(result)

            if not result.success:
                console.print(f"[red]✗ Migration failed: {result.error_message}[/red]")
                # Rollback all previous migrations
                self._rollback_migrations(file_path, results, console)
                return results

            # Verify migration
            if not migrator.verify_migration(file_path, console):
                console.print("[red]✗ Migration verification failed[/red]")
                result.success = False
                result.error_message = "Verification failed"
                # Rollback all previous migrations
                self._rollback_migrations(file_path, results, console)
                return results

            console.print(f"[green]✓ Migration successful ({result.changes_made} changes)[/green]")

        console.print(f"\n[green]✓ All migrations completed successfully![/green]")
        return results

    def _rollback_migrations(self, file_path: Path, results: List[MigrationResult], console: Console):
        """Rollback migrations by restoring from the first backup."""
        console.print("[yellow]Rolling back migrations...[/yellow]")

        # Find the first successful backup
        for result in results:
            if result.backup_path and result.backup_path.exists():
                try:
                    shutil.copy2(result.backup_path, file_path)
                    console.print(f"[green]✓ Restored from backup: {result.backup_path}[/green]")
                    return
                except Exception as e:
                    console.print(f"[red]✗ Failed to restore backup: {e}[/red]")

        console.print("[red]✗ No valid backup found for rollback[/red]")


def detect_files_needing_migration(file_paths: List[Path], console: Console) -> List[FileVersionInfo]:
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
                needs_migration = detected_version != orchestrator.current_version

                file_infos.append(
                    FileVersionInfo(
                        file_path=file_path,
                        detected_version=detected_version,
                        needs_migration=needs_migration,
                        target_version=orchestrator.current_version,
                    )
                )

            except Exception as e:
                console.print(f"[red]Error analyzing {file_path}: {e}[/red]")

            progress.update(task, advance=1)

    return file_infos


def migrate(
    files: List[Path] = typer.Argument(..., help="MCAP files to migrate"),
    target_version: Optional[str] = typer.Option(None, "--target", "-t", help="Target version (default: latest)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be migrated without making changes"),
    force: bool = typer.Option(False, "--force", help="Force migration even if files appear up-to-date"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed migration information"),
    keep_backups: bool = typer.Option(True, "--keep-backups/--no-backups", help="Keep backup files after migration"),
) -> None:
    """
    Migrate MCAP files to the latest version with automatic version detection.

    This command automatically detects the version of MCAP files and applies
    sequential migrations to bring them up to the latest version. Each migration
    step is verified and can be rolled back on failure.

    Examples:
        owl mcap migrate *.mcap                      # Migrate all MCAP files (shell expands glob)
        owl mcap migrate data/**/*.mcap              # Migrate all MCAP files recursively
        owl mcap migrate recording.mcap --dry-run    # Preview migration for single file
        owl mcap migrate *.mcap --target 0.3.2       # Migrate to specific version
    """
    console = Console()
    orchestrator = MigrationOrchestrator()

    if target_version is None:
        target_version = orchestrator.current_version

    console.print("[bold blue]MCAP Migration Tool[/bold blue]")
    console.print(f"Target version: {target_version}")
    console.print(f"Files to process: {len(files)}")

    if dry_run:
        console.print("[yellow]DRY RUN MODE - No files will be modified[/yellow]")

    # Detect files needing migration
    file_infos = detect_files_needing_migration(files, console)

    if not file_infos:
        console.print("[yellow]No files found matching the pattern[/yellow]")
        return

    # Filter files that need migration
    files_needing_migration = [info for info in file_infos if info.needs_migration or force]

    if not files_needing_migration:
        console.print("[green]✓ All files are already at the target version[/green]")
        return

    # Show summary table
    console.print(f"\n[bold]Migration Summary[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("Current Version", style="yellow")
    table.add_column("Target Version", style="green")
    table.add_column("Status", justify="center")

    for info in file_infos:
        status = "🔄" if (info.needs_migration or force) else "✅"
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
            results = orchestrator.migrate_file(info.file_path, target_version, console, verbose)

            if results and all(r.success for r in results):
                successful_migrations += 1
                # Collect backup paths
                backup_paths.extend([r.backup_path for r in results if r.backup_path])
            else:
                failed_migrations += 1

        except Exception as e:
            console.print(f"[red]✗ Unexpected error: {e}[/red]")
            failed_migrations += 1

    # Final summary
    console.print(f"\n[bold]Migration Complete[/bold]")
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
        console.print(f"\n[blue]Backup files saved in .mcap_migration_backups directories[/blue]")

    if failed_migrations > 0:
        raise typer.Exit(1)
