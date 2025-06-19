#!/usr/bin/env python3
"""
Example script demonstrating the MCAP migration system.

This script shows how to use the new migration system to automatically
detect and migrate MCAP files to the latest version.
"""

from pathlib import Path

from rich.console import Console

from owa.cli.mcap.migrate import MigrationOrchestrator, detect_files_needing_migration


def main():
    """Demonstrate the migration system."""
    console = Console()

    console.print("[bold blue]MCAP Migration System Demo[/bold blue]")
    console.print()

    # Create orchestrator
    orchestrator = MigrationOrchestrator()

    console.print(f"Current target version: {orchestrator.current_version}")
    console.print(f"Available migrators: {len(orchestrator.migrators)}")

    for migrator in orchestrator.migrators:
        console.print(f"  - {migrator.from_version} → {migrator.to_version}")

    console.print()

    # Example 1: Show migration paths
    console.print("[bold]Example Migration Paths:[/bold]")

    test_paths = [
        ("legacy", "0.4.0"),
        ("0.3.0", "0.4.0"),
        ("0.3.2", "0.4.0"),
        ("0.4.0", "0.4.0"),
    ]

    for from_version, to_version in test_paths:
        try:
            path = orchestrator.get_migration_path(from_version, to_version)
            if path:
                path_str = " → ".join([m.from_version for m in path] + [to_version])
                console.print(f"  {from_version} to {to_version}: {path_str}")
            else:
                console.print(f"  {from_version} to {to_version}: No migration needed")
        except ValueError as e:
            console.print(f"  {from_version} to {to_version}: [red]{e}[/red]")

    console.print()

    # Example 2: Detect files (if any exist)
    console.print("[bold]File Detection Example:[/bold]")

    # Look for MCAP files in current directory
    import glob

    mcap_files = [Path(f) for f in glob.glob("*.mcap")]
    file_infos = detect_files_needing_migration(mcap_files, console)

    if file_infos:
        console.print(f"Found {len(file_infos)} MCAP files:")
        for info in file_infos:
            status = "🔄 Needs migration" if info.needs_migration else "✅ Up to date"
            console.print(f"  {info.file_path.name}: {info.detected_version} → {info.target_version} ({status})")
    else:
        console.print("No MCAP files found in current directory")

    console.print()

    # Example 3: Show how to use the CLI command
    console.print("[bold]CLI Usage Examples:[/bold]")
    console.print("  owl mcap migrate *.mcap                      # Migrate all MCAP files (shell expands)")
    console.print("  owl mcap migrate data/**/*.mcap              # Migrate recursively (shell expands)")
    console.print("  owl mcap migrate recording.mcap --dry-run    # Preview migration")
    console.print("  owl mcap migrate *.mcap --target 0.3.2       # Migrate to specific version")
    console.print("  owl mcap migrate *.mcap --verbose            # Show detailed output")
    console.print("  owl mcap migrate *.mcap --no-backups         # Don't keep backup files")


if __name__ == "__main__":
    main()
