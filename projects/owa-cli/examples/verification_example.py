#!/usr/bin/env python3
"""
Example demonstrating the flexible MCAP migration verification system.

This example shows how to use the individual verification functions
and the configurable verify_migration_integrity function.
"""

from pathlib import Path
from rich.console import Console

from owa.cli.mcap.migrators.base import (
    FileStats,
    verify_message_count,
    verify_file_size,
    verify_topics_preserved,
    verify_migration_integrity
)


def example_individual_verifications():
    """Example of using individual verification functions."""
    console = Console()
    
    # Example file statistics
    migrated_stats = FileStats(
        message_count=1000,
        file_size=50000,  # 50KB
        topics={"keyboard/events", "mouse/events", "screen/capture"},
        schemas={"desktop/KeyboardEvent", "desktop/MouseEvent", "desktop/ScreenCaptured"}
    )
    
    backup_stats = FileStats(
        message_count=1000,
        file_size=48000,  # 48KB (4% smaller)
        topics={"keyboard/events", "mouse/events", "screen/capture"},
        schemas={"owa.env.desktop.msg.KeyboardEvent", "owa.env.desktop.msg.MouseEvent", "owa.env.gst.msg.ScreenEmitted"}
    )
    
    console.print("[bold blue]Individual Verification Examples[/bold blue]")
    
    # Test message count verification
    console.print("\n1. Message Count Verification:")
    result = verify_message_count(migrated_stats, backup_stats, console)
    console.print(f"   Result: {'✓ PASS' if result else '✗ FAIL'}")
    
    # Test file size verification
    console.print("\n2. File Size Verification (10% tolerance):")
    result = verify_file_size(migrated_stats, backup_stats, console, tolerance_percent=10.0)
    console.print(f"   Result: {'✓ PASS' if result else '✗ FAIL'}")
    
    # Test file size verification with stricter tolerance
    console.print("\n3. File Size Verification (2% tolerance):")
    result = verify_file_size(migrated_stats, backup_stats, console, tolerance_percent=2.0)
    console.print(f"   Result: {'✓ PASS' if result else '✗ FAIL'}")
    
    # Test topic preservation
    console.print("\n4. Topic Preservation Verification:")
    result = verify_topics_preserved(migrated_stats, backup_stats, console)
    console.print(f"   Result: {'✓ PASS' if result else '✗ FAIL'}")


def example_configurable_verification():
    """Example of using configurable verification."""
    console = Console()
    
    # Create example files (these would be real MCAP files in practice)
    migrated_file = Path("example_migrated.mcap")
    backup_file = Path("example_backup.mcap")
    
    console.print("\n[bold blue]Configurable Verification Examples[/bold blue]")
    console.print("(Note: These examples use mock files and would fail in practice)")
    
    # Example 1: Full verification (default)
    console.print("\n1. Full Verification (all checks enabled):")
    result = verify_migration_integrity(migrated_file, backup_file, console)
    console.print(f"   Result: {'✓ PASS' if result else '✗ FAIL'}")
    
    # Example 2: Only message count verification
    console.print("\n2. Message Count Only:")
    result = verify_migration_integrity(
        migrated_file, backup_file, console,
        check_message_count=True,
        check_file_size=False,
        check_topics=False
    )
    console.print(f"   Result: {'✓ PASS' if result else '✗ FAIL'}")
    
    # Example 3: Only file size verification with custom tolerance
    console.print("\n3. File Size Only (5% tolerance):")
    result = verify_migration_integrity(
        migrated_file, backup_file, console,
        check_message_count=False,
        check_file_size=True,
        check_topics=False,
        size_tolerance_percent=5.0
    )
    console.print(f"   Result: {'✓ PASS' if result else '✗ FAIL'}")
    
    # Example 4: Skip all integrity checks (minimal verification)
    console.print("\n4. No Integrity Checks (minimal verification):")
    result = verify_migration_integrity(
        migrated_file, backup_file, console,
        check_message_count=False,
        check_file_size=False,
        check_topics=False
    )
    console.print(f"   Result: {'✓ PASS' if result else '✗ FAIL'}")


def example_migrator_usage():
    """Example of how migrators can use the verification system."""
    console = Console()
    
    console.print("\n[bold blue]Migrator Usage Example[/bold blue]")
    console.print("This shows how a migrator might use different verification strategies:")
    
    # Example migrator verification logic
    migrated_file = Path("example.mcap")
    backup_file = Path("example.mcap.backup")
    
    console.print("\n1. Schema Migration Verification:")
    console.print("   - First check: Legacy schemas removed")
    console.print("   - Then: Full integrity verification if backup available")
    
    console.print("\n2. Field Migration Verification:")
    console.print("   - First check: Specific field changes applied")
    console.print("   - Then: Message count + topics (skip file size for field changes)")
    
    console.print("\n3. Performance-Critical Migration:")
    console.print("   - Only message count verification (fastest)")
    console.print("   - Skip file size and topic checks for speed")


if __name__ == "__main__":
    console = Console()
    console.print("[bold green]MCAP Migration Verification System Examples[/bold green]")
    
    example_individual_verifications()
    example_configurable_verification()
    example_migrator_usage()
    
    console.print("\n[bold green]Examples completed![/bold green]")
    console.print("See the source code for implementation details.")
