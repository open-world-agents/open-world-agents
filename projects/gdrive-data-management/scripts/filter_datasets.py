#!/usr/bin/env python3
"""
Filter OWA game datasets based on version and migrate to new directory.

This script:
1. Checks version of each MCAP file
2. Copies files with version >= 0.5.5 to filtered directory
3. Migrates files to target version (0.5.5)
4. Copies paired MKV files
"""

import shutil
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Import functions from reference files
import sys
sys.path.append('reference')

from mcap_owa.highlevel import OWAMcapReader


def get_file_version(mcap_path: Path) -> Optional[str]:
    """Get the file version from MCAP file."""
    try:
        with OWAMcapReader(mcap_path) as reader:
            return reader.file_version
    except Exception as e:
        print(f"Error reading version from {mcap_path}: {e}")
        return None


def version_compare(version1: str, version2: str) -> int:
    """
    Compare two version strings.
    Returns: -1 if version1 < version2, 0 if equal, 1 if version1 > version2
    """
    def normalize_version(v):
        # Remove 'v' prefix if present and split by dots
        v = v.lstrip('v')
        return [int(x) for x in v.split('.')]
    
    try:
        v1_parts = normalize_version(version1)
        v2_parts = normalize_version(version2)
        
        # Pad shorter version with zeros
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        for i in range(max_len):
            if v1_parts[i] < v2_parts[i]:
                return -1
            elif v1_parts[i] > v2_parts[i]:
                return 1
        return 0
    except (ValueError, AttributeError):
        # If version parsing fails, assume incompatible
        return -1


def is_version_compatible(file_version: str, min_version: str = "0.5.5") -> bool:
    """Check if file version meets minimum requirement."""
    if not file_version:
        return False
    return version_compare(file_version, min_version) >= 0



def find_paired_mkv(mcap_file: Path) -> Optional[Path]:
    """Find the paired MKV file for a given MCAP file."""
    # Remove .mcap extension and add .mkv
    mkv_file = mcap_file.with_suffix('.mkv')

    if mkv_file.exists():
        return mkv_file

    # Try with _mig removed if present
    if '_mig' in mcap_file.stem:
        base_name = mcap_file.stem.replace('_mig', '')
        mkv_file = mcap_file.parent / f"{base_name}.mkv"
        if mkv_file.exists():
            return mkv_file

    return None


def copy_with_structure(source_file: Path, source_root: Path, dest_root: Path) -> Path:
    """Copy file maintaining directory structure relative to source root."""
    relative_path = source_file.relative_to(source_root)
    dest_file = dest_root / relative_path
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    return dest_file


def filter_datasets(
    source_root: str = "/mnt/raid12/datasets/owa_game_dataset",
    dest_root: str = "/mnt/raid12/datasets/owa_game_dataset_filtered",
    min_version: str = "0.5.5",
    target_version: str = "0.5.5"
):
    """Main function to filter and migrate datasets."""
    console = Console()
    source_path = Path(source_root)
    dest_path = Path(dest_root)
    
    if not source_path.exists():
        console.print(f"[red]Source directory not found: {source_path}[/red]")
        return
    
    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Destination directory: {dest_path}[/green]")
    
    # Find all MCAP files at any depth within user directories
    mcap_files = list(source_path.glob("*/**/*.mcap"))
    console.print(f"[blue]Found {len(mcap_files)} MCAP files to check[/blue]")
    
    if not mcap_files:
        console.print("[yellow]No MCAP files found![/yellow]")
        return
    
    compatible_files = []
    incompatible_files = []
    
    # First pass: check versions
    console.print(f"[blue]Checking file versions (minimum: {min_version})[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        
        version_task = progress.add_task("Checking versions...", total=len(mcap_files))
        
        for mcap_file in mcap_files:
            try:
                file_version = get_file_version(mcap_file)
                
                if file_version and is_version_compatible(file_version, min_version):
                    compatible_files.append((mcap_file, file_version))
                else:
                    incompatible_files.append((mcap_file, file_version))
                
                progress.advance(version_task)
                
            except Exception as e:
                console.print(f"[red]Error checking {mcap_file}: {e}[/red]")
                incompatible_files.append((mcap_file, None))
                progress.advance(version_task)
    
    console.print(f"[green]Compatible files: {len(compatible_files)}[/green]")
    console.print(f"[yellow]Incompatible files: {len(incompatible_files)}[/yellow]")
    
    if not compatible_files:
        console.print("[yellow]No compatible files found to migrate![/yellow]")
        return
    
    # Second pass: copy files to filtered directory
    console.print("[blue]Copying compatible files to filtered directory[/blue]")

    successful_copies = 0
    failed_copies = 0
    copied_mkvs = 0
    missing_mkvs = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        copy_task = progress.add_task("Copying files...", total=len(compatible_files))
        
        for mcap_file, file_version in compatible_files:
            try:
                # Determine source file (prefer _mig.mcap if available, else original)
                source_mcap = mcap_file
                if '_mig' not in mcap_file.stem:
                    mig_path = mcap_file.parent / f"{mcap_file.stem}_mig.mcap"
                    if mig_path.exists():
                        source_mcap = mig_path

                # Create destination path (always without _mig suffix)
                base_name = source_mcap.stem.replace('_mig', '')
                dest_mcap = copy_with_structure(
                    source_mcap.parent / f"{base_name}.mcap",
                    source_path,
                    dest_path
                )

                progress.update(copy_task, description=f"Copying {source_mcap.name}")

                # Copy MCAP file (no migration needed, already done in analyze step)
                try:
                    shutil.copy2(source_mcap, dest_mcap)
                    successful_copies += 1

                    # Find and copy paired MKV file
                    mkv_file = find_paired_mkv(mcap_file)  # Always look for original MKV
                    if mkv_file:
                        dest_mkv = copy_with_structure(mkv_file, source_path, dest_path)
                        shutil.copy2(mkv_file, dest_mkv)
                        copied_mkvs += 1
                    else:
                        missing_mkvs += 1
                        console.print(f"[yellow]No MKV found for {mcap_file.name}[/yellow]")
                except Exception as e:
                    failed_copies += 1
                    console.print(f"[red]Copy failed for {source_mcap.name}: {e}[/red]")

                progress.advance(copy_task)
                
            except Exception as e:
                console.print(f"[red]Error processing {mcap_file}: {e}[/red]")
                failed_copies += 1
                progress.advance(copy_task)

    # Final summary
    console.print("\n[bold]Copy Summary[/bold]")
    console.print(f"[green]MCAP files copied successfully: {successful_copies}[/green]")
    console.print(f"[red]MCAP files failed: {failed_copies}[/red]")
    console.print(f"[green]MKV files copied: {copied_mkvs}[/green]")
    console.print(f"[yellow]MKV files missing: {missing_mkvs}[/yellow]")

    if successful_copies > 0:
        console.print(f"\n[green]✓ Filtering complete! {successful_copies} datasets copied to {dest_path}[/green]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter and migrate OWA game datasets")
    parser.add_argument("--source-root", default="/mnt/raid12/datasets/owa_game_dataset",
                       help="Source dataset directory")
    parser.add_argument("--dest-root", default="/mnt/raid12/datasets/owa_game_dataset_filtered",
                       help="Destination directory for filtered datasets")
    parser.add_argument("--min-version", default="0.5.5",
                       help="Minimum version requirement")
    parser.add_argument("--target-version", default="0.5.5",
                       help="Target version for migration")
    
    args = parser.parse_args()
    filter_datasets(args.source_root, args.dest_root, args.min_version, args.target_version)
