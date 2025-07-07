#!/usr/bin/env python3
"""
02_event_dataset_to_binned_dataset.py

Convert event-per-row dataset (output of 01_raw_events_to_event_dataset.py) into a binned dataset format.

Usage (CLI):
    python 02_event_dataset_to_binned_dataset.py \
        --input-dir /path/to/input_event_dataset \
        --output-dir /path/to/output_binned_dataset \
        [--fps 10]

- Bins events into fixed-rate time intervals at the specified FPS.
- Each output row contains: file_path, bin_idx, timestamp_ns, state, actions.
"""

import time
from pathlib import Path
from typing import Any, Dict, List

import typer
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

app = typer.Typer(add_completion=False)
console = Console()


def aggregate_events_to_bins(
    events: List[Dict[str, Any]],
    fps: float,
    filter_empty_actions: bool = False,
) -> List[Dict[str, Any]]:
    """
    Aggregate events into time bins at the specified FPS.
    Args:
        events: List of event dicts (from input event dataset).
        fps: Global FPS for bins.
        filter_empty_actions: If True, filter out bins with no actions.
    Returns:
        List of dicts, each representing a time bin with state and actions.
    """
    if not events:
        return []

    # Sort by timestamp
    events.sort(key=lambda e: e["timestamp_ns"])

    # Find min/max timestamp
    min_ts = events[0]["timestamp_ns"]
    max_ts = events[-1]["timestamp_ns"]
    bin_interval_ns = int(1e9 / fps)

    # Calculate total number of bins for progress tracking
    total_bins = int((max_ts - min_ts) / bin_interval_ns) + 1

    bins = []
    bin_idx = 0
    bin_start = min_ts
    bin_end = bin_start + bin_interval_ns
    event_idx = 0
    last_screen = None

    # Add progress bar for bin generation (only for large datasets)
    bin_pbar = tqdm(total=total_bins, desc="Generating bins", leave=False, disable=total_bins < 100)

    while bin_start <= max_ts:
        # Aggregate actions in this bin
        actions = []
        # Find all events in [bin_start, bin_end)
        while event_idx < len(events) and events[event_idx]["timestamp_ns"] < bin_end:
            ev = events[event_idx]
            if ev["topic"].startswith("screen"):
                last_screen = ev  # Use latest screen as state
            elif ev["topic"].startswith("keyboard") or ev["topic"].startswith("mouse"):
                actions.append(ev["mcap_message"])  # Store serialized McapMessage bytes
            event_idx += 1

        # Compose bin
        bin_data = {
            "file_path": events[0]["file_path"],
            "bin_idx": bin_idx,
            "timestamp_ns": bin_start,
            "state": [last_screen["mcap_message"]]
            if last_screen
            else [],  # Store as list of serialized McapMessage bytes
            "actions": actions,  # Store list of serialized McapMessage bytes
        }

        # Filter out bins with no actions if requested
        if not filter_empty_actions or len(actions) > 0:
            bins.append(bin_data)

        bin_idx += 1
        bin_start = bin_end
        bin_end += bin_interval_ns

        # Update progress
        bin_pbar.update(1)

    bin_pbar.close()
    return bins


@app.command()
def main(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        "-i",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Input event dataset directory (output of 01_raw_events_to_event_dataset.py)",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Output binned dataset directory",
    ),
    fps: float = typer.Option(10.0, "--fps", help="Global FPS for bins (default: 10)"),
    filter_empty_actions: bool = typer.Option(
        False,
        "--filter-empty-actions/--no-filter-empty-actions",
        help="Filter out bins with no actions (default: False)",
    ),
):
    """
    Convert event-per-row dataset to binned dataset format with state/actions per bin.
    """
    start_time = time.time()

    # Print header
    console.print(Panel.fit("🗂️ Event Dataset to Binned Dataset", style="bold blue"))

    console.print(f"[cyan]📁[/cyan] Loading from: {input_dir}")
    console.print(f"[cyan]⚡[/cyan] Target FPS: [bold]{fps}[/bold]")
    if filter_empty_actions:
        console.print(
            "[yellow]🔍[/yellow] Filter empty actions: [bold]ENABLED[/bold] - bins with no actions will be filtered out"
        )
    else:
        console.print("[cyan]🔍[/cyan] Filter empty actions: [bold]DISABLED[/bold] - all bins will be kept")
    ds_dict = load_from_disk(str(input_dir))
    # Support both DatasetDict and Dataset
    if hasattr(ds_dict, "keys"):
        splits = list(ds_dict.keys())
    else:
        splits = [None]

    # Store all processed datasets
    processed_datasets = {}

    for split in splits:
        if split:
            ds = ds_dict[split]
        else:
            ds = ds_dict
        # Group by file_path more efficiently
        console.print(f"[cyan]🔍[/cyan] Analyzing [bold]{len(ds):,}[/bold] events to group by file...")
        file_paths = sorted(set(ds["file_path"]))  # Sort for consistent ordering
        all_binned_data = []

        console.print(f"[green]📊[/green] Found [bold]{len(file_paths)}[/bold] files to process")

        # Create a progress bar for files
        file_pbar = tqdm(file_paths, desc=f"Processing {split or 'dataset'} files")
        for fp in file_pbar:
            file_pbar.set_postfix({"current_file": Path(fp).name})

            # Get all events for this file
            file_ds = ds.filter(lambda example: example["file_path"] == fp)

            # Convert to list of dicts
            events = []
            for i in range(len(file_ds)):
                event = file_ds[i]
                events.append(event)

            binned_data = aggregate_events_to_bins(events, fps, filter_empty_actions)
            all_binned_data.extend(binned_data)

            # Update file progress with bin count
            file_pbar.set_postfix({"current_file": Path(fp).name, "events": len(events), "bins": len(binned_data)})

        file_pbar.close()
        # Define features
        features = Features(
            {
                "file_path": Value("string"),
                "bin_idx": Value("int32"),
                "timestamp_ns": Value("int64"),
                "state": Sequence(feature=Value("binary"), length=-1),  # Sequence of serialized McapMessage bytes
                "actions": Sequence(feature=Value("binary"), length=-1),  # Sequence of serialized McapMessage bytes
            }
        )
        # McapMessage objects are already serialized as bytes from previous step
        console.print(f"[cyan]🔧[/cyan] Creating dataset from [bold]{len(all_binned_data):,}[/bold] binned entries...")
        binned_dataset = Dataset.from_list(all_binned_data, features=features)

        # Store the dataset for this split
        split_name = split if split else "train"  # Default to "train" if no split
        processed_datasets[split_name] = binned_dataset

        console.print(
            f"[green]✓[/green] Created [bold]{len(binned_dataset):,}[/bold] binned entries for [bold]{split_name}[/bold] split"
        )

    # Save all datasets as DatasetDict or single Dataset
    if len(processed_datasets) > 1:
        # Multiple splits - create DatasetDict
        from datasets import DatasetDict

        final_dataset = DatasetDict(processed_datasets)
    else:
        # Single split - save as Dataset
        final_dataset = list(processed_datasets.values())[0]

    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]💾[/cyan] Saving to {output_dir}")
    final_dataset.save_to_disk(str(output_dir))

    # Calculate and display timing information
    elapsed_time = time.time() - start_time
    if len(processed_datasets) > 1:
        total_entries = sum(len(ds) for ds in processed_datasets.values())
        console.print(f"[green]✓[/green] Saved [bold]{total_entries:,}[/bold] total binned entries")
        for split_name, ds in processed_datasets.items():
            console.print(f"  [cyan]•[/cyan] {split_name}: [bold]{len(ds):,}[/bold] entries")
    else:
        split_name = list(processed_datasets.keys())[0]
        ds = list(processed_datasets.values())[0]
        console.print(f"[green]✓[/green] Saved [bold]{len(ds):,}[/bold] binned entries ([bold]{split_name}[/bold])")

    console.print(
        f"[green]🎉[/green] Completed in [bold]{elapsed_time:.2f}s[/bold] ([bold]{elapsed_time / 60:.1f}min[/bold])"
    )


if __name__ == "__main__":
    app()
