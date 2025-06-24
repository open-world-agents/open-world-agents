#!/usr/bin/env python3
"""
02_event_dataset_to_binned_dataset.py

Convert event-per-row dataset (output of 01_raw_events_to_event_dataset.py) into a binned dataset format.

Usage (CLI):
    python 02_event_dataset_to_binned_dataset.py \
        --input_dir /path/to/input_event_dataset \
        --output_dir /path/to/output_binned_dataset \
        [--fps 10]

- Bins events into fixed-rate time intervals at the specified FPS.
- Each output row contains: file_path, bin_idx, timestamp_ns, state, actions.
"""

import time
from pathlib import Path
from typing import Any, Dict, List

import typer
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from tqdm import tqdm

app = typer.Typer(add_completion=False)


def aggregate_events_to_bins(
    events: List[Dict[str, Any]],
    fps: float,
) -> List[Dict[str, Any]]:
    """
    Aggregate events into time bins at the specified FPS.
    Args:
        events: List of event dicts (from input event dataset).
        fps: Global FPS for bins.
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
        "--input_dir",
        "-i",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Input event dataset directory (output of 01_raw_events_to_event_dataset.py)",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output_dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Output binned dataset directory",
    ),
    fps: float = typer.Option(10.0, "--fps", help="Global FPS for bins (default: 10)"),
):
    """
    Convert event-per-row dataset to binned dataset format with state/actions per bin.
    """
    start_time = time.time()
    typer.echo(f"Loading event dataset from {input_dir} ...")
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
        typer.echo(f"Analyzing {len(ds)} events to group by file...")
        file_paths = sorted(set(ds["file_path"]))  # Sort for consistent ordering
        all_binned_data = []

        typer.echo(f"Found {len(file_paths)} unique files to process")

        # Create a progress bar for files
        file_pbar = tqdm(file_paths, desc=f"Processing {split or 'dataset'} files")
        for fp in file_pbar:
            file_pbar.set_postfix({"current_file": Path(fp).name})

            # Get all events for this file
            file_ds = ds.filter(lambda example: example["file_path"] == fp)

            # Convert to list of dicts
            events = []
            for i in range(len(file_ds)):
                event = {k: file_ds[k][i] for k in file_ds.column_names}
                events.append(event)

            binned_data = aggregate_events_to_bins(events, fps)
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
        typer.echo(f"Creating dataset from {len(all_binned_data)} binned entries...")
        binned_dataset = Dataset.from_list(all_binned_data, features=features)

        # Store the dataset for this split
        split_name = split if split else "train"  # Default to "train" if no split
        processed_datasets[split_name] = binned_dataset

        typer.echo(f"Processed {len(binned_dataset)} binned entries for {split_name} split")

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
    final_dataset.save_to_disk(str(output_dir))

    # Calculate and display timing information
    elapsed_time = time.time() - start_time
    if len(processed_datasets) > 1:
        total_entries = sum(len(ds) for ds in processed_datasets.values())
        typer.echo(f"Saved DatasetDict with {total_entries} total binned entries to {output_dir}")
        for split_name, ds in processed_datasets.items():
            typer.echo(f"  {split_name}: {len(ds)} entries")
    else:
        split_name = list(processed_datasets.keys())[0]
        ds = list(processed_datasets.values())[0]
        typer.echo(f"Saved {len(ds)} binned entries ({split_name}) to {output_dir}")

    typer.echo(f"Processing completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.1f} minutes)")


if __name__ == "__main__":
    app()
