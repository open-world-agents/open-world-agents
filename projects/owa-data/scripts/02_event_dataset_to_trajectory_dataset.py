#!/usr/bin/env python3
"""
02_event_dataset_to_trajectory_dataset.py

Convert event-per-row dataset (output of 01_raw_events_to_event_dataset.py) into a trajectory dataset format.

Usage (CLI):
    python 02_event_dataset_to_trajectory_dataset.py \
        --input_dir /path/to/input_event_dataset \
        --output_dir /path/to/output_trajectory_dataset \
        [--fps 10] \
        [--keep_topic screen --keep_topic keyboard --keep_topic mouse]

- Aggregates events into ticks at the specified FPS.
- Keeps only specified topics (drops all others if not specified).
- Each output row contains: file_path, tick_idx, timestamp_ns, state, action.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from datasets import Dataset, Features, Value, load_from_disk
from tqdm import tqdm

app = typer.Typer(add_completion=False)


def aggregate_events_to_trajectory_ticks(
    events: List[Dict[str, Any]],
    fps: float,
    keep_topics: List[str],
) -> List[Dict[str, Any]]:
    """
    Aggregate events into trajectory ticks at the specified FPS.
    Args:
        events: List of event dicts (from input event dataset).
        fps: Global FPS for ticks.
        keep_topics: List of topics to keep.
    Returns:
        List of dicts, each representing a trajectory tick with state and action.
    """
    if not events:
        return []

    # Filter events by topic
    if keep_topics:
        events = [e for e in events if e["topic"] in keep_topics]

    if not events:
        return []

    # Sort by timestamp
    events.sort(key=lambda e: e["timestamp_ns"])

    # Find min/max timestamp
    min_ts = events[0]["timestamp_ns"]
    max_ts = events[-1]["timestamp_ns"]
    tick_interval_ns = int(1e9 / fps)

    # Calculate total number of ticks for progress tracking
    total_ticks = int((max_ts - min_ts) / tick_interval_ns) + 1

    ticks = []
    tick_idx = 0
    tick_start = min_ts
    tick_end = tick_start + tick_interval_ns
    event_idx = 0
    last_screen = None

    # Add progress bar for tick generation (only for large datasets)
    tick_pbar = tqdm(total=total_ticks, desc="Generating ticks", leave=False, disable=total_ticks < 100)

    while tick_start <= max_ts:
        # Aggregate actions in this tick
        actions = []
        # Find all events in [tick_start, tick_end)
        while event_idx < len(events) and events[event_idx]["timestamp_ns"] < tick_end:
            ev = events[event_idx]
            if ev["topic"].startswith("screen"):
                last_screen = ev  # Use latest screen as state
            elif ev["topic"].startswith("keyboard") or ev["topic"].startswith("mouse"):
                actions.append(ev)
            event_idx += 1

        # Compose tick
        tick = {
            "file_path": events[0]["file_path"],
            "tick_idx": tick_idx,
            "timestamp_ns": tick_start,
            "state": last_screen["msg"] if last_screen else None,
            "action": [a["msg"] for a in actions],
        }
        ticks.append(tick)
        tick_idx += 1
        tick_start = tick_end
        tick_end += tick_interval_ns

        # Update progress
        tick_pbar.update(1)

    tick_pbar.close()
    return ticks


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
        help="Output trajectory dataset directory",
    ),
    fps: float = typer.Option(10.0, "--fps", help="Global FPS for ticks (default: 10)"),
    keep_topic: Optional[List[str]] = typer.Option(
        None, "--keep_topic", help="Topic to keep (repeatable). If not specified, all topics are dropped."
    ),
):
    """
    Convert event-per-row dataset to trajectory dataset format with state/action per tick.
    """
    start_time = time.time()
    typer.echo(f"Loading event dataset from {input_dir} ...")
    ds_dict = load_from_disk(str(input_dir))
    # Support both DatasetDict and Dataset
    if hasattr(ds_dict, "keys"):
        splits = list(ds_dict.keys())
    else:
        splits = [None]
    for split in splits:
        if split:
            ds = ds_dict[split]
        else:
            ds = ds_dict
        # Group by file_path more efficiently
        typer.echo(f"Analyzing {len(ds)} events to group by file...")
        file_paths = sorted(set(ds["file_path"]))  # Sort for consistent ordering
        all_trajectory_ticks = []

        typer.echo(f"Found {len(file_paths)} unique files to process")

        # Create a progress bar for files
        file_pbar = tqdm(file_paths, desc=f"Processing {split or 'dataset'} files")
        for fp in file_pbar:
            file_pbar.set_postfix({"current_file": Path(fp).name})

            # Get all events for this file - use indices for better performance on large datasets
            if len(ds) > 10000:
                # For very large datasets, use indices instead of filter
                indices = [i for i, path in enumerate(ds["file_path"]) if path == fp]
                file_ds = ds.select(indices)
            else:
                # For smaller datasets, filter is fine
                file_ds = ds.filter(lambda example: example["file_path"] == fp)

            if len(file_ds) > 1000:
                typer.echo(f"  Extracting {len(file_ds)} events from large file...")

            # Convert to list of dicts using HuggingFace's efficient methods
            # This is much faster than individual indexing
            try:
                # Use to_dict() method which is optimized for batch conversion
                events_dict = file_ds.to_dict()
                events = []

                # Convert columnar format to row format efficiently
                num_events = len(events_dict[list(events_dict.keys())[0]])

                if len(file_ds) > 1000:
                    event_range = tqdm(range(num_events), desc="Converting to rows", leave=False)
                else:
                    event_range = range(num_events)

                for i in event_range:
                    event = {k: v[i] for k, v in events_dict.items()}
                    events.append(event)

            except Exception as e:
                # Fallback to slower method if to_dict() fails
                typer.echo(f"  Warning: Fast conversion failed ({e}), using slower method...")
                events = []
                for i in tqdm(range(len(file_ds)), desc="Extracting events", leave=False):
                    events.append({k: file_ds[k][i] for k in file_ds.column_names})

            trajectory_ticks = aggregate_events_to_trajectory_ticks(events, fps, keep_topic or [])
            all_trajectory_ticks.extend(trajectory_ticks)

            # Update file progress with tick count
            file_pbar.set_postfix(
                {"current_file": Path(fp).name, "events": len(events), "ticks": len(trajectory_ticks)}
            )

        file_pbar.close()
        # Define features
        features = Features(
            {
                "file_path": Value("string"),
                "tick_idx": Value("int32"),
                "timestamp_ns": Value("int64"),
                "state": Value("binary"),
                "action": Value("binary"),
            }
        )
        # Convert state/action to binary (if not None)
        typer.echo(f"Converting {len(all_trajectory_ticks)} trajectory ticks to binary format...")
        for t in tqdm(all_trajectory_ticks, desc="Converting to binary", disable=len(all_trajectory_ticks) < 1000):
            if t["state"] is not None and not isinstance(t["state"], bytes):
                t["state"] = json.dumps([*map(bytes.decode, t["state"])]).encode("utf-8")
            if t["action"] is not None and not isinstance(t["action"], bytes):
                t["action"] = json.dumps([*map(bytes.decode, t["action"])]).encode("utf-8")
        trajectory_dataset = Dataset.from_list(all_trajectory_ticks, features=features)
        # Save
        out_path = output_dir / (split if split else "data")
        out_path.mkdir(parents=True, exist_ok=True)
        trajectory_dataset.save_to_disk(str(out_path))

        # Calculate and display timing information
        elapsed_time = time.time() - start_time
        typer.echo(f"Saved {len(trajectory_dataset)} trajectory ticks to {out_path}")
        typer.echo(f"Processing completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.1f} minutes)")


if __name__ == "__main__":
    app()
