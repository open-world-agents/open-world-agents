#!/usr/bin/env python3
"""Convert event dataset to binned dataset format."""

from pathlib import Path
from typing import Any, Dict, List

import typer
from datasets import Dataset as HFDataset
from datasets import Features, Sequence, Value
from loguru import logger
from tqdm import tqdm

from owa.data.datasets import Dataset, DatasetConfig, DatasetDict, DatasetStage, load_from_disk

# Re-enable logging for owa.data
logger.enable("owa.data")

app = typer.Typer(add_completion=False)


def aggregate_events_to_bins(
    events: List[Dict[str, Any]], fps: float, filter_empty_actions: bool = False
) -> List[Dict[str, Any]]:
    """Aggregate events into time bins at the specified FPS."""
    if not events:
        return []

    events.sort(key=lambda e: e["timestamp_ns"])
    min_ts, max_ts = events[0]["timestamp_ns"], events[-1]["timestamp_ns"]
    bin_interval_ns = int(1e9 / fps)

    bins = []
    bin_idx = 0
    bin_start = min_ts
    event_idx = 0
    last_screen = None

    while bin_start <= max_ts:
        bin_end = bin_start + bin_interval_ns
        actions = []

        # Process events in this bin
        while event_idx < len(events) and events[event_idx]["timestamp_ns"] < bin_end:
            ev = events[event_idx]
            if ev["topic"].startswith("screen"):
                last_screen = ev
            elif ev["topic"].startswith("keyboard") or ev["topic"].startswith("mouse/raw"):
                actions.append(ev["mcap_message"])
            event_idx += 1

        # Create bin data
        bin_data = {
            "episode_path": events[0]["episode_path"],
            "bin_idx": bin_idx,
            "timestamp_ns": bin_start,
            "state": [last_screen["mcap_message"]] if last_screen else [],
            "actions": actions,
        }

        if not filter_empty_actions or actions:
            bins.append(bin_data)

        bin_idx += 1
        bin_start = bin_end

    return bins


@app.command()
def main(
    input_dir: Path = typer.Option(..., "--input-dir", help="Input event dataset directory"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Output binned dataset directory"),
    fps: float = typer.Option(10.0, "--fps", help="Global FPS for bins"),
    filter_empty_actions: bool = typer.Option(False, "--filter-empty-actions", help="Filter out bins with no actions"),
):
    """Convert event dataset to binned dataset format."""
    print(f"Loading from: {input_dir}")
    print(f"Target FPS: {fps}")
    if filter_empty_actions:
        print("Filter empty actions: ENABLED")
    else:
        print("Filter empty actions: DISABLED")

    # Load dataset
    ds_dict = load_from_disk(str(input_dir))

    # Get config from first dataset if available
    if isinstance(ds_dict, DatasetDict):
        print(f"Loaded DatasetDict with splits: {list(ds_dict.keys())}")
        event_config = next(iter(ds_dict.values())).owa_config
        splits = list(ds_dict.keys())
    else:
        print("Loaded single Dataset")
        event_config = ds_dict.owa_config
        splits = [None]

    # Create binned config
    binned_config = DatasetConfig(
        stage=DatasetStage.BINNED,
        mcap_root_directory=event_config.mcap_root_directory if event_config else str(input_dir),
    )

    processed_datasets = {}

    for split in splits:
        ds = ds_dict[split] if split else ds_dict
        print(f"Processing {len(ds):,} events from {split or 'dataset'}")

        all_binned_data = []

        # Process dataset sequentially, grouping by episode_path
        current_episode_events = []
        current_episode_path = None
        processed_episodes = 0

        print("Processing events sequentially...")

        with tqdm(total=len(ds), desc=f"Processing {split or 'dataset'} events") as pbar:
            for i in range(len(ds)):
                event = ds[i]
                episode_path = event["episode_path"]

                # If we encounter a new episode path, process the previous episode
                if current_episode_path is not None and episode_path != current_episode_path:
                    if current_episode_events:
                        binned_data = aggregate_events_to_bins(current_episode_events, fps, filter_empty_actions)
                        all_binned_data.extend(binned_data)
                        processed_episodes += 1
                    current_episode_events = []

                # Add current event to the episode
                current_episode_path = episode_path
                current_episode_events.append(event)
                pbar.update(1)

            # Process the last episode
            if current_episode_events:
                binned_data = aggregate_events_to_bins(current_episode_events, fps, filter_empty_actions)
                all_binned_data.extend(binned_data)
                processed_episodes += 1

        print(f"Processed {processed_episodes} episode files")
        # Create dataset
        features = Features(
            {
                "episode_path": Value("string"),
                "bin_idx": Value("int32"),
                "timestamp_ns": Value("int64"),
                "state": Sequence(feature=Value("binary"), length=-1),
                "actions": Sequence(feature=Value("binary"), length=-1),
            }
        )

        print(f"Creating dataset from {len(all_binned_data):,} binned entries...")
        hf_dataset = HFDataset.from_list(all_binned_data, features=features)

        binned_dataset = Dataset(
            arrow_table=hf_dataset.data,
            info=hf_dataset.info,
            split=hf_dataset.split,
            indices_table=hf_dataset._indices,
            fingerprint=hf_dataset._fingerprint,
            owa_config=binned_config,
        )

        split_name = split if split else "train"
        processed_datasets[split_name] = binned_dataset
        print(f"Created {len(binned_dataset):,} binned entries for {split_name} split")

    # Save dataset
    final_dataset = (
        DatasetDict(processed_datasets) if len(processed_datasets) > 1 else list(processed_datasets.values())[0]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_dir}")
    final_dataset.save_to_disk(str(output_dir))

    if len(processed_datasets) > 1:
        total_entries = sum(len(ds) for ds in processed_datasets.values())
        print(f"Saved {total_entries:,} total binned entries")
        for split_name, ds in processed_datasets.items():
            print(f"  {split_name}: {len(ds):,} entries")
    else:
        split_name = list(processed_datasets.keys())[0]
        ds = list(processed_datasets.values())[0]
        print(f"Saved {len(ds):,} binned entries ({split_name})")


if __name__ == "__main__":
    app()
