#!/usr/bin/env python3
"""Process raw MCAP files to create event datasets."""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import typer
from datasets import Dataset as HFDataset
from datasets import DatasetDict, Features, Value
from datasets import DatasetInfo as HFDatasetInfo
from loguru import logger
from tqdm import tqdm

from mcap_owa.highlevel import OWAMcapReader
from owa.data.datasets import Dataset, DatasetConfig, DatasetStage
from owa.data.interval import Intervals
from owa.data.interval.selector import All
from owa.data.processing.resampler import EventResampler, create_resampler

# Re-enable logging for owa.data
logger.enable("owa.data")

app = typer.Typer(add_completion=False)


def parse_rate_argument(rate_args: List[str]) -> Dict[str, float]:
    """Parse CLI --rate arguments of the form "topic=Hz"."""
    rate_settings: Dict[str, float] = {}
    for arg in rate_args:
        if "=" not in arg:
            raise typer.BadParameter(f"Invalid rate argument '{arg}'. Expected format: topic=Hz")
        topic, rate_str = arg.split("=", maxsplit=1)
        try:
            rate = float(rate_str)
            if rate <= 0:
                raise ValueError("Rate must be positive")
        except ValueError as e:
            raise typer.BadParameter(f"Invalid rate value in '{arg}': {e}")
        rate_settings[topic] = rate
    return rate_settings


def process_raw_events_file(
    episode_path: str,
    rate_settings: Dict[str, float],
    keep_topics: Optional[List[str]] = None,
    mcap_root_directory: Optional[str] = None,
    time_shift_seconds: Optional[float] = None,
    action_topics: Optional[List[str]] = None,
) -> List[Dict]:
    """Process MCAP file with resampling."""
    events: List[Dict] = []
    interval_extractor = All()  # Select all intervals
    valid_intervals: Intervals = interval_extractor.extract_intervals(Path(episode_path))

    # Initialize resamplers for all topics
    resamplers: Dict[str, EventResampler] = {}
    for topic in keep_topics or []:
        rate_hz = rate_settings.get(topic, 0)  # 0 = no rate limit
        min_interval_ns = 0 if rate_hz == 0 else int((1.0 / rate_hz) * 1e9)
        resamplers[topic] = create_resampler(topic, min_interval_ns=min_interval_ns)

    with OWAMcapReader(Path(episode_path)) as reader:
        for interval in valid_intervals:
            for mcap_msg in reader.iter_messages(start_time=interval.start, end_time=interval.end, topics=keep_topics):
                topic = mcap_msg.topic

                # Process event through resampler
                resamplers[topic].add_event(mcap_msg)
                ready_events = resamplers[topic].pop_event()

                # Process all ready events
                for mcap_message_obj in ready_events:
                    # Serialize McapMessage to bytes using model_dump_json
                    mcap_message_bytes = mcap_message_obj.model_dump_json().encode("utf-8")

                    # Store relative path if mcap_root_directory is provided
                    stored_episode_path = episode_path
                    if mcap_root_directory:
                        stored_episode_path = Path(episode_path).relative_to(mcap_root_directory).as_posix()

                    events.append(
                        {
                            "episode_path": stored_episode_path,
                            "topic": topic,
                            "timestamp_ns": mcap_message_obj.timestamp,
                            "message_type": mcap_message_obj.message_type,
                            "mcap_message": mcap_message_bytes,  # Store serialized bytes
                        }
                    )

    if time_shift_seconds is None or not action_topics:
        return events

    action_topics_set = set(action_topics)
    time_shift_ns = int(time_shift_seconds * 1e9)

    for event in events:
        if event["topic"] in action_topics_set:
            event["timestamp_ns"] += time_shift_ns

    events.sort(key=lambda e: e["timestamp_ns"])
    return events


def generate_event_examples(
    episode_paths: List[str],
    rate_settings: Dict[str, float],
    keep_topics: Optional[List[str]] = None,
    num_workers: int = 4,
    mcap_root_directory: Optional[str] = None,
    time_shift_seconds: Optional[float] = None,
    action_topics: Optional[List[str]] = None,
):
    """
    Generator function that yields event examples by processing each raw events file
    in parallel using multiple processes.

    Args:
        episode_paths: List of MCAP file paths (strings).
        rate_settings: Mapping from topic to desired rate (Hz).
        keep_topics: Optional list of topics to keep. If None, all topics are kept.
        num_workers: Number of parallel worker processes.
        time_shift_seconds: Time shift in seconds to add to action topics.
        action_topics: List of topics to apply time shift to.

    Yields:
        Individual event dictionaries suitable for Hugging Face Dataset.
    """
    total_files = len(episode_paths)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {
            executor.submit(
                process_raw_events_file,
                fp,
                rate_settings,
                keep_topics,
                mcap_root_directory,
                time_shift_seconds,
                action_topics,
            ): fp
            for fp in episode_paths
        }
        with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
            for future in as_completed(future_to_path):
                fp = future_to_path[future]
                try:
                    events = future.result()
                    for event in events:
                        yield event
                except Exception as e:
                    # Log error but don't stop processing other files
                    print(f"⚠ Failed to process file {Path(fp).name}: {e}", file=sys.stderr)
                finally:
                    pbar.update(1)


def create_event_dataset(
    episode_paths: List[Path],
    rate_settings: Dict[str, float],
    keep_topics: Optional[List[str]] = None,
    num_workers: int = 4,
    split: str = "train",
    mcap_root_directory: Optional[str] = None,
    time_shift_seconds: Optional[float] = None,
    action_topics: Optional[List[str]] = None,
) -> Dataset:
    """
    Create a Hugging Face event dataset from the given MCAP file paths by streaming
    examples from a generator.

    Args:
        episode_paths: List of pathlib.Path objects pointing to MCAP files.
        rate_settings: Mapping from topic to rate (Hz) to apply drop-only downsampling.
        keep_topics: Optional list of topics to keep. If None, all topics are kept.
        num_workers: Number of worker processes for parallel file processing.
        time_shift_seconds: Time shift in seconds to add to action topics.
        action_topics: List of topics to apply time shift to.

    Returns:
        A Hugging Face Dataset containing the combined events.
    """
    episode_path_strs = [str(fp) for fp in episode_paths]

    features = Features(
        {
            "episode_path": Value("string"),
            "topic": Value("string"),
            "timestamp_ns": Value("int64"),
            "message_type": Value("string"),
            "mcap_message": Value("binary"),  # Use bytes serialization for McapMessage
        }
    )

    # Create HF Dataset first
    hf_dataset = HFDataset.from_generator(
        generate_event_examples,
        gen_kwargs={
            "episode_paths": episode_path_strs,
            "rate_settings": rate_settings,
            "keep_topics": keep_topics,
            "num_workers": num_workers,
            "mcap_root_directory": mcap_root_directory,
            "time_shift_seconds": time_shift_seconds,  # Add this parameter
            "action_topics": action_topics,
        },
        features=features,
        split=split,
    )
    info_to_update = HFDatasetInfo(
        description="",
        dataset_name="open-world-agents/goat",
        homepage="https://github.com/open-world-agents",
    )
    hf_dataset.info.update(info_to_update)

    # Convert to unified Dataset
    event_dataset = Dataset(
        arrow_table=hf_dataset.data,
        info=hf_dataset.info,
        split=hf_dataset.split,
        indices_table=hf_dataset._indices,
        fingerprint=hf_dataset._fingerprint,
        owa_config=DatasetConfig(
            stage=DatasetStage.EVENT,
            mcap_root_directory=mcap_root_directory,
        ),
    )

    return event_dataset


@app.command()
def main(
    train_dir: Path = typer.Option(..., "--train-dir", help="Directory containing MCAP files for training"),
    test_dir: Optional[Path] = typer.Option(None, "--test-dir", help="Directory containing MCAP files for testing"),
    test_percent: float = typer.Option(0.1, "--test_percent", help="Fraction of training files for test set"),
    max_test_files: int = typer.Option(32, "--max-test-files", help="Maximum number of test files"),
    rate: Optional[List[str]] = typer.Option(None, "--rate", help="Rate-limiting per topic in 'topic=Hz' format"),
    num_workers: int = typer.Option(4, "--num-workers", help="Number of parallel worker processes"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Directory to save the dataset"),
    keep_topic: Optional[List[str]] = typer.Option(
        None, "--keep-topic", help="Topics to keep (default: screen, keyboard, mouse/raw)"
    ),
    time_shift: Optional[float] = typer.Option(
        None, "--time-shift", help="Time shift in seconds to add to action topics"
    ),
    action_topic: Optional[List[str]] = typer.Option(
        None, "--action-topic", help="Topics to apply time shift (default: keyboard, mouse/raw)"
    ),
):
    """Generate event dataset from raw MCAP files."""

    # Validate test_percent
    if test_percent <= 0 or test_percent >= 1:
        raise typer.BadParameter("--test_percent must be between 0 and 1 (exclusive)")

    # Parse rate settings or use defaults
    rate_settings = parse_rate_argument(rate) if rate else {"mouse/raw": 20.0, "screen": 10.0}
    topics_to_keep = keep_topic if keep_topic else ["screen", "keyboard", "mouse/raw"]

    # Define action topics that should be time-shifted
    action_topics = action_topic if action_topic else ["keyboard", "mouse/raw"]

    # Gather MCAP files
    train_files = sorted(train_dir.glob("*.mcap"))
    if not train_files:
        raise typer.BadParameter(f"No MCAP files found in train-dir: {train_dir}")

    # Determine test files
    if test_dir:
        test_files = sorted(test_dir.glob("*.mcap"))
        if not test_files:
            raise typer.BadParameter(f"No MCAP files found in test-dir: {test_dir}")
        # Check for overlap
        train_set = set(str(p) for p in train_files)
        overlap = set(str(p) for p in test_files).intersection(train_set)
        if overlap:
            raise typer.BadParameter(f"Same files present in train-dir and test-dir: {len(overlap)} files")
    else:
        shuffled = train_files.copy()
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        shuffled_index = rng.permutation(len(shuffled))
        shuffled = [shuffled[i] for i in shuffled_index]
        test_count = min(max(1, int(len(shuffled) * test_percent)), max_test_files)
        test_files = shuffled[:test_count]
        train_files = shuffled[test_count:]

    print(f"Processing {len(train_files)} train files, {len(test_files)} test files with {num_workers} workers")
    if time_shift is not None and action_topics:
        print(f"Applying {time_shift}s time shift to action topics: {action_topics}")
    else:
        print("No time shift applied")

    # Confirm if no output directory
    if not output_dir:
        if not typer.confirm("No --output-dir given. Continue without saving to disk?", default=False):
            raise typer.Exit(1)

    # Create event datasets
    mcap_root_directory = str(train_dir)
    train_dataset = create_event_dataset(
        train_files,
        rate_settings,
        topics_to_keep,
        num_workers,
        "train",
        mcap_root_directory,
        time_shift,
        action_topics,
    )
    test_dataset = create_event_dataset(
        test_files, rate_settings, topics_to_keep, num_workers, "test", mcap_root_directory, time_shift, action_topics
    )

    # Combine into DatasetDict
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    print(f"Created {len(train_dataset):,} train examples, {len(test_dataset):,} test examples")

    # Save to disk if requested
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {output_dir}")
        dataset_dict.save_to_disk(str(output_dir))
        print("Saved successfully")


if __name__ == "__main__":
    app()
