#!/usr/bin/env python3
"""Process raw MCAP files to create event datasets."""

import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import typer
from datasets import Dataset as HFDataset
from datasets import DatasetDict, Features, Value
from datasets import DatasetInfo as HFDatasetInfo
from tqdm import tqdm

from mcap_owa.highlevel import McapMessage, OWAMcapReader
from owa.data.datasets import Dataset, DatasetConfig, DatasetStage
from owa.data.interval import Intervals
from owa.data.interval.selector import All

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
) -> List[Dict]:
    """
    Process a single MCAP file to extract raw events, applying rate-limiting
    (drop-only) per topic and optional topic filtering.

    Args:
        episode_path: Path to the MCAP file (string).
        rate_settings: Mapping from topic name to desired rate in Hz.
        keep_topics: Optional list of topics to keep. If None, all topics are kept.

    Returns:
        List of event dictionaries with keys: episode_path, topic, timestamp_ns, message_type, mcap_message.
        Messages are returned as McapMessage objects for binary storage.
    """
    events: List[Dict] = []
    interval_extractor = All()  # Select all intervals
    valid_intervals: Intervals = interval_extractor.extract_intervals(Path(episode_path))

    # Prepare per-topic tracking for last-kept timestamp in nanoseconds
    last_kept_ts: Dict[str, int] = {topic: 0 for topic in rate_settings.keys()}

    with OWAMcapReader(Path(episode_path)) as reader:
        for interval in valid_intervals:
            for mcap_msg in reader.iter_messages(start_time=interval.start, end_time=interval.end):
                topic, timestamp_ns, msg = mcap_msg.topic, mcap_msg.timestamp, mcap_msg.message
                message_type = mcap_msg.message_type

                # Filter by topic if keep_topics is specified
                if keep_topics is not None and topic not in keep_topics:
                    continue

                if topic in rate_settings:
                    # Convert rate (Hz) to minimum nanoseconds between messages
                    min_interval_ns = int((1.0 / rate_settings[topic]) * 1e9)
                    if (timestamp_ns - last_kept_ts[topic]) < min_interval_ns:
                        continue
                    last_kept_ts[topic] = timestamp_ns

                # Create McapMessage object and serialize to bytes
                mcap_message_obj = McapMessage(
                    topic=topic, timestamp=timestamp_ns, message=msg, message_type=message_type
                )
                # Serialize McapMessage to bytes using model_dump_json
                mcap_message_bytes = mcap_message_obj.model_dump_json().encode("utf-8")

                # Store relative path if mcap_root_directory is provided
                stored_episode_path = episode_path
                if mcap_root_directory:
                    try:
                        stored_episode_path = str(Path(episode_path).relative_to(mcap_root_directory))
                    except ValueError:
                        # If path is not relative to mcap_root_directory, store absolute path
                        stored_episode_path = episode_path

                events.append(
                    {
                        "episode_path": stored_episode_path,
                        "topic": topic,
                        "timestamp_ns": timestamp_ns,
                        "message_type": message_type,
                        "mcap_message": mcap_message_bytes,  # Store serialized bytes
                    }
                )

    return events


def generate_event_examples(
    episode_paths: List[str],
    rate_settings: Dict[str, float],
    keep_topics: Optional[List[str]] = None,
    num_workers: int = 4,
    mcap_root_directory: Optional[str] = None,
):
    """
    Generator function that yields event examples by processing each raw events file
    in parallel using multiple processes.

    Args:
        episode_paths: List of MCAP file paths (strings).
        rate_settings: Mapping from topic to desired rate (Hz).
        keep_topics: Optional list of topics to keep. If None, all topics are kept.
        num_workers: Number of parallel worker processes.

    Yields:
        Individual event dictionaries suitable for Hugging Face Dataset.
    """
    total_files = len(episode_paths)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {
            executor.submit(process_raw_events_file, fp, rate_settings, keep_topics, mcap_root_directory): fp
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
) -> Dataset:
    """
    Create a Hugging Face event dataset from the given MCAP file paths by streaming
    examples from a generator.

    Args:
        episode_paths: List of pathlib.Path objects pointing to MCAP files.
        rate_settings: Mapping from topic to rate (Hz) to apply drop-only downsampling.
        keep_topics: Optional list of topics to keep. If None, all topics are kept.
        num_workers: Number of worker processes for parallel file processing.

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
    test_percent: float = typer.Option(0.2, "--test_percent", help="Fraction of training files for test set"),
    rate: Optional[List[str]] = typer.Option(None, "--rate", help="Rate-limiting per topic in 'topic=Hz' format"),
    num_workers: int = typer.Option(4, "--num-workers", help="Number of parallel worker processes"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Directory to save the dataset"),
    keep_topic: Optional[List[str]] = typer.Option(
        None, "--keep_topic", help="Topics to keep (default: screen, keyboard, mouse)"
    ),
):
    """Generate event dataset from raw MCAP files."""

    # Validate test_percent
    if test_percent <= 0 or test_percent >= 1:
        raise typer.BadParameter("--test_percent must be between 0 and 1 (exclusive)")

    # Parse rate settings or use defaults
    rate_settings = parse_rate_argument(rate) if rate else {"mouse": 60.0, "screen": 20.0}
    topics_to_keep = keep_topic if keep_topic else ["screen", "keyboard", "mouse"]

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
        random.shuffle(shuffled)
        test_count = max(1, int(len(shuffled) * test_percent))
        test_files = shuffled[:test_count]
        train_files = shuffled[test_count:]

    print(f"Processing {len(train_files)} train files, {len(test_files)} test files with {num_workers} workers")

    # Confirm if no output directory
    if not output_dir:
        if not typer.confirm("No --output-dir given. Continue without saving to disk?", default=False):
            raise typer.Exit(1)

    # Create event datasets
    mcap_root_directory = str(train_dir)
    train_dataset = create_event_dataset(
        train_files, rate_settings, topics_to_keep, num_workers, "train", mcap_root_directory
    )
    test_dataset = create_event_dataset(
        test_files, rate_settings, topics_to_keep, num_workers, "test", mcap_root_directory
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
