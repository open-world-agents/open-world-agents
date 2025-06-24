#!/usr/bin/env python3
"""
01_raw_events_to_event_dataset.py

This script processes raw event data from MCAP files in given directories to produce a Hugging Face DatasetDict with
"train" and "test" splits. You can supply separate directories for training and testing; if no test directory is provided,
a certain percentage of training files will be randomly split into a test set.

Usage (CLI):
    python 01_raw_events_to_event_dataset.py \
        --train-dir /path/to/train_folder \
        [--test-dir /path/to/test_folder] \
        [--test_percent 0.2] \
        [--rate mouse=60 screen=20] \
        [--keep_topic screen --keep_topic keyboard --keep_topic mouse] \
        [--num_workers 8] \
        [--output_dir /path/to/save_dataset]

    - If --test-dir is omitted, test set is formed by randomly sampling `test_percent` fraction of files in train-dir.
    - --rate topic=Hz can be repeated to apply drop-only downsampling per topic. Defaults to mouse=60, screen=20 if omitted.
    - --keep_topic can be repeated to specify which topics to keep. Defaults to screen, keyboard, mouse if omitted.
    - Output is saved (optional) as an event dataset with "train" and "test" keys.
"""

import random
import time

# Concurrency imports
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

# Hugging Face Datasets imports
import typer
from datasets import Dataset as HFDataset
from datasets import DatasetDict, Features, Value
from datasets import DatasetInfo as HFDatasetInfo
from rich.console import Console
from rich.panel import Panel

# Progress bar and styling
from tqdm import tqdm

# MCAP and interval extraction imports
from mcap_owa.highlevel import McapMessage, OWAMcapReader
from owa.data.interval import Intervals
from owa.data.interval.selector import All

app = typer.Typer(add_completion=False)
console = Console()


def parse_rate_argument(rate_args: List[str]) -> Dict[str, float]:
    """
    Parse CLI --rate arguments of the form "topic=Hz" into a mapping
    from topic name to target frequency (Hz).

    Args:
        rate_args: List of strings, each in "topic=Hz" format.

    Returns:
        Dictionary mapping topic (str) to rate (float).
    """
    rate_settings: Dict[str, float] = {}
    for arg in rate_args:
        if "=" not in arg:
            console.print(f"[red]✗[/red] Invalid rate argument '{arg}'. Expected format: topic=Hz")
            raise typer.Exit(code=1)
        topic, rate_str = arg.split("=", maxsplit=1)
        try:
            rate = float(rate_str)
            if rate <= 0:
                raise ValueError("Rate must be positive")
        except ValueError as e:
            console.print(f"[red]✗[/red] Invalid rate value in '{arg}': {e}")
            raise typer.Exit(code=1)
        rate_settings[topic] = rate
    return rate_settings


def process_raw_events_file(
    file_path: str,
    rate_settings: Dict[str, float],
    keep_topics: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Process a single MCAP file to extract raw events, applying rate-limiting
    (drop-only) per topic and optional topic filtering.

    Args:
        file_path: Path to the MCAP file (string).
        rate_settings: Mapping from topic name to desired rate in Hz.
        keep_topics: Optional list of topics to keep. If None, all topics are kept.

    Returns:
        List of event dictionaries with keys: file_path, topic, timestamp_ns, message_type, mcap_message.
        Messages are returned as McapMessage objects for binary storage.
    """
    events: List[Dict] = []
    interval_extractor = All()  # Select all intervals
    try:
        valid_intervals: Intervals = interval_extractor.extract_intervals(Path(file_path))
    except Exception as e:
        # Use print instead of console to avoid pickling issues in multiprocessing
        print(f"⚠ Failed to extract intervals from {Path(file_path).name}: {e}")
        return events

    # Prepare per-topic tracking for last-kept timestamp in nanoseconds
    last_kept_ts: Dict[str, int] = {topic: 0 for topic in rate_settings.keys()}

    try:
        with OWAMcapReader(Path(file_path)) as reader:
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

                    events.append(
                        {
                            "file_path": file_path,
                            "topic": topic,
                            "timestamp_ns": timestamp_ns,
                            "message_type": message_type,
                            "mcap_message": mcap_message_bytes,  # Store serialized bytes
                        }
                    )
    except Exception as e:
        # Use print instead of console to avoid pickling issues in multiprocessing
        print(f"⚠ Error reading file {Path(file_path).name}: {e}")

    return events


def generate_event_examples(
    file_paths: List[str],
    rate_settings: Dict[str, float],
    keep_topics: Optional[List[str]] = None,
    num_workers: int = 4,
):
    """
    Generator function that yields event examples by processing each raw events file
    in parallel using multiple processes.

    Args:
        file_paths: List of MCAP file paths (strings).
        rate_settings: Mapping from topic to desired rate (Hz).
        keep_topics: Optional list of topics to keep. If None, all topics are kept.
        num_workers: Number of parallel worker processes.

    Yields:
        Individual event dictionaries suitable for Hugging Face Dataset.
    """
    total_files = len(file_paths)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {
            executor.submit(process_raw_events_file, fp, rate_settings, keep_topics): fp for fp in file_paths
        }
        with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
            for future in as_completed(future_to_path):
                fp = future_to_path[future]
                try:
                    events = future.result()
                    for event in events:
                        yield event
                except Exception as e:
                    # Use print instead of console to avoid pickling issues in multiprocessing
                    print(f"⚠ Exception raised for file {Path(fp).name}: {e}")
                finally:
                    pbar.update(1)


def create_event_dataset(
    file_paths: List[Path],
    rate_settings: Dict[str, float],
    keep_topics: Optional[List[str]] = None,
    num_workers: int = 4,
    split: str = "train",
) -> HFDataset:
    """
    Create a Hugging Face event dataset from the given MCAP file paths by streaming
    examples from a generator.

    Args:
        file_paths: List of pathlib.Path objects pointing to MCAP files.
        rate_settings: Mapping from topic to rate (Hz) to apply drop-only downsampling.
        keep_topics: Optional list of topics to keep. If None, all topics are kept.
        num_workers: Number of worker processes for parallel file processing.

    Returns:
        A Hugging Face Dataset containing the combined events.
    """
    file_path_strs = [str(fp) for fp in file_paths]

    features = Features(
        {
            "file_path": Value("string"),
            "topic": Value("string"),
            "timestamp_ns": Value("int64"),
            "message_type": Value("string"),
            "mcap_message": Value("binary"),  # Use bytes serialization for McapMessage
        }
    )

    event_dataset = HFDataset.from_generator(
        generate_event_examples,
        gen_kwargs={
            "file_paths": file_path_strs,
            "rate_settings": rate_settings,
            "keep_topics": keep_topics,
            "num_workers": num_workers,
        },
        features=features,
        split=split,
    )
    info_to_update = HFDatasetInfo(
        description="",
        dataset_name="open-world-agents/goat",
        homepage="https://github.com/open-world-agents",
    )
    event_dataset.info.update(info_to_update)

    return event_dataset


@app.command()
def main(
    train_dir: Path = typer.Option(
        ...,
        "--train-dir",
        "-tr",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing MCAP files to use for training.",
    ),
    test_dir: Optional[Path] = typer.Option(
        None,
        "--test-dir",
        "-te",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="(Optional) Directory containing MCAP files to use for testing. If omitted, a fraction of train-dir is used.",
    ),
    test_percent: float = typer.Option(
        0.2,
        "--test_percent",
        "-p",
        help="Fraction of training files to allocate to test set if --test_dir is not provided (0 < value < 1).",
    ),
    rate: Optional[List[str]] = typer.Option(
        None, "--rate", "-r", help="Rate-limiting per topic in 'topic=Hz' format. Can be specified multiple times."
    ),
    num_workers: int = typer.Option(
        4, "--num-workers", "-n", help="Number of parallel worker processes for reading files."
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Directory to save the resulting event dataset via save_to_disk."
    ),
    keep_topic: Optional[List[str]] = typer.Option(
        None,
        "--keep_topic",
        help="Topic to keep (repeatable). Replaces defaults. Default: screen, keyboard, mouse. Use --keep_topic multiple times to specify different topics.",
    ),
):
    """
    Generate a Hugging Face event dataset with 'train' and 'test' splits from raw MCAP files in specified directories.
    If --test_dir is omitted, randomly split files in train_dir according to --test_percent.
    """
    start_time = time.time()

    # Print header
    console.print(Panel.fit("🔄 Raw Events to Event Dataset", style="bold blue"))

    # 1. Validate test_percent
    if test_percent <= 0 or test_percent >= 1:
        console.print("[red]✗[/red] --test_percent must be between 0 and 1 (exclusive).")
        raise typer.Exit(code=1)

    # 2. Parse rate settings or set defaults
    default_rates = {"mouse": 60.0, "screen": 20.0}
    if rate:
        rate_settings = parse_rate_argument(rate)
        console.print(f"[cyan]📊[/cyan] Rate settings: {rate_settings}")
    else:
        rate_settings = default_rates
        console.print(f"[cyan]📊[/cyan] Using default rates: {rate_settings}")

    # 3. Set topic filtering or use defaults
    default_topics = ["screen", "keyboard", "mouse"]
    if keep_topic:
        topics_to_keep = keep_topic
        console.print(f"[cyan]🎯[/cyan] Keeping topics: {topics_to_keep}")
    else:
        topics_to_keep = default_topics
        console.print(f"[cyan]🎯[/cyan] Using default topics: {topics_to_keep}")

    # 4. Gather all MCAP files in train_dir
    console.print(f"[cyan]📁[/cyan] Loading from: {train_dir}")
    train_files = sorted(train_dir.glob("*.mcap"))
    if not train_files:
        console.print(f"[red]✗[/red] No MCAP files found in train-dir: {train_dir}")
        raise typer.Exit(code=1)

    # 5. Determine test_files
    if test_dir:
        console.print(f"[cyan]📁[/cyan] Loading test from: {test_dir}")
        test_files = sorted(test_dir.glob("*.mcap"))
        if not test_files:
            console.print(f"[red]✗[/red] No MCAP files found in test-dir: {test_dir}")
            raise typer.Exit(code=1)
        # Ensure train and test do not overlap
        train_set = set(str(p) for p in train_files)
        overlap = set(str(p) for p in test_files).intersection(train_set)
        if overlap:
            console.print(f"[red]✗[/red] Same files present in train-dir and test-dir: {len(overlap)} files")
            raise typer.Exit(code=1)
    else:
        shuffled = train_files.copy()
        random.shuffle(shuffled)
        test_count = max(1, int(len(shuffled) * test_percent))
        test_files = shuffled[:test_count]
        train_files = shuffled[test_count:]
        percent = (test_count / len(shuffled)) * 100
        console.print(f"[cyan]🔀[/cyan] Split {test_count} of {len(shuffled)} files into test set ({percent:.1f}%)")

    console.print(
        f"[green]📊[/green] Found [bold]{len(train_files)}[/bold] train, [bold]{len(test_files)}[/bold] test files"
    )
    console.print(f"[cyan]⚙️[/cyan] Processing with [bold]{num_workers}[/bold] workers")

    # 6. Prompt for confirmation if output_dir not provided
    if not output_dir:
        confirm = typer.confirm("No --output-dir given. Continue without saving to disk?", default=False)
        if not confirm:
            console.print("[yellow]⚠[/yellow] Aborting because no output directory was provided.")
            raise typer.Exit(code=1)

    # 7. Create event datasets for train and test
    train_dataset = create_event_dataset(train_files, rate_settings, topics_to_keep, num_workers, split="train")
    test_dataset = create_event_dataset(test_files, rate_settings, topics_to_keep, num_workers, split="test")

    # 8. Combine into DatasetDict
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    console.print(
        f"[green]✓[/green] Created [bold]{len(train_dataset):,}[/bold] train, [bold]{len(test_dataset):,}[/bold] test examples"
    )

    # 9. Save to disk if requested
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[cyan]💾[/cyan] Saving to {output_dir}")
        dataset_dict.save_to_disk(str(output_dir))
        console.print("[green]✓[/green] Saved successfully")

    # 10. Display timing information
    elapsed_time = time.time() - start_time
    console.print(
        f"[green]🎉[/green] Completed in [bold]{elapsed_time:.2f}s[/bold] ([bold]{elapsed_time / 60:.1f}min[/bold])"
    )


if __name__ == "__main__":
    app()
