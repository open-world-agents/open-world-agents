#!/usr/bin/env python3
"""
01_raw_events_to_event_dataset.py

This script processes raw event data from MCAP files in given directories to produce a Hugging Face DatasetDict with
"train" and "test" splits. You can supply separate directories for training and testing; if no test directory is provided,
a certain percentage of training files will be randomly split into a test set.

Usage (CLI):
    python 01_raw_events_to_event_dataset.py \
        --train_dir /path/to/train_folder \
        [--test_dir /path/to/test_folder] \
        [--test_percent 0.2] \
        [--rate mouse=60 screen=20] \
        [--num_workers 8] \
        [--output_dir /path/to/save_dataset]

    - If --test_dir is omitted, test set is formed by randomly sampling `test_percent` fraction of files in train_dir.
    - --rate topic=Hz can be repeated to apply drop-only downsampling per topic. Defaults to mouse=60, screen=20 if omitted.
    - Output is saved (optional) as an event dataset with "train" and "test" keys.
"""

import json
import random

# Concurrency imports
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

# Hugging Face Datasets imports
import typer
from datasets import Dataset as HFDataset
from datasets import DatasetDict, Features, Value
from datasets import DatasetInfo as HFDatasetInfo

# Progress bar
from tqdm import tqdm

# MCAP and interval extraction imports
from mcap_owa.highlevel import OWAMcapReader
from owa.data.interval import Intervals
from owa.data.interval.selector import All

app = typer.Typer(add_completion=False)


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
            typer.echo(f"Invalid rate argument '{arg}'. Expected format: topic=Hz", err=True)
            raise typer.Exit(code=1)
        topic, rate_str = arg.split("=", maxsplit=1)
        try:
            rate = float(rate_str)
            if rate <= 0:
                raise ValueError("Rate must be positive")
        except ValueError as e:
            typer.echo(f"Invalid rate value in '{arg}': {e}", err=True)
            raise typer.Exit(code=1)
        rate_settings[topic] = rate
    return rate_settings


def process_raw_events_file(
    file_path: str,
    rate_settings: Dict[str, float],
) -> List[Dict]:
    """
    Process a single MCAP file to extract raw events, applying rate-limiting
    (drop-only) per topic.

    Args:
        file_path: Path to the MCAP file (string).
        rate_settings: Mapping from topic name to desired rate in Hz.

    Returns:
        List of event dictionaries with keys: file_path, topic, timestamp, msg.
        Messages are returned as raw bytes for encoding preservation.
    """
    events: List[Dict] = []
    interval_extractor = All()  # Select all intervals
    try:
        valid_intervals: Intervals = interval_extractor.extract_intervals(Path(file_path))
    except Exception as e:
        typer.echo(f"[Warning] Failed to extract intervals from {file_path}: {e}", err=True)
        return events

    # Prepare per-topic tracking for last-kept timestamp in nanoseconds
    last_kept_ts: Dict[str, int] = {topic: 0 for topic in rate_settings.keys()}

    try:
        with OWAMcapReader(Path(file_path)) as reader:
            for interval in valid_intervals:
                for mcap_msg in reader.iter_messages(start_time=interval.start, end_time=interval.end):
                    topic, timestamp_ns, msg = mcap_msg.topic, mcap_msg.timestamp, mcap_msg.message
                    message_type = mcap_msg.message_type

                    if topic in rate_settings:
                        # Convert rate (Hz) to minimum nanoseconds between messages
                        min_interval_ns = int((1.0 / rate_settings[topic]) * 1e9)
                        if (timestamp_ns - last_kept_ts[topic]) < min_interval_ns:
                            continue
                        last_kept_ts[topic] = timestamp_ns

                    events.append(
                        {
                            "file_path": file_path,
                            "topic": topic,
                            "timestamp_ns": timestamp_ns,
                            "message_type": message_type,
                            "msg": msg,
                        }
                    )
    except Exception as e:
        typer.echo(f"[Warning] Error reading file {file_path}: {e}", err=True)

    return events


def generate_event_examples(file_paths: List[str], rate_settings: Dict[str, float], num_workers: int = 4):
    """
    Generator function that yields event examples by processing each raw events file
    in parallel using multiple processes.

    Args:
        file_paths: List of MCAP file paths (strings).
        rate_settings: Mapping from topic to desired rate (Hz).
        num_workers: Number of parallel worker processes.

    Yields:
        Individual event dictionaries suitable for Hugging Face Dataset.
    """
    total_files = len(file_paths)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(process_raw_events_file, fp, rate_settings): fp for fp in file_paths}
        with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
            for future in as_completed(future_to_path):
                fp = future_to_path[future]
                try:
                    events = future.result()
                    for event in events:
                        yield event
                except Exception as e:
                    typer.echo(f"[Warning] Exception raised for file {fp}: {e}", err=True)
                finally:
                    pbar.update(1)


def create_event_dataset(
    file_paths: List[Path],
    rate_settings: Dict[str, float],
    num_workers: int = 4,
    split: str = "train",
) -> HFDataset:
    """
    Create a Hugging Face event dataset from the given MCAP file paths by streaming
    examples from a generator.

    Args:
        file_paths: List of pathlib.Path objects pointing to MCAP files.
        rate_settings: Mapping from topic to rate (Hz) to apply drop-only downsampling.
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
            "msg": Value("binary"),
        }
    )

    event_dataset = HFDataset.from_generator(
        generate_event_examples,
        gen_kwargs={
            "file_paths": file_path_strs,
            "rate_settings": rate_settings,
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
        "--train_dir",
        "-tr",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing MCAP files to use for training.",
    ),
    test_dir: Optional[Path] = typer.Option(
        None,
        "--test_dir",
        "-te",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="(Optional) Directory containing MCAP files to use for testing. If omitted, a fraction of train_dir is used.",
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
):
    """
    Generate a Hugging Face event dataset with 'train' and 'test' splits from raw MCAP files in specified directories.
    If --test_dir is omitted, randomly split files in train_dir according to --test_percent.
    """
    # 1. Validate test_percent
    if test_percent <= 0 or test_percent >= 1:
        typer.echo("[Error] --test_percent must be between 0 and 1 (exclusive).", err=True)
        raise typer.Exit(code=1)

    # 2. Parse rate settings or set defaults
    default_rates = {"mouse": 60.0, "screen": 20.0}
    if rate:
        rate_settings = parse_rate_argument(rate)
    else:
        rate_settings = default_rates
        typer.echo(f"No --rate given. Using default rates: {rate_settings}")

    # 3. Gather all MCAP files in train_dir
    train_files = sorted(train_dir.glob("*.mcap"))
    if not train_files:
        typer.echo(f"[Error] No MCAP files found in train_dir: {train_dir}", err=True)
        raise typer.Exit(code=1)

    # 4. Determine test_files
    if test_dir:
        test_files = sorted(test_dir.glob("*.mcap"))
        if not test_files:
            typer.echo(f"[Error] No MCAP files found in test_dir: {test_dir}", err=True)
            raise typer.Exit(code=1)
        # Ensure train and test do not overlap
        train_set = set(str(p) for p in train_files)
        overlap = set(str(p) for p in test_files).intersection(train_set)
        if overlap:
            typer.echo(f"[Error] Same files present in train_dir and test_dir: {overlap}", err=True)
            raise typer.Exit(code=1)
    else:
        shuffled = train_files.copy()
        random.shuffle(shuffled)
        test_count = max(1, int(len(shuffled) * test_percent))
        test_files = shuffled[:test_count]
        train_files = shuffled[test_count:]
        percent = (test_count / len(shuffled)) * 100
        typer.echo(
            f"No --test_dir given. Split {test_count} of {len(shuffled)} train files into test set ({percent:.1f}%)."
        )

    typer.echo(f"Train files: {len(train_files)}, Test files: {len(test_files)}")
    typer.echo(f"Processing with {num_workers} workers...")

    # 5. Prompt for confirmation if output_dir not provided
    if not output_dir:
        confirm = typer.confirm("No --output-dir given. Continue without saving to disk?", default=False)
        if not confirm:
            typer.echo("Aborting because no output directory was provided.", err=True)
            raise typer.Exit(code=1)

    # 6. Create event datasets for train and test
    train_dataset = create_event_dataset(train_files, rate_settings, num_workers, split="train")
    test_dataset = create_event_dataset(test_files, rate_settings, num_workers, split="test")

    # 7. Combine into DatasetDict
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    typer.echo(f"DatasetDict created. Train examples: {len(train_dataset)}, Test examples: {len(test_dataset)}")

    # 8. Save to disk if requested
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(output_dir))
        typer.echo(f"DatasetDict saved to {output_dir}")

    # 9. Show few examples from each split in a more readable format
    def pretty_print_example(example: Dict):
        typer.echo("----------------------------------------")
        typer.echo(f"file_path: {example['file_path']}")
        typer.echo(f"topic:     {example['topic']}")
        typer.echo(f"timestamp_ns: {example['timestamp_ns']}")
        # Attempt to decode and pretty-print the msg field
        raw_msg = example["msg"]
        try:
            decoded = raw_msg.decode("utf-8")
            parsed_json = json.loads(decoded)
            pretty_msg = json.dumps(parsed_json, ensure_ascii=False, indent=2)
        except Exception:
            # Fallback to repr if not valid JSON
            pretty_msg = repr(raw_msg)
        typer.echo("msg:\n" + pretty_msg)

    typer.echo("=== Train sample ===")
    for example in train_dataset.select(range(min(3, len(train_dataset)))):
        pretty_print_example(example)

    # For each unique topic in train, print one example
    seen_topics = set()
    typer.echo("=== One example per topic (train) ===")
    for example in train_dataset:
        topic = example["topic"]
        if topic not in seen_topics:
            pretty_print_example(example)
            seen_topics.add(topic)
        if len(seen_topics) == len(set(train_dataset["topic"])):
            break

    typer.echo("=== Test sample ===")
    for example in test_dataset.select(range(min(3, len(test_dataset)))):
        pretty_print_example(example)


if __name__ == "__main__":
    app()


"""
=== Train sample ===
----------------------------------------
file_path: /mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-jy-1.mcap
topic:     window
timestamp_ns: 1743127093025190600
msg:
{
  "title": "Conda Environment (env) - owl  mcap record expert-jy-1.mkv --window-name hexagon",
  "rect": [
    167,
    542,
    1160,
    1061
  ],
  "hWnd": 1772442
}
----------------------------------------
file_path: /mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-jy-1.mcap
topic:     keyboard/state
timestamp_ns: 1743127093025190600
msg:
{
  "buttons": []
}
----------------------------------------
file_path: /mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-jy-1.mcap
topic:     mouse/state
timestamp_ns: 1743127093025190600
msg:
{
  "x": 1607,
  "y": 853,
  "buttons": []
}
=== One example per topic (train) ===
----------------------------------------
file_path: /mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-jy-1.mcap
topic:     window
timestamp_ns: 1743127093025190600
msg:
{
  "title": "Conda Environment (env) - owl  mcap record expert-jy-1.mkv --window-name hexagon",
  "rect": [
    167,
    542,
    1160,
    1061
  ],
  "hWnd": 1772442
}
----------------------------------------
file_path: /mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-jy-1.mcap
topic:     keyboard/state
timestamp_ns: 1743127093025190600
msg:
{
  "buttons": []
}
----------------------------------------
file_path: /mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-jy-1.mcap
topic:     mouse/state
timestamp_ns: 1743127093025190600
msg:
{
  "x": 1607,
  "y": 853,
  "buttons": []
}
----------------------------------------
file_path: /mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-jy-1.mcap
topic:     mouse
timestamp_ns: 1743127093103980900
msg:
{
  "event_type": "click",
  "x": 1607,
  "y": 853,
  "button": "left",
  "pressed": true
}
----------------------------------------
file_path: /mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-jy-1.mcap
topic:     screen
timestamp_ns: 1743127093215683600
msg:
{
  "path": "expert-jy-1.mkv",
  "pts": 40000000,
  "utc_ns": 1743127093215683600
}
----------------------------------------
file_path: /mnt/raid11/datasets/owa/mcaps/super-hexagon/expert-jy-1.mcap
topic:     keyboard
timestamp_ns: 1743127093849720900
msg:
{
  "event_type": "press",
  "vk": 32
}
=== Test sample ===
----------------------------------------
file_path: /mnt/raid11/datasets/owa/mcaps/super-hexagon-30s/expert-half-jy-1.mcap
topic:     window
timestamp_ns: 1743593677761598900
msg:
{
  "title": "Super Hexagon",
  "rect": [
    1284,
    473,
    2068,
    992
  ],
  "hWnd": 6688344
}
----------------------------------------
file_path: /mnt/raid11/datasets/owa/mcaps/super-hexagon-30s/expert-half-jy-1.mcap
topic:     keyboard/state
timestamp_ns: 1743593677762595900
msg:
{
  "buttons": [
    120
  ]
}
----------------------------------------
file_path: /mnt/raid11/datasets/owa/mcaps/super-hexagon-30s/expert-half-jy-1.mcap
topic:     mouse/state
timestamp_ns: 1743593677762595900
msg:
{
  "x": 1615,
  "y": 690,
  "buttons": []
}
"""
