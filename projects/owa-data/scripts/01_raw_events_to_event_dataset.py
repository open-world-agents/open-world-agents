#!/usr/bin/env python3
"""Process raw MCAP files to create event datasets."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import typer
from datasets import DatasetDict
from loguru import logger

from owa.data.processing import McapToEventConfig, build_event_dataset

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
):
    """Generate event dataset from raw MCAP files."""

    # Validate test_percent
    if test_percent <= 0 or test_percent >= 1:
        raise typer.BadParameter("--test_percent must be between 0 and 1 (exclusive)")

    # Parse rate settings or use defaults
    rate_settings = parse_rate_argument(rate) if rate else {"mouse/raw": 20.0, "screen": 20.0}
    topics_to_keep = keep_topic if keep_topic else ["screen", "keyboard", "mouse/raw"]

    # Gather MCAP files
    train_files = sorted(train_dir.rglob("*.mcap"))
    if not train_files:
        raise typer.BadParameter(f"No MCAP files found in train-dir: {train_dir}")

    # Determine test files
    if test_dir:
        test_files = sorted(test_dir.rglob("*.mcap"))
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

    # Confirm if no output directory
    if not output_dir:
        if not typer.confirm("No --output-dir given. Continue without saving to disk?", default=False):
            raise typer.Exit(1)

    # Create configuration object
    config = McapToEventConfig(
        rate_settings=rate_settings,
        keep_topics=topics_to_keep,
        num_workers=num_workers,
    )

    # Create event datasets
    mcap_root_directory = str(train_dir)
    train_dataset = build_event_dataset(
        episode_paths=train_files, config=config, split="train", mcap_root_directory=mcap_root_directory
    )
    test_dataset = build_event_dataset(
        episode_paths=test_files, config=config, split="test", mcap_root_directory=mcap_root_directory
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
