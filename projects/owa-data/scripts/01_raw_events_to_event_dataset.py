#!/usr/bin/env python3
"""Process raw MCAP files to create event datasets."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer
from datasets import DatasetDict
from loguru import logger

from owa.data.processing import McapToEventConfig, build_event_dataset

# Re-enable logging for owa.data
logger.enable("owa.data")

app = typer.Typer(add_completion=False)


@dataclass
class Config:
    """Configuration for raw events to event dataset conversion CLI."""

    # Required paths (no defaults)
    output_dir: Path  # Directory to save the dataset
    mcap_to_event_config: McapToEventConfig

    # Optional paths (with defaults)
    input_dir: Optional[Path] = None  # Directory containing MCAP files
    train_files: Optional[List[str]] = None  # List of MCAP file paths for training
    test_files: Optional[List[str]] = None   # List of MCAP file paths for testing


def main_jsonargparse(cfg: Config):
    """Generate event dataset from raw MCAP files using jsonargparse config."""
    # Determine MCAP files to process
    if cfg.train_files or cfg.test_files:
        # Use provided file lists
        train_file_paths = [Path(f) for f in (cfg.train_files or [])]
        test_file_paths = [Path(f) for f in (cfg.test_files or [])]

        # Validate all files exist
        all_files = train_file_paths + test_file_paths
        missing_files = [f for f in all_files if not f.exists()]
        if missing_files:
            raise ValueError(f"Missing files: {[str(f) for f in missing_files[:5]]}")

        # Use common parent directory as root, or None if files are scattered
        if all_files:
            try:
                common_parent = Path(all_files[0]).parent
                for file_path in all_files[1:]:
                    while not str(file_path).startswith(str(common_parent)):
                        common_parent = common_parent.parent
                        if common_parent == common_parent.parent:  # Reached filesystem root
                            common_parent = None
                            break
                    if common_parent is None:
                        break
                mcap_root_directory = str(common_parent) if common_parent else None
            except Exception:
                mcap_root_directory = None
    else:
        # Use input_dir (original behavior)
        if not cfg.input_dir:
            raise ValueError("Must specify either input_dir or train_files/test_files")

        mcap_files = sorted(cfg.input_dir.rglob("*.mcap"))
        if not mcap_files:
            raise ValueError(f"No MCAP files found in input-dir: {cfg.input_dir}")

        # Split into train/test (80/20 split)
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        shuffled_indices = rng.permutation(len(mcap_files))
        test_count = max(1, int(len(mcap_files) * 0.2))

        test_file_paths = [mcap_files[i] for i in shuffled_indices[:test_count]]
        train_file_paths = [mcap_files[i] for i in shuffled_indices[test_count:]]
        mcap_root_directory = str(cfg.input_dir)

    logger.info(f"Processing {len(train_file_paths)} train files, {len(test_file_paths)} test files with {cfg.mcap_to_event_config.num_workers} workers")

    # Create datasets
    datasets = {}

    if train_file_paths:
        train_dataset = build_event_dataset(
            train_file_paths,
            config=cfg.mcap_to_event_config,
            mcap_root_directory=mcap_root_directory
        )
        datasets["train"] = train_dataset
        logger.info(f"Created {len(train_dataset):,} train examples")

    if test_file_paths:
        test_dataset = build_event_dataset(
            test_file_paths,
            config=cfg.mcap_to_event_config,
            mcap_root_directory=mcap_root_directory
        )
        datasets["test"] = test_dataset
        logger.info(f"Created {len(test_dataset):,} test examples")

    # Combine into DatasetDict if multiple splits
    if len(datasets) > 1:
        dataset_dict = DatasetDict(datasets)
    else:
        dataset_dict = list(datasets.values())[0]

    # Save to disk
    logger.info(f"Saving event dataset to: {cfg.output_dir}")
    dataset_dict.save_to_disk(str(cfg.output_dir))

    logger.info("Event dataset saved successfully!")


@app.command()
def main(
    train_files: Optional[List[str]] = typer.Option(
        None, "--train-files", help="List of MCAP file paths for training"
    ),
    test_files: Optional[List[str]] = typer.Option(
        None, "--test-files", help="List of MCAP file paths for testing"
    ),
    input_dir: Optional[Path] = typer.Option(
        None, "--input-dir", help="Directory containing MCAP files (alternative to train/test files)"
    ),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory to save the dataset"),
    rate: Optional[List[str]] = typer.Option(None, "--rate", help="Rate-limiting per topic in 'topic=Hz' format"),
    num_workers: int = typer.Option(4, "--num-workers", help="Number of parallel worker processes"),
    keep_topic: Optional[List[str]] = typer.Option(
        None, "--keep-topic", help="Topics to keep (default: screen, keyboard, mouse/raw)"
    ),
    action_topic: Optional[List[str]] = typer.Option(
        None, "--action-topic", help="Topics to consider as action topics"
    ),
):
    """Generate event dataset from raw MCAP files."""

    # Validate inputs
    if not train_files and not test_files and not input_dir:
        raise typer.BadParameter("Must specify either --input-dir or --train-files/--test-files")

    if input_dir and (train_files or test_files):
        raise typer.BadParameter("Cannot specify both --input-dir and --train-files/--test-files")

    # Parse rate settings or use defaults
    def parse_rate_argument(rate_args: List[str]) -> dict:
        """Parse CLI --rate arguments of the form "topic=Hz"."""
        rate_settings = {}
        for arg in rate_args:
            if "=" not in arg:
                raise typer.BadParameter(f"Invalid rate argument '{arg}'. Expected format: topic=Hz")
            topic, rate_str = arg.split("=", maxsplit=1)
            try:
                rate_val = float(rate_str)
                if rate_val <= 0:
                    raise ValueError("Rate must be positive")
            except ValueError as e:
                raise typer.BadParameter(f"Invalid rate value in '{arg}': {e}")
            rate_settings[topic] = rate_val
        return rate_settings

    rate_settings = parse_rate_argument(rate) if rate else {"mouse/raw": 20.0, "screen": 20.0}
    topics_to_keep = keep_topic if keep_topic else ["screen", "keyboard", "mouse/raw"]

    # Create McapToEventConfig
    mcap_config = McapToEventConfig(
        rate_settings=rate_settings,
        keep_topics=topics_to_keep,
        num_workers=num_workers,
    )

    # Create Config object and call main function
    config = Config(
        input_dir=input_dir,
        output_dir=output_dir,
        mcap_to_event_config=mcap_config,
        train_files=train_files,
        test_files=test_files,
    )

    main_jsonargparse(config)


if __name__ == "__main__":
    # Check if we're being called with jsonargparse (config file) or typer (CLI args)
    if len(sys.argv) > 1 and (sys.argv[1].endswith('.yaml') or sys.argv[1].endswith('.yml') or '--config' in sys.argv):
        # Use jsonargparse for config file support
        from jsonargparse import auto_cli
        main_jsonargparse(auto_cli(Config, as_positional=False))
    else:
        # Use typer for CLI arguments
        app()
