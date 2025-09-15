#!/usr/bin/env python3
"""Process raw MCAP files to create event datasets."""

from dataclasses import dataclass
from pathlib import Path

from jsonargparse import auto_cli
from loguru import logger

from owa.data.processing import McapToEventConfig, build_event_dataset

# Re-enable logging for owa.data
logger.enable("owa.data")


@dataclass
class Config:
    """Configuration for raw events to event dataset conversion CLI."""

    # Required paths
    input_dir: Path  # Directory containing MCAP files
    output_dir: Path  # Directory to save the dataset

    # McapToEvent configuration
    mcap_to_event_config: McapToEventConfig


def main(cfg: Config):
    """Generate event dataset from raw MCAP files."""
    # Gather MCAP files
    mcap_files = sorted(cfg.input_dir.rglob("*.mcap"))
    if not mcap_files:
        raise ValueError(f"No MCAP files found in input-dir: {cfg.input_dir}")

    logger.info(f"Processing {len(mcap_files)} MCAP files with {cfg.mcap_to_event_config.num_workers} workers")

    # Create event dataset
    dataset = build_event_dataset(mcap_files, config=cfg.mcap_to_event_config, mcap_root_directory=str(cfg.input_dir))
    logger.info(f"Created {len(dataset):,} event examples")

    # Save to disk
    logger.info(f"Saving event dataset to: {cfg.output_dir}")
    dataset.save_to_disk(str(cfg.output_dir))

    logger.info("Event dataset saved successfully!")


if __name__ == "__main__":
    main(auto_cli(Config, as_positional=False))
