#!/usr/bin/env python3
"""
Convert event dataset to FSL (Fixed Sequence Length) dataset.

This script provides a CLI interface for converting event datasets to FSL format.
The core functionality is implemented in owa.data.processing.event_to_fsl.
"""

from dataclasses import dataclass
from pathlib import Path

from jsonargparse import auto_cli
from loguru import logger
from tqdm import tqdm

from owa.data.datasets import load_from_disk
from owa.data.processing import EventToFSLConfig, build_fsl_dataset

# Re-enable logging for owa.data
logger.enable("owa.data")

logger.remove()
# how to use loguru with tqdm: https://github.com/Delgan/loguru/issues/135
logger.add(lambda msg: tqdm.write(msg, end=""), filter={"owa.ocap": "DEBUG", "owa.env.gst": "INFO"}, colorize=True)


@dataclass
class Config:
    """Configuration for event to FSL conversion CLI."""

    # Required paths
    input_dir: Path  # Input event dataset directory
    output_dir: Path  # Output FSL dataset directory

    # EventToFSL configuration
    event_to_fsl_config: EventToFSLConfig


def main(cfg: Config):
    """Convert event dataset to FSL dataset format."""
    # Load event dataset
    logger.info(f"Loading event dataset from: {cfg.input_dir}")
    event_dataset = load_from_disk(str(cfg.input_dir))

    # Create FSL dataset
    final_dataset = build_fsl_dataset(event_dataset, config=cfg.event_to_fsl_config)

    # Save dataset
    logger.info(f"Saving FSL dataset to: {cfg.output_dir}")
    final_dataset.save_to_disk(str(cfg.output_dir))

    logger.info("FSL dataset saved successfully!")


if __name__ == "__main__":
    main(auto_cli(Config, as_positional=False))
