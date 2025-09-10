#!/usr/bin/env python3
"""
Convert event dataset to FSL (Fixed Sequence Length) dataset.

This script provides a CLI interface for converting event datasets to FSL format.
The core functionality is implemented in owa.data.processing.event_to_fsl.
"""

from dataclasses import dataclass, field
from pathlib import Path

from jsonargparse import auto_cli
from loguru import logger
from tqdm import tqdm

from owa.data.datasets.fsl_dataset import FSLDatasetConfig
from owa.data.processing.event_to_fsl import EventToFSLConfig, create_fsl_dataset_from_events_and_save

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

    # Model configuration
    tokenizer_name: str

    # Nested configurations
    episode_tokenize_config: dict = field(default_factory=dict)
    fsl_dataset: FSLDatasetConfig = field(default_factory=FSLDatasetConfig)

    # Processing options
    num_proc: int = 32  # Number of processes for tokenization
    fsl_workers: int = 4  # Number of workers for FSL processing


def main(cfg: Config):
    """Convert event dataset to FSL dataset format."""
    # Create EventToFSLConfig from CLI config
    event_to_fsl_config = EventToFSLConfig(
        tokenizer_name=cfg.tokenizer_name,
        episode_tokenize_config=cfg.episode_tokenize_config,
        fsl_dataset=cfg.fsl_dataset,
        num_proc=cfg.num_proc,
        fsl_workers=cfg.fsl_workers,
    )

    # Use the source function to perform the conversion
    create_fsl_dataset_from_events_and_save(
        input_dir=cfg.input_dir,
        output_dir=cfg.output_dir,
        config=event_to_fsl_config,
    )


if __name__ == "__main__":
    main(auto_cli(Config, as_positional=False))
