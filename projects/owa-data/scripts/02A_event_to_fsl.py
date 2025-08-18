#!/usr/bin/env python3
"""
Convert event dataset to FSL (Fixed Sequence Length) dataset.

This script performs the following steps:
1. Loads an event dataset (created by 01_raw_events_to_event_dataset.py)
2. Tokenizes the events using EpisodeTokenizer
3. Creates pre-computed FSL sequences for efficient training

The FSL dataset is pre-computed (excluding image loading) and implements proper OWA Dataset
with transforms for on-the-fly image loading, following the user's preferred approach.
"""

from dataclasses import dataclass, field
from pathlib import Path

from jsonargparse import auto_cli
from loguru import logger
from transformers import AutoTokenizer

from owa.data.datasets import DatasetDict, DatasetStage, load_from_disk
from owa.data.datasets.fsl_dataset import FSLDatasetConfig, precompute_fsl_dataset
from owa.data.episode_tokenizer import EpisodeTokenizer, EpisodeTokenizerConfig

# Re-enable logging for owa.data
logger.enable("owa.data")


@dataclass
class Config:
    """Configuration for event to FSL conversion."""

    # Required paths
    input_dir: Path  # Input event dataset directory
    output_dir: Path  # Output FSL dataset directory

    # Model configuration
    tokenizer_name: str

    # Nested configurations
    episode_tokenizer: EpisodeTokenizerConfig = field(default_factory=EpisodeTokenizerConfig)
    fsl_dataset: FSLDatasetConfig = field(default_factory=FSLDatasetConfig)

    # Processing options
    num_proc: int = 32  # Number of processes for tokenization
    fsl_workers: int = 4  # Number of workers for FSL processing


def main(cfg: Config):
    """Convert event dataset to FSL dataset format."""
    print(f"Loading event dataset from: {cfg.input_dir}")
    print(f"Output directory: {cfg.output_dir}")
    print(f"Tokenizer: {cfg.tokenizer_name}")
    print(f"Max sequence length: {cfg.fsl_dataset.max_sequence_length}")

    print("Episode tokenizer cfg:")
    print(f"  - Image token: {cfg.episode_tokenizer.image_token}")
    print(f"  - Image token length: {cfg.episode_tokenizer.image_token_length}")
    print(f"  - Image token prefix: {cfg.episode_tokenizer.image_token_prefix}")
    print(f"  - Image token suffix: {cfg.episode_tokenizer.image_token_suffix}")

    # Load event dataset
    ds_dict = load_from_disk(str(cfg.input_dir))

    # Validate input dataset stage
    if isinstance(ds_dict, DatasetDict):
        print(f"Loaded DatasetDict with splits: {list(ds_dict.keys())}")
        first_dataset = next(iter(ds_dict.values()))
        splits = list(ds_dict.keys())
    else:
        print("Loaded single Dataset")
        first_dataset = ds_dict
        splits = [None]

    if first_dataset.owa_config.stage != DatasetStage.EVENT:
        raise ValueError(
            f"Input dataset must be EVENT stage, got {first_dataset.owa_config.stage}. "
            "Use 01_raw_events_to_event_dataset.py to create event datasets first."
        )

    # Load tokenizer
    print(f"Loading tokenizer: {cfg.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize episode tokenizer
    episode_tokenizer = EpisodeTokenizer(cfg.episode_tokenizer)
    episode_tokenizer.prepare_model(tokenizer=tokenizer)

    # Configure FSL dataset
    cfg.fsl_dataset.pad_token_id = tokenizer.pad_token_id
    print("FSL dataset cfg:")
    print(f"  - Pad token ID: {cfg.fsl_dataset.pad_token_id}")
    print(f"  - Max sequence length: {cfg.fsl_dataset.max_sequence_length}")

    processed_datasets = {}

    for split in splits:
        ds = ds_dict[split] if split else ds_dict
        split_name = split if split else "train"
        print(f"Processing {len(ds):,} events from {split_name} split")

        # Step 1: Tokenize event dataset
        print(f"Tokenizing {split_name} events...")
        tokenized_dataset = episode_tokenizer.tokenize_event_dataset(ds, map_kwargs={"num_proc": cfg.num_proc})
        print(f"Created {len(tokenized_dataset):,} tokenized events")

        # Step 2: Create FSL dataset
        print("Creating FSL dataset from tokenized events...")
        fsl_dataset = precompute_fsl_dataset(tokenized_dataset, config=cfg.fsl_dataset, num_workers=cfg.fsl_workers)
        print(f"Created {len(fsl_dataset):,} FSL sequences for {split_name} split")

        processed_datasets[split_name] = fsl_dataset

    # Combine into DatasetDict if multiple splits
    final_dataset = (
        DatasetDict(processed_datasets) if len(processed_datasets) > 1 else list(processed_datasets.values())[0]
    )

    # Save dataset
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving FSL dataset to {cfg.output_dir}")
    final_dataset.save_to_disk(str(cfg.output_dir))

    # Print summary
    if len(processed_datasets) > 1:
        total_sequences = sum(len(ds) for ds in processed_datasets.values())
        print(f"Saved {total_sequences:,} total FSL sequences")
        for split_name, ds in processed_datasets.items():
            print(f"  {split_name}: {len(ds):,} sequences")
    else:
        split_name = list(processed_datasets.keys())[0]
        ds = list(processed_datasets.values())[0]
        print(f"Saved {len(ds):,} FSL sequences ({split_name})")

    print("FSL dataset creation completed successfully!")


if __name__ == "__main__":
    main(auto_cli(Config, as_positional=False))
