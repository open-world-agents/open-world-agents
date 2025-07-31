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

from pathlib import Path

import typer
from loguru import logger
from transformers import AutoTokenizer

from owa.data.datasets import DatasetDict, DatasetStage, load_from_disk

# Import FSL functionality directly
from owa.data.datasets.fsl_dataset import FSLDatasetConfig, precompute_fsl_dataset
from owa.data.episode_tokenizer import EpisodeTokenizer, EpisodeTokenizerConfig

# Re-enable logging for owa.data
logger.enable("owa.data")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_dir: Path = typer.Option(..., "--input-dir", help="Input event dataset directory"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Output FSL dataset directory"),
    tokenizer_name: str = typer.Option(
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct", "--tokenizer", help="HuggingFace tokenizer model name"
    ),
    max_sequence_length: int = typer.Option(8192, "--max-sequence-length", help="Maximum sequence length for FSL"),
    image_token_length: int = typer.Option(64, "--image-token-length", help="Number of image tokens per image"),
    encoder_type: str = typer.Option("hierarchical", "--encoder-type", help="Encoder type for episode tokenizer"),
    image_token: str = typer.Option("<image>", "--image-token", help="Image token string"),
    num_proc: int = typer.Option(32, "--num-proc", help="Number of processes for tokenization"),
):
    """Convert event dataset to FSL dataset format."""
    print(f"Loading event dataset from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Max sequence length: {max_sequence_length}")
    print(f"Image token length: {image_token_length}")

    # Load event dataset
    ds_dict = load_from_disk(str(input_dir))

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
        raise typer.BadParameter(
            f"Input dataset must be EVENT stage, got {first_dataset.owa_config.stage}. "
            "Use 01_raw_events_to_event_dataset.py to create event datasets first."
        )

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure episode tokenizer
    episode_tokenizer_config = EpisodeTokenizerConfig(
        encoder_type=encoder_type,
        image_token_length=image_token_length,
        image_token=image_token,
    )

    # Initialize episode tokenizer
    episode_tokenizer = EpisodeTokenizer(config=episode_tokenizer_config)
    episode_tokenizer.prepare_model(tokenizer=tokenizer)

    # Configure FSL dataset
    fsl_config = FSLDatasetConfig(
        pad_token_id=tokenizer.pad_token_id,
        max_sequence_length=max_sequence_length,
    )

    processed_datasets = {}

    for split in splits:
        ds = ds_dict[split] if split else ds_dict
        split_name = split if split else "train"
        print(f"Processing {len(ds):,} events from {split_name} split")

        # Step 1: Tokenize event dataset
        print(f"Tokenizing {split_name} events...")
        tokenized_dataset = episode_tokenizer.tokenize_event_dataset(ds, map_kwargs={"num_proc": num_proc})
        print(f"Created {len(tokenized_dataset):,} tokenized events")

        # Step 2: Create FSL dataset
        print("Creating FSL dataset from tokenized events...")
        fsl_dataset = precompute_fsl_dataset(tokenized_dataset, config=fsl_config)
        print(f"Created {len(fsl_dataset):,} FSL sequences for {split_name} split")

        processed_datasets[split_name] = fsl_dataset

    # Combine into DatasetDict if multiple splits
    final_dataset = (
        DatasetDict(processed_datasets) if len(processed_datasets) > 1 else list(processed_datasets.values())[0]
    )

    # Save dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving FSL dataset to {output_dir}")
    final_dataset.save_to_disk(str(output_dir))

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
    app()
