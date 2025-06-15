#!/usr/bin/env python3
"""
03_binned_dataset_to_mllm_dataset.py

Convert binned dataset (output of 02_event_dataset_to_binned_dataset.py) into MLLM dataset format.

Usage (CLI):
    python 03_binned_dataset_to_mllm_dataset.py \
        --input_dir /path/to/input_binned_dataset \
        --output_dir /path/to/output_mllm_dataset \
        [--instruction "Complete the computer task"] \
        [--show-example/--no-show-example] \
        [--filter-empty-actions/--no-filter-empty-actions]

- Converts each bin into one training sample (1:1 conversion).
- Extracts image reference from screen event for lazy loading.
- Encodes actions using EventEncoder for text representation.
- Each output row contains: instruction, state_image_ref, target_actions, metadata.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import typer
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from tqdm import tqdm

from owa.data import BaseEventEncoder, HierarchicalEventEncoder, JSONEventEncoder  # noqa: F401

app = typer.Typer(add_completion=False)


def extract_image_reference(state_msg: Union[bytes, str]) -> Optional[Dict[str, Any]]:
    """
    Extract image reference from screen event state message.

    Args:
        state_msg: Message from screen event (bytes or string)

    Returns:
        Dict with image reference info or None if not a screen event
    """
    if state_msg is None:
        return None

    try:
        # Decode the state message
        if isinstance(state_msg, bytes):
            state_data = json.loads(state_msg.decode("utf-8"))
        else:
            state_data = state_msg

        # Check if it's a screen event with path and pts
        if isinstance(state_data, dict) and "path" in state_data and "pts" in state_data:
            return {
                "path": state_data["path"],
                "pts": state_data["pts"],
                "utc_ns": state_data.get("utc_ns"),
            }
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    return None


def encode_actions(actions_msg: Union[bytes, str], encoder: BaseEventEncoder) -> List[str]:
    """
    Encode actions using BaseEventEncoder.

    Args:
        actions_msg: Message containing list of full action events (bytes or string)
        encoder: BaseEventEncoder instance

    Returns:
        List of encoded action strings

    Note:
        This function expects full event objects with all required fields (topic, timestamp_ns,
        message_type, msg) rather than just the message content.
    """
    if actions_msg is None:
        return []

    try:
        # Decode the actions message
        if isinstance(actions_msg, bytes):
            actions_data = json.loads(actions_msg.decode("utf-8"))
        else:
            actions_data = actions_msg

        if not isinstance(actions_data, list):
            return []

        encoded_actions = []
        for action_event in actions_data:
            # action_event should already be a full event dict with all required fields
            if not isinstance(action_event, dict):
                continue

            # Verify required fields are present
            required_fields = {"topic", "timestamp_ns", "message_type", "msg"}
            if not required_fields.issubset(action_event.keys()):
                continue

            # BaseEventEncoder now handles both bytes and string msg formats
            try:
                encoded_text, _ = encoder.encode(action_event)
                encoded_actions.append(encoded_text)
            except Exception:
                # Skip invalid actions
                continue

        return encoded_actions

    except (json.JSONDecodeError, TypeError):
        return []


def convert_bin_to_sample(
    bin_data: Dict[str, Any],
    instruction: str,
    encoder: BaseEventEncoder,
    filter_empty_actions: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Convert a single bin into one MLLM training sample.

    Args:
        bin_data: Single binned data entry
        instruction: Instruction text for the sample
        encoder: BaseEventEncoder instance
        filter_empty_actions: If True, skip samples with no actions

    Returns:
        MLLM training sample or None if no valid state/actions or filtered out
    """
    # Extract image reference from state
    image_ref = extract_image_reference(bin_data["state"])
    if not image_ref:
        # Skip bins without valid screen state
        return None

    # Add bin metadata to image reference
    image_ref["timestamp_ns"] = bin_data["timestamp_ns"]
    image_ref["bin_idx"] = bin_data["bin_idx"]

    # Encode actions
    target_actions = encode_actions(bin_data["actions"], encoder)

    # Filter out samples with no actions if requested
    if filter_empty_actions and len(target_actions) == 0:
        return None

    # Create MLLM sample
    mllm_sample = {
        "instruction": instruction,
        "state_image_ref": image_ref,
        "target_actions": target_actions,
        "metadata": {
            "file_path": bin_data["file_path"],
            "bin_idx": bin_data["bin_idx"],
            "timestamp_ns": bin_data["timestamp_ns"],
            "num_actions": len(target_actions),
        },
    }

    return mllm_sample


@app.command()
def main(
    input_dir: Path = typer.Option(
        ...,
        "--input_dir",
        "-i",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Input binned dataset directory (output of 02_event_dataset_to_binned_dataset.py)",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output_dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Output MLLM dataset directory",
    ),
    instruction: str = typer.Option(
        "Complete the computer task", "--instruction", help="Instruction text for all samples"
    ),
    show_example: bool = typer.Option(
        False, "--show-example/--no-show-example", help="Show full content of an example sample (default: False)"
    ),
    filter_empty_actions: bool = typer.Option(
        True,
        "--filter-empty-actions/--no-filter-empty-actions",
        help="Filter out samples with no actions (default: True)",
    ),
):
    """
    Convert binned dataset to MLLM dataset format with 1:1 bin-to-sample conversion for VLA training.
    """
    start_time = time.time()

    # Initialize HierarchicalEventEncoder
    encoder = HierarchicalEventEncoder()

    typer.echo(f"Loading binned dataset from {input_dir} ...")
    ds_dict = load_from_disk(str(input_dir))

    # Support both DatasetDict and Dataset
    if hasattr(ds_dict, "keys"):
        splits = list(ds_dict.keys())
    else:
        splits = [None]

    # Store all processed datasets
    processed_datasets = {}

    for split in splits:
        if split:
            ds = ds_dict[split]
            split_name = split
        else:
            ds = ds_dict
            split_name = "data"

        typer.echo(f"Processing {split_name} split with {len(ds)} binned entries...")

        # Convert dataset to pandas for much faster processing
        typer.echo("Converting dataset to pandas for faster processing...")
        df = ds.to_pandas()

        all_mllm_samples = []
        valid_samples = 0
        skipped_no_state = 0
        skipped_no_actions = 0

        # Process each bin directly (1:1 conversion)
        bin_pbar = tqdm(df.to_dict("records"), desc=f"Converting {split_name} bins to samples")
        for bin_data in bin_pbar:
            # Convert bin to sample
            sample = convert_bin_to_sample(bin_data, instruction, encoder, filter_empty_actions)

            if sample:
                all_mllm_samples.append(sample)
                valid_samples += 1
            else:
                # Check why it was skipped
                if extract_image_reference(bin_data["state"]) is None:
                    skipped_no_state += 1
                else:
                    # Must be due to empty actions (if filtering is enabled)
                    skipped_no_actions += 1

            # Update progress
            bin_pbar.set_postfix(
                {
                    "valid": valid_samples,
                    "no_state": skipped_no_state,
                    "no_actions": skipped_no_actions,
                    "file": Path(bin_data["file_path"]).name,
                }
            )

        bin_pbar.close()

        total_skipped = skipped_no_state + skipped_no_actions
        typer.echo(f"Converted {valid_samples} bins to samples")
        if total_skipped > 0:
            typer.echo(
                f"Skipped {total_skipped} bins: {skipped_no_state} without state, {skipped_no_actions} without actions"
            )

        # Define features for MLLM dataset
        features = Features(
            {
                "instruction": Value("string"),
                "state_image_ref": {
                    "path": Value("string"),
                    "pts": Value("int64"),
                    "utc_ns": Value("int64"),
                    "timestamp_ns": Value("int64"),
                    "bin_idx": Value("int32"),
                },
                "target_actions": Sequence(Value("string")),
                "metadata": {
                    "file_path": Value("string"),
                    "bin_idx": Value("int32"),
                    "timestamp_ns": Value("int64"),
                    "num_actions": Value("int32"),
                },
            }
        )

        # Create MLLM dataset
        typer.echo(f"Creating MLLM dataset with {len(all_mllm_samples)} samples...")
        mllm_dataset = Dataset.from_list(all_mllm_samples, features=features)

        # Store the dataset for this split
        split_name = split if split else "train"  # Default to "train" if no split
        processed_datasets[split_name] = mllm_dataset

        typer.echo(f"Processed {len(mllm_dataset)} MLLM samples for {split_name} split")

        # Show full example content (if enabled)
        if show_example and len(all_mllm_samples) > 0:
            sample = all_mllm_samples[0]
            typer.echo("\n" + "=" * 80)
            typer.echo("EXAMPLE SAMPLE CONTENT:")
            typer.echo("=" * 80)
            typer.echo("\nInstruction:")
            typer.echo(f"  {sample['instruction']}")

            typer.echo("\nState Image Reference:")
            typer.echo(f"  {sample['state_image_ref']}")

            typer.echo(f"\nTarget Actions ({len(sample['target_actions'])} total):")
            for i, action in enumerate(sample["target_actions"]):
                typer.echo(f"  [{i:3d}] {action}")

            typer.echo("\nMetadata:")
            for key, value in sample["metadata"].items():
                typer.echo(f"  {key}: {value}")

            typer.echo("=" * 80)

    # Save all datasets as DatasetDict or single Dataset
    if len(processed_datasets) > 1:
        # Multiple splits - create DatasetDict
        from datasets import DatasetDict

        final_dataset = DatasetDict(processed_datasets)
    else:
        # Single split - save as Dataset
        final_dataset = list(processed_datasets.values())[0]

    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dataset.save_to_disk(str(output_dir))

    # Calculate and display timing information
    elapsed_time = time.time() - start_time
    if len(processed_datasets) > 1:
        total_samples = sum(len(ds) for ds in processed_datasets.values())
        typer.echo(f"Saved DatasetDict with {total_samples} total MLLM samples to {output_dir}")
        for split_name, ds in processed_datasets.items():
            typer.echo(f"  {split_name}: {len(ds)} samples")
    else:
        split_name = list(processed_datasets.keys())[0]
        ds = list(processed_datasets.values())[0]
        typer.echo(f"Saved {len(ds)} MLLM samples ({split_name}) to {output_dir}")

    typer.echo(f"Processing completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.1f} minutes)")


if __name__ == "__main__":
    app()
