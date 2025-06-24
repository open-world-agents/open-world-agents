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

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from tqdm import tqdm

from mcap_owa.hf_integration import ScreenCapturedFeature
from owa.data import BaseEventEncoder, HierarchicalEventEncoder, JSONEventEncoder  # noqa: F401

app = typer.Typer(add_completion=False)


def extract_screen_captured_objects(state_sequence: List[Any]) -> List[Any]:
    """
    Extract ScreenCaptured objects from a sequence of McapMessage objects or dictionaries.

    Args:
        state_sequence: List of McapMessage objects or dictionaries from the state field

    Returns:
        List of ScreenCaptured objects
    """
    if state_sequence is None or len(state_sequence) == 0:
        return []

    screen_captured_objects = []

    for item in state_sequence:
        try:
            # Handle both McapMessage objects and dictionaries
            if hasattr(item, "decoded"):
                # It's a McapMessage object
                decoded_msg = item.decoded
            elif isinstance(item, dict):
                # It's already a dictionary (from pandas conversion)
                # Parse the message field if it's bytes
                if "message" in item and isinstance(item["message"], bytes):
                    import json

                    decoded_msg = json.loads(item["message"].decode("utf-8"))
                else:
                    decoded_msg = item
            else:
                continue

            # Check if it's a screen event with the required fields
            if isinstance(decoded_msg, dict) and "path" in decoded_msg and "pts" in decoded_msg:
                # Import ScreenCaptured here to avoid circular imports
                try:
                    from owa.msgs.desktop.screen import ScreenCaptured
                except ImportError:
                    raise ImportError("ScreenCaptured not available. Install owa-msgs package.")

                # Create ScreenCaptured object from the decoded message
                screen_obj = ScreenCaptured(
                    utc_ns=decoded_msg.get("utc_ns"),
                    path=decoded_msg["path"],
                    pts=decoded_msg["pts"],
                    original_shape=tuple(decoded_msg["original_shape"]) if decoded_msg.get("original_shape") else None,
                    shape=tuple(decoded_msg["shape"]) if decoded_msg.get("shape") else None,
                )
                screen_captured_objects.append(screen_obj)
        except Exception:
            # Skip invalid messages
            continue

    return screen_captured_objects


def encode_actions(actions_sequence: List[Any], encoder: BaseEventEncoder) -> List[str]:
    """
    Encode actions using BaseEventEncoder.

    Args:
        actions_sequence: List of McapMessage objects or dictionaries containing action events
        encoder: BaseEventEncoder instance

    Returns:
        List of encoded action strings
    """
    if actions_sequence is None or len(actions_sequence) == 0:
        return []

    encoded_actions = []
    for item in actions_sequence:
        try:
            # Handle both McapMessage objects and dictionaries
            if hasattr(item, "topic"):
                # It's a McapMessage object
                action_event = {
                    "topic": item.topic,
                    "timestamp_ns": item.timestamp,
                    "message_type": item.message_type,
                    "msg": item.message.decode("utf-8") if isinstance(item.message, bytes) else item.message,
                }
            elif isinstance(item, dict):
                # It's already a dictionary (from pandas conversion)
                action_event = {
                    "topic": item.get("topic"),
                    "timestamp_ns": item.get("timestamp"),
                    "message_type": item.get("message_type"),
                    "msg": item.get("message").decode("utf-8")
                    if isinstance(item.get("message"), bytes)
                    else item.get("message"),
                }
            else:
                continue

            # BaseEventEncoder expects this format
            encoded_text, _ = encoder.encode(action_event)
            encoded_actions.append(encoded_text)
        except Exception:
            # Skip invalid actions
            continue

    return encoded_actions


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
    # Extract ScreenCaptured objects from state sequence
    screen_captured_objects = extract_screen_captured_objects(bin_data["state"])
    if not screen_captured_objects:
        # Skip bins without valid screen state
        return None

    # Encode actions
    target_actions = encode_actions(bin_data["actions"], encoder)

    # Filter out samples with no actions if requested
    if filter_empty_actions and len(target_actions) == 0:
        return None

    # Create MLLM sample
    mllm_sample = {
        "instruction": instruction,
        "image_refs": screen_captured_objects,  # Now a list of ScreenCaptured objects
        "encoded_events": target_actions,
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
                if not extract_screen_captured_objects(bin_data["state"]):
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
                "image_refs": Sequence(
                    feature=ScreenCapturedFeature(decode=True), length=-1
                ),  # Sequence of ScreenCaptured
                "encoded_events": Sequence(Value("string")),
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

        # Try creating dataset without explicit features first to see what gets inferred
        try:
            mllm_dataset = Dataset.from_list(all_mllm_samples, features=features)
        except Exception as e:
            typer.echo(f"Error with explicit features: {e}")
            typer.echo("Trying without explicit features...")
            mllm_dataset = Dataset.from_list(all_mllm_samples)

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

            typer.echo("\nImage References:")
            typer.echo(f"  {sample['image_refs']}")

            typer.echo(f"\nEncoded Events ({len(sample['encoded_events'])} total):")
            for i, action in enumerate(sample["encoded_events"]):
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
