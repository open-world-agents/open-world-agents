#!/usr/bin/env python3
"""
03_binned_dataset_to_mllm_dataset.py

Convert binned dataset (output of 02_event_dataset_to_binned_dataset.py) into MLLM dataset format.

Usage (CLI):
    python 03_binned_dataset_to_mllm_dataset.py \
        --input_dir /path/to/input_binned_dataset \
        --output_dir /path/to/output_mllm_dataset \
        [--sequence_length 32] \
        [--instruction "Complete the computer task"] \
        [--overlap_ratio 0.5]

- Groups bins into sequences of specified length for MLLM training.
- Extracts image references from screen events for lazy loading.
- Encodes actions using EventEncoder for text representation.
- Each output row contains: instruction, encoded_events, image_refs, metadata.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from datasets import Dataset, Features, Value, Sequence, load_from_disk
from tqdm import tqdm

from owa.data.event_encoder import EventEncoder

app = typer.Typer(add_completion=False)


def extract_image_references(state_msg: bytes) -> Optional[Dict[str, Any]]:
    """
    Extract image reference from screen event state message.
    
    Args:
        state_msg: Binary message from screen event
        
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


def encode_actions(actions_msg: bytes, encoder: EventEncoder) -> List[str]:
    """
    Encode actions using EventEncoder.
    
    Args:
        actions_msg: Binary message containing list of action events
        encoder: EventEncoder instance
        
    Returns:
        List of encoded action strings
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
        for action_msg in actions_data:
            # Reconstruct event format for EventEncoder
            # Note: We need to reconstruct the full event format
            # This is a simplified version - in practice you might need more metadata
            fake_event = {
                "topic": "action",  # Generic topic for actions
                "timestamp_ns": 0,  # Will be overridden by sequence timestamp
                "message_type": "generic.action",
                "msg": action_msg.encode("utf-8") if isinstance(action_msg, str) else action_msg
            }
            
            try:
                encoded_text, _ = encoder.encode(fake_event)
                encoded_actions.append(encoded_text)
            except Exception:
                # Skip invalid actions
                continue
                
        return encoded_actions
        
    except (json.JSONDecodeError, TypeError):
        return []


def create_sequences(
    binned_data: List[Dict[str, Any]], 
    sequence_length: int,
    overlap_ratio: float,
    instruction: str,
    encoder: EventEncoder
) -> List[Dict[str, Any]]:
    """
    Create sequences from binned data for MLLM training.
    
    Args:
        binned_data: List of binned data entries
        sequence_length: Number of bins per sequence
        overlap_ratio: Overlap ratio between sequences (0.0 to 1.0)
        instruction: Instruction text for all sequences
        encoder: EventEncoder instance
        
    Returns:
        List of MLLM training sequences
    """
    if len(binned_data) < sequence_length:
        # If not enough data, create one sequence with available data
        sequences = [binned_data]
    else:
        # Create overlapping sequences
        step_size = max(1, int(sequence_length * (1 - overlap_ratio)))
        sequences = []
        
        for start_idx in range(0, len(binned_data) - sequence_length + 1, step_size):
            end_idx = start_idx + sequence_length
            sequences.append(binned_data[start_idx:end_idx])
    
    mllm_sequences = []
    
    for seq_idx, sequence in enumerate(sequences):
        # Extract image references and encoded events
        image_refs = []
        encoded_events = []
        
        for bin_data in sequence:
            # Extract image reference from state
            img_ref = extract_image_references(bin_data["state"])
            if img_ref:
                img_ref["timestamp_ns"] = bin_data["timestamp_ns"]
                img_ref["bin_idx"] = bin_data["bin_idx"]
                image_refs.append(img_ref)
            
            # Encode actions
            actions = encode_actions(bin_data["actions"], encoder)
            encoded_events.extend(actions)
        
        # Create MLLM sequence
        mllm_sequence = {
            "instruction": instruction,
            "encoded_events": encoded_events,
            "image_refs": image_refs,
            "metadata": {
                "file_path": sequence[0]["file_path"],
                "sequence_idx": seq_idx,
                "start_bin_idx": sequence[0]["bin_idx"],
                "end_bin_idx": sequence[-1]["bin_idx"],
                "start_timestamp_ns": sequence[0]["timestamp_ns"],
                "end_timestamp_ns": sequence[-1]["timestamp_ns"],
                "num_bins": len(sequence),
                "num_images": len(image_refs),
                "num_actions": len(encoded_events),
            }
        }
        
        mllm_sequences.append(mllm_sequence)
    
    return mllm_sequences


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
    sequence_length: int = typer.Option(32, "--sequence_length", help="Number of bins per sequence (default: 32)"),
    instruction: str = typer.Option(
        "Complete the computer task", 
        "--instruction", 
        help="Instruction text for all sequences"
    ),
    overlap_ratio: float = typer.Option(
        0.5, 
        "--overlap_ratio", 
        help="Overlap ratio between sequences (0.0 to 1.0, default: 0.5)"
    ),
):
    """
    Convert binned dataset to MLLM dataset format with sequences ready for VLA training.
    """
    start_time = time.time()
    
    # Validate overlap_ratio
    if not 0.0 <= overlap_ratio < 1.0:
        typer.echo("[Error] --overlap_ratio must be between 0.0 and 1.0 (exclusive)", err=True)
        raise typer.Exit(code=1)
    
    # Initialize EventEncoder
    encoder = EventEncoder(drop_file_path=True)
    
    typer.echo(f"Loading binned dataset from {input_dir} ...")
    ds_dict = load_from_disk(str(input_dir))
    
    # Support both DatasetDict and Dataset
    if hasattr(ds_dict, "keys"):
        splits = list(ds_dict.keys())
    else:
        splits = [None]
    
    for split in splits:
        if split:
            ds = ds_dict[split]
            split_name = split
        else:
            ds = ds_dict
            split_name = "data"
        
        typer.echo(f"Processing {split_name} split with {len(ds)} binned entries...")
        
        # Group by file_path
        file_paths = sorted(set(ds["file_path"]))
        all_mllm_sequences = []
        
        typer.echo(f"Found {len(file_paths)} unique files to process")
        
        # Process each file
        file_pbar = tqdm(file_paths, desc=f"Processing {split_name} files")
        for fp in file_pbar:
            file_pbar.set_postfix({"current_file": Path(fp).name})
            
            # Get all bins for this file
            file_ds = ds.filter(lambda example: example["file_path"] == fp)
            
            # Convert to list of dicts
            binned_data = []
            for i in range(len(file_ds)):
                bin_data = {k: file_ds[k][i] for k in file_ds.column_names}
                binned_data.append(bin_data)
            
            # Sort by bin_idx to ensure correct order
            binned_data.sort(key=lambda x: x["bin_idx"])
            
            # Create sequences for this file
            file_sequences = create_sequences(
                binned_data, sequence_length, overlap_ratio, instruction, encoder
            )
            all_mllm_sequences.extend(file_sequences)
            
            # Update progress
            file_pbar.set_postfix({
                "current_file": Path(fp).name,
                "bins": len(binned_data),
                "sequences": len(file_sequences)
            })
        
        file_pbar.close()
        
        # Define features for MLLM dataset
        features = Features({
            "instruction": Value("string"),
            "encoded_events": Sequence(Value("string")),
            "image_refs": Sequence({
                "path": Value("string"),
                "pts": Value("int64"),
                "utc_ns": Value("int64"),
                "timestamp_ns": Value("int64"),
                "bin_idx": Value("int32"),
            }),
            "metadata": {
                "file_path": Value("string"),
                "sequence_idx": Value("int32"),
                "start_bin_idx": Value("int32"),
                "end_bin_idx": Value("int32"),
                "start_timestamp_ns": Value("int64"),
                "end_timestamp_ns": Value("int64"),
                "num_bins": Value("int32"),
                "num_images": Value("int32"),
                "num_actions": Value("int32"),
            }
        })
        
        # Create MLLM dataset
        typer.echo(f"Creating MLLM dataset with {len(all_mllm_sequences)} sequences...")
        mllm_dataset = Dataset.from_list(all_mllm_sequences, features=features)
        
        # Save
        out_path = output_dir / split_name
        out_path.mkdir(parents=True, exist_ok=True)
        mllm_dataset.save_to_disk(str(out_path))
        
        # Calculate and display timing information
        elapsed_time = time.time() - start_time
        typer.echo(f"Saved {len(mllm_dataset)} MLLM sequences to {out_path}")
        typer.echo(f"Processing completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.1f} minutes)")
        
        # Show sample statistics
        if len(all_mllm_sequences) > 0:
            sample = all_mllm_sequences[0]
            typer.echo(f"\nSample sequence statistics:")
            typer.echo(f"  Instruction: {sample['instruction']}")
            typer.echo(f"  Encoded events: {len(sample['encoded_events'])}")
            typer.echo(f"  Image references: {len(sample['image_refs'])}")
            typer.echo(f"  Metadata: {sample['metadata']}")


if __name__ == "__main__":
    app()
