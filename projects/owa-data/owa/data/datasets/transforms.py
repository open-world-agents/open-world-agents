"""Stage-specific transforms for OWA datasets."""

from typing import Optional

from .config import DatasetStage


def create_event_transform(encoder_type: str = "hierarchical", load_images: bool = True):
    """Create transform for EVENT stage - converts raw MCAP events to processed format."""

    def transform_batch(batch):
        from mcap_owa.highlevel import McapMessage
        from owa.data.encoders import create_encoder

        encoder = create_encoder(encoder_type)
        results = {"encoded_event": [], "image": []}

        for i in range(len(batch["mcap_message"])):
            example = {key: batch[key][i] for key in batch.keys()}
            result = _transform_event_single(example, encoder, load_images, McapMessage)
            results["encoded_event"].append(result["encoded_event"])
            results["image"].append(result["image"])

        return results

    return transform_batch


def create_binned_transform(
    instruction: str = "Complete the computer task",
    encoder_type: str = "hierarchical",
    load_images: bool = True,
    encode_actions: bool = True,
):
    """Create transform for BINNED stage - converts binned data to VLA format."""

    def transform_batch(batch):
        from owa.data.encoders import create_encoder

        encoder = create_encoder(encoder_type) if encode_actions else None
        results = {"instruction": [], "state": [], "actions": []}

        for i in range(len(batch[list(batch.keys())[0]])):
            example = {key: batch[key][i] for key in batch.keys()}
            result = _transform_binned_single(example, instruction, encoder, load_images)
            results["instruction"].append(result["instruction"])
            results["state"].append(result["state"])
            results["actions"].append(result["actions"])

        return results

    return transform_batch


def create_tokenized_transform():
    """Create transform for TOKENIZED stage - basic passthrough."""

    def transform_batch(batch):
        return batch

    return transform_batch


def create_fsl_transform(max_sequence_length: int = 8192, pad_token_id: int = 0, load_images: bool = True):
    """Create transform for FSL stage - fixed sequence length packing."""
    import numpy as np
    import torch

    # State for FSL transform
    _prepared = False
    _cumsum = None

    def prepare_fsl(dataset):
        """Prepare FSL dataset by computing cumulative token counts."""
        nonlocal _prepared, _cumsum
        total_token_counts = dataset["total_token_count"]
        _cumsum = np.cumsum(total_token_counts)
        _prepared = True

    def fsl_getitem(dataset, idx):
        """Get FSL training sequence by concatenating events up to max_sequence_length."""
        nonlocal _prepared, _cumsum

        if not _prepared:
            raise RuntimeError("FSL transform must be prepared first. Call prepare_fsl(dataset).")

        start_token_index = idx * max_sequence_length
        start_event_index = np.searchsorted(_cumsum, start_token_index, side="left")

        all_token_ids = []
        all_image_msgs = []
        texts = []
        tokens_so_far = 0

        for event_idx in range(start_event_index, len(dataset)):
            event = dataset[event_idx]
            token_ids = event["token_ids"]
            total_token_count = event["total_token_count"]

            if tokens_so_far + total_token_count > max_sequence_length:
                break

            all_token_ids.extend(token_ids)
            texts.append(event["text"])
            tokens_so_far += total_token_count

            # Handle images if available
            if load_images and "images" in event:
                import json

                images = event["images"]
                if isinstance(images, str):
                    screen_images = json.loads(images)
                else:
                    screen_images = images  # Already a list
                all_image_msgs.extend(screen_images)

        # Pad to max_sequence_length
        if tokens_so_far < max_sequence_length:
            padding_length = max_sequence_length - tokens_so_far
            all_token_ids.extend([pad_token_id] * padding_length)
            tokens_so_far += padding_length

        assert len(all_token_ids) == max_sequence_length == tokens_so_far

        # Create result
        result = {
            "texts": "".join(texts),
            "input_ids": torch.tensor(all_token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if token_id != pad_token_id else 0 for token_id in all_token_ids], dtype=torch.long
            ),
            "images": all_image_msgs,
        }

        return result

    def fsl_len(dataset):
        """Calculate FSL dataset length."""
        nonlocal _prepared, _cumsum
        if _prepared and _cumsum is not None:
            total_tokens = _cumsum[-1]
            return max(1, total_tokens // max_sequence_length)
        else:
            return len(dataset)

    return {"prepare": prepare_fsl, "getitem": fsl_getitem, "len": fsl_len}


def _transform_event_single(example, encoder, load_images, McapMessage):
    """Transform a single event example."""
    result = {"encoded_event": None, "image": None}
    try:
        mcap_msg = McapMessage.model_validate_json(example["mcap_message"].decode("utf-8"))
        encoded_text, screen_captured = encoder.encode(mcap_msg)

        # Only try to load images if load_images=True and we have screen events
        if example["topic"] == "screen" and screen_captured and load_images:
            result["image"] = (
                screen_captured[0].to_pil_image() if hasattr(screen_captured[0], "to_pil_image") else None
            )

        result["encoded_event"] = encoded_text
    except Exception as e:
        print(f"Warning: Could not process {example['topic']} event: {e}")

    return result


def _transform_binned_single(example, instruction, encoder, load_images):
    """Transform a single binned example."""
    result = {"instruction": instruction, "state": [], "actions": []}

    if load_images:
        state_sequence = example.get("state", [])
        result["state"] = [f"state_image_{i}" for i in range(len(state_sequence))]

    if encoder:
        actions_sequence = example.get("actions", [])
        result["actions"] = [f"action_{i}" for i in range(len(actions_sequence))]

    return result


def create_transform(stage: str, mcap_root_directory: str, **kwargs):
    """Create a transform function for a given stage."""
    if stage == DatasetStage.EVENT:
        return create_event_transform(**kwargs)
    elif stage == DatasetStage.BINNED:
        return create_binned_transform(**kwargs)
    elif stage == DatasetStage.TOKENIZED:
        return create_tokenized_transform(**kwargs)
    elif stage == DatasetStage.FSL:
        return create_fsl_transform(**kwargs)
    else:
        raise ValueError(f"Unknown dataset stage: {stage}")
