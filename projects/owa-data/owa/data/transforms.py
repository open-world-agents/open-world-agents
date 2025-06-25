"""
Dataset transforms for Event Dataset and Binned Dataset.

This module provides transform functions that can be applied to HuggingFace datasets
using set_transform to enable event encoding and image loading on-the-fly.
These transforms extract the core functionality from VLADataset for broader reuse.
"""

from typing import Any, Dict, List, Optional, Union

from mcap_owa.highlevel import McapMessage
from owa.data.encoders import BaseEventEncoder
from owa.data.owa_dataset import create_encoder
from owa.msgs.desktop.screen import ScreenCaptured


def create_event_dataset_transform(
    encoder: Optional[BaseEventEncoder] = None,
    encoder_type: str = "hierarchical",
    load_images: bool = True,
    encode_actions: bool = True,
    keep_original: bool = False,
):
    """
    Create a transform function for Event Dataset that can be used with dataset.set_transform().

    Args:
        encoder: Event encoder for action serialization (takes precedence over encoder_type)
        encoder_type: Type of encoder to create if encoder is None ('hierarchical', 'json', 'flat')
        load_images: Whether to load images for screen events
        encode_actions: Whether to encode action events (keyboard/mouse)
        keep_original: Whether to keep original fields in the output (default: False)

    Returns:
        Transform function that can be used with dataset.set_transform()

    Example:
        ```python
        from datasets import load_from_disk
        from owa.data import create_event_dataset_transform

        # Load event dataset
        event_dataset = load_from_disk("/path/to/event/dataset")

        # Create and apply transform
        transform = create_event_dataset_transform(
            encoder_type="hierarchical",
            load_images=True,  # Load images for screen events
            encode_actions=True,  # Encode keyboard/mouse events
        )
        event_dataset.set_transform(transform)

        # Use transformed dataset
        for sample in event_dataset["train"].take(10):
            print(f"{sample=}")
        ```
    """
    # Initialize encoder once
    event_encoder = encoder or create_encoder(encoder_type)

    def transform_examples(examples: Union[Dict[str, Any], Dict[str, List[Any]]]) -> Dict[str, Any]:
        """
        Transform function for Event Dataset.

        Handles both single examples and batches of examples.
        """
        # Check if this is a batch (lists) or single example
        is_batch = isinstance(examples.get("file_path", ""), list)

        if is_batch:
            return _transform_event_batch(examples, event_encoder, load_images, encode_actions, keep_original)
        else:
            return _transform_event_single(examples, event_encoder, load_images, encode_actions, keep_original)

    return transform_examples


def create_binned_dataset_transform(
    instruction: str = "Complete the computer task",
    encoder: Optional[BaseEventEncoder] = None,
    encoder_type: str = "hierarchical",
    load_images: bool = True,
    encode_actions: bool = True,
    keep_original: bool = False,
):
    """
    Create a transform function for Binned Dataset that can be used with dataset.set_transform().

    Args:
        instruction: Instruction text for all samples
        encoder: Event encoder for action serialization (takes precedence over encoder_type)
        encoder_type: Type of encoder to create if encoder is None ('hierarchical', 'json', 'flat')
        load_images: Whether to load images from state sequence
        encode_actions: Whether to encode action events
        keep_original: Whether to keep original fields in the output (default: False)

    Returns:
        Transform function that can be used with dataset.set_transform()

    Example:
        ```python
        from datasets import load_from_disk
        from owa.data import create_binned_dataset_transform

        # Load binned dataset
        binned_dataset = load_from_disk("/path/to/binned/dataset")

        # Create and apply transform
        transform = create_binned_dataset_transform(
            encoder_type="hierarchical",
            instruction="Complete the computer task",
            load_images=True,
            encode_actions=True,
        )
        binned_dataset.set_transform(transform)

        # Use transformed dataset (same format as VLADataset)
        for sample in binned_dataset["train"].take(10):
            print(f"{sample=}")
        ```
    """
    # Initialize encoder once
    event_encoder = encoder or create_encoder(encoder_type)

    def transform_examples(examples: Union[Dict[str, Any], Dict[str, List[Any]]]) -> Dict[str, Any]:
        """
        Transform function for Binned Dataset.

        Handles both single examples and batches of examples.
        """
        # Check if this is a batch (lists) or single example
        is_batch = isinstance(examples.get("file_path", ""), list)

        if is_batch:
            return _transform_binned_batch(
                examples, instruction, event_encoder, load_images, encode_actions, keep_original
            )
        else:
            return _transform_binned_single(
                examples, instruction, event_encoder, load_images, encode_actions, keep_original
            )

    return transform_examples


def _transform_event_single(
    example: Dict[str, Any],
    encoder: BaseEventEncoder,
    load_images: bool,
    encode_actions: bool,
    keep_original: bool,
) -> Dict[str, Any]:
    """Transform a single Event Dataset example."""
    if keep_original:
        result = example.copy()
    else:
        result = {}

    # Initialize new fields
    result["encoded_event"] = None
    result["image"] = None

    topic = example["topic"]
    mcap_message_bytes = example["mcap_message"]

    try:
        # Deserialize McapMessage
        mcap_msg = McapMessage.model_validate_json(mcap_message_bytes.decode("utf-8"))
        if topic == "screen" and load_images:
            # Load image for screen events
            screen_captured = ScreenCaptured.model_validate_json(mcap_msg.message)

            # Resolve path and load image
            screen_captured = _resolve_video_path(screen_captured, example)
            image = screen_captured.to_pil_image()
            result["image"] = image

        elif topic in ["keyboard", "mouse"] and encode_actions:
            # Encode action events
            encoded_text, _ = encoder.encode(mcap_msg)
            result["encoded_event"] = encoded_text

    except Exception as e:
        print(f"Warning: Could not process {topic} event: {e}")

    return result


def _transform_event_batch(
    examples: Dict[str, List[Any]],
    encoder: BaseEventEncoder,
    load_images: bool,
    encode_actions: bool,
    keep_original: bool,
) -> Dict[str, Any]:
    """Transform a batch of Event Dataset examples."""
    batch_size = len(examples["file_path"])

    # Initialize result with original data
    if keep_original:
        result = examples.copy()
    else:
        result = {}
    result["encoded_event"] = [None] * batch_size
    result["image"] = [None] * batch_size

    for i in range(batch_size):
        single_example = {key: values[i] for key, values in examples.items()}
        transformed = _transform_event_single(single_example, encoder, load_images, encode_actions, keep_original)
        result["encoded_event"][i] = transformed["encoded_event"]
        result["image"][i] = transformed["image"]

    return result


def _transform_binned_single(
    example: Dict[str, Any],
    instruction: str,
    encoder: BaseEventEncoder,
    load_images: bool,
    encode_actions: bool,
    keep_original: bool,
) -> Dict[str, Any]:
    """Transform a single Binned Dataset example."""
    result = {
        "instruction": instruction,
        "images": [],
        "encoded_events": [],
    }

    # Keep original fields if requested
    if keep_original:
        result.update(example)

    if load_images:
        # Load images from state sequence
        state_sequence = example.get("state", [])
        images = _load_images_from_state(state_sequence, example)
        result["images"] = images

    if encode_actions:
        # Encode actions from actions sequence
        actions_sequence = example.get("actions", [])
        encoded_events = _encode_actions(actions_sequence, encoder)
        result["encoded_events"] = encoded_events

    return result


def _transform_binned_batch(
    examples: Dict[str, List[Any]],
    instruction: str,
    encoder: BaseEventEncoder,
    load_images: bool,
    encode_actions: bool,
    keep_original: bool,
) -> Dict[str, Any]:
    """Transform a batch of Binned Dataset examples."""
    batch_size = len(examples["file_path"])

    # Initialize result
    if keep_original:
        result = examples.copy()
    else:
        result = {}
    result["instruction"] = [instruction] * batch_size
    result["images"] = []
    result["encoded_events"] = []

    for i in range(batch_size):
        single_example = {key: values[i] for key, values in examples.items()}
        transformed = _transform_binned_single(
            single_example, instruction, encoder, load_images, encode_actions, keep_original
        )
        result["images"].append(transformed["images"])
        result["encoded_events"].append(transformed["encoded_events"])

    return result


def _load_images_from_state(state_sequence: List[bytes], metadata: Dict[str, Any]) -> List:
    """Load images from state sequence (serialized McapMessage bytes)."""
    images = []

    for state_bytes in state_sequence:
        try:
            # Deserialize McapMessage
            mcap_msg = McapMessage.model_validate_json(state_bytes.decode("utf-8"))

            # Extract ScreenCaptured from message
            screen_captured = ScreenCaptured.model_validate_json(mcap_msg.message)

            # Resolve path and load image
            screen_captured = _resolve_video_path(screen_captured, metadata)
            image = screen_captured.to_pil_image()

            if image is not None:
                images.append(image)

        except Exception as e:
            print(f"Warning: Could not load image from state: {e}")
            continue

    return images


def _encode_actions(actions_sequence: List[bytes], encoder: BaseEventEncoder) -> List[str]:
    """Encode actions using the configured encoder."""
    if not actions_sequence:
        return []
    encoded_actions = []

    for action_bytes in actions_sequence:
        try:
            # Deserialize McapMessage
            mcap_msg = McapMessage.model_validate_json(action_bytes.decode("utf-8"))

            # Encode using EventEncoder
            encoded_text, _ = encoder.encode(mcap_msg)
            encoded_actions.append(encoded_text)

        except Exception as e:
            print(f"Warning: Could not encode action: {e}")
            continue

    return encoded_actions


def _resolve_video_path(screen_captured: ScreenCaptured, metadata: Dict[str, Any]) -> ScreenCaptured:
    """
    Resolve relative video path using metadata.

    Args:
        screen_captured: ScreenCaptured object with potentially relative path
        metadata: Sample metadata containing file_path

    Returns:
        ScreenCaptured object with resolved absolute path
    """
    if screen_captured.media_ref is not None:
        if screen_captured.media_ref.type == "external_video":
            screen_captured.resolve_external_path(metadata.get("file_path"))

    # For other media_ref types or if no file_path, return as-is
    return screen_captured
