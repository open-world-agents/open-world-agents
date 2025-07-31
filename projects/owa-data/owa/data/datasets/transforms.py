"""Stage-specific transforms for OWA datasets."""

import os
from typing import Optional

from mcap_owa.highlevel import McapMessage
from owa.data.encoders import create_encoder

from .config import DatasetStage


def resolve_episode_path(episode_path: str, mcap_root_directory: Optional[str] = None) -> str:
    """Resolve episode path, raising error if relative path needs mcap_root_directory."""
    if not episode_path or os.path.isabs(episode_path):
        return episode_path

    if not mcap_root_directory:
        raise ValueError(f"mcap_root_directory required for relative path: '{episode_path}'")

    return os.path.join(mcap_root_directory, episode_path)


def create_event_transform(
    encoder_type: str = "hierarchical", load_images: bool = True, mcap_root_directory: Optional[str] = None
):
    """Create transform for EVENT stage."""

    def transform_batch(batch):
        encoder = create_encoder(encoder_type)
        episode_paths = [resolve_episode_path(path, mcap_root_directory) for path in batch.get("episode_path", [])]
        results = {"encoded_event": [], "images": []}

        for i in range(len(batch["mcap_message"])):
            mcap_msg = McapMessage.model_validate_json(batch["mcap_message"][i].decode("utf-8"))
            encoded_text, screen_captured = encoder.encode(mcap_msg)

            images = []
            if batch["topic"][i] == "screen" and screen_captured and load_images:
                for screen in screen_captured:
                    screen.resolve_relative_path(episode_paths[i])
                    images.append(screen.to_pil_image())

            results["encoded_event"].append(encoded_text)
            results["images"].append(images)

        return results

    return transform_batch


def create_binned_transform(
    instruction: str = "Complete the computer task",
    encoder_type: str = "hierarchical",
    load_images: bool = True,
    encode_actions: bool = True,
    mcap_root_directory: Optional[str] = None,
):
    """Create transform for BINNED stage."""

    def transform_batch(batch):
        encoder = create_encoder(encoder_type) if encode_actions else None
        episode_paths = [resolve_episode_path(path, mcap_root_directory) for path in batch.get("episode_path", [])]
        batch_size = len(batch[list(batch.keys())[0]])
        state, actions = [], []
        for i in range(batch_size):
            _state, _action = [], []
            for msg in batch["state"][i]:
                mcap_msg = McapMessage.model_validate_json(msg.decode("utf-8"))
                if mcap_msg.message_type == "desktop/ScreenCaptured":
                    screen = mcap_msg.decoded
                    screen.resolve_relative_path(episode_paths[i])
                screen_value = screen.to_pil_image() if load_images else screen
                _state.append(screen_value)
            for msg in batch["actions"][i]:
                mcap_msg = McapMessage.model_validate_json(msg.decode("utf-8"))
                if encode_actions:
                    action, image = encoder.encode(mcap_msg)
                    assert len(image) == 0, "Action encoding should not produce images"
                    _action.append(action)
                else:
                    _action.append(mcap_msg)
            if encode_actions:
                _action = "".join(_action)
            state.append(_state)
            actions.append(_action)

        return {
            "instruction": [instruction] * batch_size,
            "state": state,
            "actions": actions,
        }

    return transform_batch


def create_tokenized_transform():
    """Create transform for TOKENIZED stage."""
    return lambda batch: batch


def create_fsl_transform(
    image_processor=None,
    load_images: bool = True,
    mcap_root_directory: Optional[str] = None,
):
    """Create transform for FSL stage - handles image loading on-the-fly."""
    import concurrent.futures
    import warnings

    import torch

    from owa.msgs.desktop.screen import ScreenCaptured

    is_decoding_server_available = "VIDEO_DECODING_SERVER_URL" in os.environ

    def transform_batch(batch):
        batch_size = len(batch["input_ids"])
        results = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "texts": batch["texts"],
            "images": [],
        }

        for i in range(batch_size):
            image_msgs_json = batch["images"][i]
            episode_path = resolve_episode_path(batch["episode_path"][i], mcap_root_directory)

            # Deserialize ScreenCaptured messages
            image_msgs = [
                ScreenCaptured.model_validate_json(img_json).resolve_relative_path(episode_path)
                for img_json in image_msgs_json
            ]

            if not load_images:
                results["images"].append(image_msgs)
                continue

            # Preload images in parallel if decoding server is available
            if is_decoding_server_available and image_msgs:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(img.to_pil_image) for img in image_msgs]
                    for idx, future in enumerate(futures):
                        try:
                            future.result(timeout=30)
                        except Exception as e:
                            import numpy as np

                            image_msgs[idx].frame_arr = np.zeros((512, 512, 3), dtype=np.uint8)
                            warnings.warn(f"Failed to load image at index {idx}: {e}. Using placeholder.", UserWarning)

            # Convert to PIL images
            all_images = [img.to_pil_image() for img in image_msgs]

            # Process with image processor if available
            if image_processor is not None and all_images:
                pixel_values = []
                for image in all_images:
                    processed = image_processor(image, return_tensors="pt")
                    pixel_value = processed["pixel_values"].squeeze(0).squeeze(0)
                    pixel_values.append(pixel_value)
                results["images"].append(torch.stack(pixel_values) if pixel_values else torch.empty(0, 3, 224, 224))
            else:
                results["images"].append(all_images)

        return results

    return transform_batch


def create_transform(stage: str, mcap_root_directory: str, **kwargs):
    """Create a transform function for a given stage."""
    if stage == DatasetStage.EVENT:
        return create_event_transform(mcap_root_directory=mcap_root_directory, **kwargs)
    elif stage == DatasetStage.BINNED:
        return create_binned_transform(mcap_root_directory=mcap_root_directory, **kwargs)
    elif stage == DatasetStage.TOKENIZED:
        return create_tokenized_transform(**kwargs)
    elif stage == DatasetStage.FSL:
        return create_fsl_transform(mcap_root_directory=mcap_root_directory, **kwargs)
    else:
        raise ValueError(f"Unknown dataset stage: {stage}")
