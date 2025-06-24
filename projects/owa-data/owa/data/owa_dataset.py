"""
VLA Dataset for Vision-Language-Action training.

This module provides PyTorch Dataset interfaces for VLA training, supporting both
pre-converted MLLM datasets and on-the-fly conversion from binned datasets.
"""

from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from mcap_owa.highlevel import McapMessage
from owa.data.encoders import BaseEventEncoder
from owa.msgs.desktop.screen import ScreenCaptured


def create_encoder(encoder_type: str) -> BaseEventEncoder:
    """Create an encoder instance based on the specified type."""
    from owa.data.encoders import FlatEventEncoder, HierarchicalEventEncoder, JSONEventEncoder

    encoder_type = encoder_type.lower()

    if encoder_type == "hierarchical":
        return HierarchicalEventEncoder()
    elif encoder_type == "json":
        return JSONEventEncoder()
    elif encoder_type == "flat":
        return FlatEventEncoder()
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}. Supported types: hierarchical, json, flat")


class VLADataset(Dataset):
    """
    Vision-Language-Action Dataset with on-the-fly conversion from binned data.

    This dataset works with binned datasets and converts them on-the-fly to MLLM format.

    Key features:
    - Configurable instruction text
    - Selectable event encoders (hierarchical, json, flat)
    - Lazy image loading and action encoding
    - Optional sample caching for performance
    - Unified interface for VLA training
    """

    def __init__(
        self,
        dataset,
        instruction: str = "Complete the computer task",
        encoder: Optional[BaseEventEncoder] = None,
        encoder_type: str = "hierarchical",
        cache_samples: bool = False,
    ):
        """
        Initialize the VLA dataset.

        Args:
            dataset: HuggingFace dataset in binned format
            instruction: Instruction text for all samples
            encoder: Event encoder for action serialization (takes precedence over encoder_type)
            encoder_type: Type of encoder to create if encoder is None ('hierarchical', 'json', 'flat')
            cache_samples: Whether to cache converted samples for performance
        """
        self.dataset = dataset
        self.instruction = instruction
        self.encoder = encoder or create_encoder(encoder_type)
        self.cache_samples = cache_samples

        # Determine dataset format by checking first sample
        if len(dataset) > 0:
            sample = dataset[0]
            self.is_mllm_format = "encoded_events" in sample
        else:
            self.is_mllm_format = False

        # Initialize cache if enabled
        self._cache = {} if cache_samples else None

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample by index.

        Args:
            idx: Index of the sample

        Returns:
            Dict containing:
                - instruction: Task instruction
                - images: List of loaded PIL images
                - encoded_events: List of encoded action strings
        """
        # Check cache first
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        item = self.dataset[idx]

        if self.is_mllm_format:
            # Pre-converted MLLM format
            result = self._process_mllm_sample(item)
        else:
            # Binned format - convert on-the-fly
            result = self._process_binned_sample(item)

        # Cache result if enabled
        if self._cache is not None:
            self._cache[idx] = result

        return result

    def _process_mllm_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a pre-converted MLLM sample."""
        # Use existing instruction or override with configured one
        instruction = (
            self.instruction
            if self.instruction != "Complete the computer task"
            else item.get("instruction", self.instruction)
        )

        # Load images
        images = self._load_images(item["image_refs"], item.get("metadata", {}))

        return {
            "instruction": instruction,
            "images": images,
            "encoded_events": item["encoded_events"],
        }

    def _process_binned_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a binned sample with on-the-fly conversion."""
        # Extract and load images from state
        images = self._load_images_from_state(item["state"], item)

        # Encode actions
        encoded_events = self._encode_actions(item["actions"])

        return {
            "instruction": self.instruction,
            "images": images,
            "encoded_events": encoded_events,
        }

    def _load_images_from_state(self, state_sequence: List[bytes], metadata: Dict[str, Any]) -> List:
        """Load images from state sequence (serialized McapMessage bytes)."""
        images = []

        for state_bytes in state_sequence:
            try:
                # Deserialize McapMessage
                mcap_msg = McapMessage.model_validate_json(state_bytes.decode("utf-8"))

                # Extract ScreenCaptured from message
                if hasattr(mcap_msg.msg, "model_validate_json"):
                    screen_captured = ScreenCaptured.model_validate_json(mcap_msg.msg)
                else:
                    screen_captured = ScreenCaptured.model_validate(mcap_msg.msg)

                # Resolve path and load image
                screen_captured = self._resolve_video_path(screen_captured, metadata)
                image = screen_captured.to_pil_image()

                if image is not None:
                    images.append(image)

            except Exception as e:
                print(f"Warning: Could not load image from state: {e}")
                continue

        return images

    def _encode_actions(self, actions_sequence: List[bytes]) -> List[str]:
        """Encode actions using the configured encoder."""
        if not actions_sequence:
            return []

        encoded_actions = []

        for action_bytes in actions_sequence:
            try:
                # Deserialize McapMessage
                mcap_msg = McapMessage.model_validate_json(action_bytes.decode("utf-8"))

                # Encode using EventEncoder
                encoded_text, _ = self.encoder.encode(mcap_msg)
                encoded_actions.append(encoded_text)

            except Exception as e:
                print(f"Warning: Could not encode action: {e}")
                continue

        return encoded_actions

    def _load_images(self, image_refs: List[bytes], metadata: Dict[str, Any]) -> List:
        """
        Load all images from image references (for MLLM format).

        Args:
            image_refs: List of serialized ScreenCaptured bytes
            metadata: Sample metadata containing file_path for path resolution

        Returns:
            List of PIL Images (skips failed loads)
        """
        images = []

        for img_ref_bytes in image_refs:
            try:
                # Deserialize ScreenCaptured from bytes
                screen_captured = ScreenCaptured.model_validate_json(img_ref_bytes.decode("utf-8"))

                # Resolve relative path using metadata
                screen_captured = self._resolve_video_path(screen_captured, metadata)

                # Load image using ScreenCaptured
                image = screen_captured.to_pil_image()
                if image is not None:
                    images.append(image)

            except Exception as e:
                print(f"Warning: Could not load image from {img_ref_bytes}: {e}")
                continue

        return images

    def _resolve_video_path(self, screen_captured: ScreenCaptured, metadata: Dict[str, Any]) -> ScreenCaptured:
        """
        Resolve relative video path using metadata.

        Args:
            screen_captured: ScreenCaptured object with potentially relative path
            metadata: Sample metadata containing file_path

        Returns:
            ScreenCaptured object with resolved absolute path
        """
        from pathlib import Path

        # If path is already absolute, return as-is
        if Path(screen_captured.path).is_absolute():
            return screen_captured

        # Get the directory of the original MCAP file
        file_path = metadata.get("file_path", "")
        if file_path:
            mcap_dir = Path(file_path).parent
            # Resolve relative path relative to MCAP directory
            resolved_path = mcap_dir / screen_captured.path

            # Create new ScreenCaptured with resolved path
            return ScreenCaptured(
                utc_ns=screen_captured.utc_ns,
                path=str(resolved_path),
                pts=screen_captured.pts,
                original_shape=screen_captured.original_shape,
                shape=screen_captured.shape,
            )

        # If no file_path in metadata, return as-is
        return screen_captured
