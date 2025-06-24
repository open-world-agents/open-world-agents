"""
Minimal OWA Dataset for Vision-Language-Action training.

This module provides a minimal PyTorch Dataset interface for MLLM datasets created by
the OWA data pipeline. It handles lazy loading of images from ScreenCaptured objects
and provides a simple interface for training.
"""

from typing import Any, Dict, List

from torch.utils.data import Dataset

from owa.msgs.desktop.screen import ScreenCaptured


class OWADataset(Dataset):
    """
    Minimal OWA Dataset for Vision-Language-Action training.

    This dataset:
    1. Loads MLLM dataset created by 03_binned_dataset_to_mllm_dataset.py
    2. Deserializes ScreenCaptured objects from bytes
    3. Loads images using ScreenCaptured.to_pil_image()
    4. Returns instruction, images, and encoded_events

    Expected MLLM dataset format:
    {
        'instruction': str,
        'encoded_events': List[str],
        'image_refs': List[bytes],  # Serialized ScreenCaptured objects
        'metadata': Dict
    }
    """

    def __init__(self, mllm_dataset):
        """
        Initialize the OWA dataset.

        Args:
            mllm_dataset: HuggingFace dataset with MLLM format
        """
        self.dataset = mllm_dataset

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
                - images: List of loaded PIL images from image_refs
                - encoded_events: List of encoded action strings
        """
        item = self.dataset[idx]

        # Get instruction and encoded events
        instruction = item["instruction"]
        encoded_events = item["encoded_events"]

        # Load all images from image_refs, passing metadata for path resolution
        images = self._load_images(item["image_refs"], item["metadata"])

        return {
            "instruction": instruction,
            "images": images,
            "encoded_events": encoded_events,
        }

    def _load_images(self, image_refs: List[bytes], metadata: Dict[str, Any]) -> List:
        """
        Load all images from image references.

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
