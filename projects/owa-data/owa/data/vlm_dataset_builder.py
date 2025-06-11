"""
VLM Dataset Builder for OWA data.

This module provides a PyTorch Dataset interface for MLLM datasets created by
the OWA data pipeline. It handles lazy loading of images from MKV files and
integrates with nanoVLM training frameworks.
"""

from typing import Any, Dict, List, Optional

try:
    import torch
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # Fallback Dataset class for when torch is not available
    class Dataset:
        pass


import numpy as np
from PIL import Image

from owa.env.gst.mkv_reader import GstMKVReader


class VLMDatasetBuilder(Dataset):
    """
    PyTorch Dataset for OWA MLLM data with lazy image loading.

    This dataset:
    1. Loads MLLM dataset created by 03_binned_dataset_to_mllm_dataset.py
    2. Provides lazy loading of images from MKV files using image references
    3. Integrates with nanoVLM training frameworks
    4. Supports efficient caching and memory management

    Expected MLLM dataset format:
    {
        'instruction': str,
        'encoded_events': List[str],
        'image_refs': List[Dict],  # [{path, pts, timestamp_ns, bin_idx}, ...]
        'metadata': Dict
    }
    """

    def __init__(
        self, mllm_dataset, image_format: str = "pil", cache_images: bool = False, max_cache_size: int = 1000
    ):
        """
        Initialize the VLM dataset builder.

        Args:
            mllm_dataset: HuggingFace dataset with MLLM format
            image_format: Output format for images ("pil", "tensor", "numpy")
            cache_images: Whether to cache loaded images in memory
            max_cache_size: Maximum number of images to cache
        """
        self.dataset = mllm_dataset
        self.image_format = image_format
        self.cache_images = cache_images
        self.max_cache_size = max_cache_size
        self._image_cache = {} if cache_images else None
        self._cache_order = [] if cache_images else None

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample by index.

        Args:
            idx: Index of the sample

        Returns:
            Dict containing:
                - instruction: Task instruction
                - encoded_events: List of encoded action strings
                - images: List of loaded images (format depends on image_format)
                - metadata: Additional metadata
        """
        item = self.dataset[idx]

        # Load images lazily
        images = self._load_images(item["image_refs"])

        return {
            "instruction": item["instruction"],
            "encoded_events": item["encoded_events"],
            "images": images,
            "metadata": item["metadata"],
        }

    def _load_images(self, image_refs: List[Dict[str, Any]]) -> List[Any]:
        """
        Load images from MKV files using image references.

        Args:
            image_refs: List of image reference dicts with path, pts, etc.

        Returns:
            List of loaded images in the specified format
        """
        images = []

        for img_ref in image_refs:
            # Create cache key
            cache_key = f"{img_ref['path']}:{img_ref['pts']}"

            # Check cache first
            if self.cache_images and cache_key in self._image_cache:
                images.append(self._image_cache[cache_key])
                continue

            # Load image from MKV file
            try:
                image = self._load_single_image(img_ref)
                if image is not None:
                    # Add to cache if enabled
                    if self.cache_images:
                        self._add_to_cache(cache_key, image)
                    images.append(image)
            except Exception as e:
                print(f"Warning: Could not load image from {img_ref}: {e}")
                continue

        return images

    def _load_single_image(self, img_ref: Dict[str, Any]) -> Optional[Any]:
        """
        Load a single image from MKV file using image reference.

        Args:
            img_ref: Image reference dict with path, pts, etc.

        Returns:
            Loaded image in the specified format or None if failed
        """
        mkv_path = img_ref["path"]
        pts_ns = img_ref["pts"]

        # Convert pts from nanoseconds to seconds
        pts_seconds = pts_ns / 1e9

        try:
            # Use GstMKVReader to load the frame
            with GstMKVReader(mkv_path) as reader:
                # Seek to the specific timestamp
                reader.seek(pts_seconds, pts_seconds + 0.1)  # Small window

                # Get the frame
                for frame_data in reader:
                    frame_pts_ns = frame_data.get("pts", 0)
                    # Check if this is the frame we want (within tolerance)
                    if abs(frame_pts_ns - pts_ns) < 1e6:  # 1ms tolerance
                        frame_array = frame_data["frame"]

                        # Convert to the requested format
                        if self.image_format == "pil":
                            return self._array_to_pil(frame_array)
                        elif self.image_format == "tensor":
                            return self._array_to_tensor(frame_array)
                        elif self.image_format == "numpy":
                            return frame_array
                        else:
                            raise ValueError(f"Unsupported image format: {self.image_format}")

        except Exception as e:
            print(f"Error loading image from {mkv_path} at {pts_ns}: {e}")
            return None

        return None

    def _array_to_pil(self, frame_array) -> Image.Image:
        """Convert numpy array to PIL Image."""
        # Assuming BGRA format from GstMKVReader
        if len(frame_array.shape) == 3 and frame_array.shape[2] == 4:
            # Convert BGRA to RGB
            rgb_array = frame_array[:, :, [2, 1, 0]]  # BGR to RGB
            return Image.fromarray(rgb_array, mode="RGB")
        elif len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
            # Already RGB
            return Image.fromarray(frame_array, mode="RGB")
        else:
            raise ValueError(f"Unsupported frame array shape: {frame_array.shape}")

    def _array_to_tensor(self, frame_array):
        """Convert numpy array to PyTorch tensor."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available. Cannot convert to tensor format.")

        # Convert to PIL first, then to tensor for consistency
        pil_image = self._array_to_pil(frame_array)

        # Convert PIL to tensor (C, H, W) format manually
        # This avoids dependency on torchvision
        img_array = np.array(pil_image)
        # Convert from (H, W, C) to (C, H, W)
        img_array = img_array.transpose(2, 0, 1)
        # Convert to float and normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        # Convert to torch tensor
        return torch.from_numpy(img_array)

    def _add_to_cache(self, cache_key: str, image: Any):
        """Add image to cache with LRU eviction."""
        if not self.cache_images:
            return

        # Remove oldest if cache is full
        if len(self._image_cache) >= self.max_cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._image_cache[oldest_key]

        # Add new image
        self._image_cache[cache_key] = image
        self._cache_order.append(cache_key)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        if not self.cache_images:
            return {"cache_enabled": False}

        return {
            "cache_enabled": True,
            "cache_size": len(self._image_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hit_ratio": getattr(self, "_cache_hits", 0) / max(getattr(self, "_cache_requests", 1), 1),
        }
