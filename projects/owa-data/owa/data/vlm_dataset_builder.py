"""
VLM Dataset Builder for OWA data.

This module provides a PyTorch Dataset interface for MLLM datasets created by
the OWA data pipeline. It handles lazy loading of images from MKV files and
integrates with nanoVLM training frameworks.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from owa.msgs.desktop.screen import ScreenCaptured


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

    def _load_images(self, image_refs) -> List[Any]:
        """
        Load images from MKV files using image references.

        Args:
            image_refs: Either list of dicts or dict with lists (HuggingFace Sequence format)

        Returns:
            List of loaded images in the specified format
        """
        images = []

        # Handle HuggingFace Sequence format (dict with lists)
        if isinstance(image_refs, dict) and "path" in image_refs:
            # Convert from HF Sequence format to list of dicts
            num_refs = len(image_refs["path"])
            img_ref_list = []
            for i in range(num_refs):
                img_ref = {
                    "path": image_refs["path"][i],
                    "pts": image_refs["pts"][i],
                    "utc_ns": image_refs["utc_ns"][i],
                    "timestamp_ns": image_refs["timestamp_ns"][i],
                    "bin_idx": image_refs["bin_idx"][i],
                }
                img_ref_list.append(img_ref)
        else:
            # Already in list format
            img_ref_list = image_refs

        for img_ref in img_ref_list:
            # Handle both ScreenCaptured objects and dictionaries
            if hasattr(img_ref, "path") and hasattr(img_ref, "pts"):
                # It's a ScreenCaptured object
                cache_key = f"{img_ref.path}:{img_ref.pts}"
                screen_captured = img_ref
            else:
                # It's a dictionary
                cache_key = f"{img_ref['path']}:{img_ref['pts']}"
                screen_captured = None

            # Check cache first
            if self.cache_images and cache_key in self._image_cache:
                images.append(self._image_cache[cache_key])
                continue

            # Load image using ScreenCaptured
            try:
                if screen_captured:
                    # Use the ScreenCaptured object directly
                    image = self._load_from_screen_captured(screen_captured)
                else:
                    # Create ScreenCaptured from dictionary
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

    def _load_from_screen_captured(self, screen_captured: ScreenCaptured) -> Optional[Any]:
        """
        Load image directly from a ScreenCaptured object.

        Args:
            screen_captured: ScreenCaptured object

        Returns:
            Loaded image in the specified format or None if failed
        """
        try:
            # Load the frame using ScreenCaptured's methods
            if self.image_format == "pil":
                return screen_captured.to_pil_image()
            elif self.image_format == "numpy":
                return screen_captured.to_rgb_array()
            elif self.image_format == "tensor":
                # Convert PIL to tensor
                pil_image = screen_captured.to_pil_image()
                return self._pil_to_tensor(pil_image)
            else:
                raise ValueError(f"Unsupported image format: {self.image_format}")

        except Exception as e:
            print(f"Error loading image from ScreenCaptured {screen_captured.path} at {screen_captured.pts}: {e}")
            return None

    def _load_single_image(self, img_ref: Dict[str, Any]) -> Optional[Any]:
        """
        Load a single image using ScreenCaptured.lazy_load().

        Args:
            img_ref: Image reference dict with path, pts, etc.

        Returns:
            Loaded image in the specified format or None if failed
        """
        try:
            # Create ScreenCaptured instance from image reference
            screen_emitted = ScreenCaptured(path=img_ref["path"], pts=img_ref["pts"], utc_ns=img_ref.get("utc_ns"))

            # Load the frame using ScreenCaptured's lazy_load method
            if self.image_format == "pil":
                return screen_emitted.to_pil_image()
            elif self.image_format == "numpy":
                return screen_emitted.to_rgb_array()
            elif self.image_format == "tensor":
                # Convert PIL to tensor
                pil_image = screen_emitted.to_pil_image()
                return self._pil_to_tensor(pil_image)
            else:
                raise ValueError(f"Unsupported image format: {self.image_format}")

        except Exception as e:
            print(f"Error loading image from {img_ref['path']} at {img_ref['pts']}: {e}")
            return None

    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to PyTorch tensor."""
        # Convert PIL to tensor (C, H, W) format
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
