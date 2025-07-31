"""FSL Transform class for clean, modular image processing."""

import concurrent.futures
import os
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from owa.msgs.desktop.screen import ScreenCaptured
from .utils import resolve_episode_path


@dataclass
class FSLTransformConfig:
    """Configuration for FSL transform."""

    load_images: bool = True
    mcap_root_directory: Optional[str] = None
    image_processor: Any = None


class FSLTransform:
    """Clean, modular FSL transform class."""

    def __init__(self, config: Optional[FSLTransformConfig] = None, **kwargs):
        """Initialize FSL transform with configuration."""
        if config is None:
            config = FSLTransformConfig()

        # Override config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.is_decoding_server_available = "VIDEO_DECODING_SERVER_URL" in os.environ

    def __call__(self, batch):
        """Transform batch for FSL stage."""
        return self.transform_batch(batch)

    def transform_batch(self, batch):
        """Transform batch - handles image loading on-the-fly."""
        batch_size = len(batch["input_ids"])
        results = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "texts": batch["texts"],
            "images": [],
        }

        for i in range(batch_size):
            image_msgs_json = batch["images"][i]
            episode_path = resolve_episode_path(batch["episode_path"][i], self.config.mcap_root_directory)

            # Deserialize ScreenCaptured messages
            image_msgs = [
                ScreenCaptured.model_validate_json(img_json).resolve_relative_path(episode_path)
                for img_json in image_msgs_json
            ]

            if not self.config.load_images:
                results["images"].append(image_msgs)
                continue

            # Preload images in parallel if decoding server is available
            if self.is_decoding_server_available and image_msgs:
                self._preload_images_parallel(image_msgs)

            # Convert to PIL images
            all_images = [img.to_pil_image() for img in image_msgs]

            # Process with image processor if available
            if self.config.image_processor is not None and all_images:
                pixel_values = []
                for image in all_images:
                    processed = self.config.image_processor(image, return_tensors="pt")
                    pixel_value = processed["pixel_values"].squeeze(0).squeeze(0)
                    pixel_values.append(pixel_value)
                results["images"].append(torch.stack(pixel_values) if pixel_values else torch.empty(0, 3, 224, 224))
            else:
                results["images"].append(all_images)

        return results

    def _preload_images_parallel(self, image_msgs: List[ScreenCaptured]) -> None:
        """Preload images in parallel with error handling."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(img.to_pil_image) for img in image_msgs]
            for idx, future in enumerate(futures):
                try:
                    future.result(timeout=30)
                except Exception as e:
                    import numpy as np

                    image_msgs[idx].frame_arr = np.zeros((512, 512, 3), dtype=np.uint8)
                    warnings.warn(f"Failed to load image at index {idx}: {e}. Using placeholder.", UserWarning)


def create_fsl_transform(
    image_processor=None, load_images: bool = True, mcap_root_directory: Optional[str] = None, **kwargs
):
    """Create FSL transform - maintains backward compatibility."""
    config = FSLTransformConfig(
        image_processor=image_processor, load_images=load_images, mcap_root_directory=mcap_root_directory, **kwargs
    )

    transform = FSLTransform(config)
    return transform.transform_batch
