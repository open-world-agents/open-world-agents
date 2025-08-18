"""FSL Transform class for clean, modular image processing."""

import concurrent.futures
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import torch
from loguru import logger

from owa.msgs.desktop.screen import ScreenCaptured

from .utils import resolve_episode_path


class FSLStatLogger:
    """Performance statistics logger with exponential moving averages."""

    def __init__(self, log_every: int = 10, decay_alpha: float = 0.9):
        self.log_every = log_every
        self.decay_alpha = decay_alpha
        self.count = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time

        # Cumulative totals
        self._totals = {"tokens": 0, "images": 0, "image_bits": 0}
        # Recent metrics (since last log)
        self._recent = {"tokens": 0, "images": 0, "samples": 0, "image_bits": 0}
        # Exponential moving averages
        self._emas = {"samples_per_sec": None, "tokens_per_sec": None, "images_per_sec": None, "image_bitrate": None}

    def update(self, count: int, tokens: int, images: int, image_bits: int):
        self.count += count

        # Update totals and recent metrics
        for key, value in zip(["tokens", "images", "image_bits"], [tokens, images, image_bits]):
            self._totals[key] += value
            self._recent[key] += value
        self._recent["samples"] += count

        if self.count % self.log_every == 0:
            self._log_stats()

    def _log_stats(self):
        current_time = time.time()
        elapsed_total = current_time - self.start_time
        elapsed_recent = current_time - self.last_log_time

        # Calculate rates
        total_rates = self._calculate_rates(self._totals, self.count, elapsed_total)
        recent_rates = self._calculate_rates(self._recent, self._recent["samples"], elapsed_recent)

        # Update EMAs
        self._update_emas(recent_rates)

        # Log message
        ema_str = self._format_ema_string() if self._emas["samples_per_sec"] is not None else ""
        logger.debug(f"FSL[{self.count}] | Total: {self._format_rates(total_rates)}{ema_str}")

        # Reset recent counters
        self._recent = {key: 0 for key in self._recent}
        self.last_log_time = current_time

    def _calculate_rates(self, metrics: dict, samples: int, elapsed: float) -> dict:
        safe_elapsed = elapsed + 1e-6
        return {
            "samples_per_sec": samples / safe_elapsed,
            "tokens_per_sec": metrics["tokens"] / safe_elapsed,
            "images_per_sec": metrics["images"] / safe_elapsed,
            "image_bitrate": metrics["image_bits"] / safe_elapsed,
        }

    def _update_emas(self, recent_rates: dict):
        for key, rate in recent_rates.items():
            if self._emas[key] is None:
                self._emas[key] = rate
            else:
                current_ema = self._emas[key]
                assert current_ema is not None  # Type hint for mypy
                self._emas[key] = self.decay_alpha * current_ema + (1 - self.decay_alpha) * rate

    def _format_rates(self, rates: dict) -> str:
        return (
            f"{rates['samples_per_sec']:.1f}s/s, {rates['tokens_per_sec']:,.0f}t/s, "
            f"{rates['images_per_sec']:.1f}i/s, {self._format_bitrate(rates['image_bitrate'])}"
        )

    def _format_ema_string(self) -> str:
        # All EMAs should be non-None when this is called
        assert all(ema is not None for ema in self._emas.values())
        image_bitrate = self._emas["image_bitrate"]
        assert image_bitrate is not None  # Type hint for mypy
        return (
            f" | EMA: {self._emas['samples_per_sec']:.1f}s/s, "
            f"{self._emas['tokens_per_sec']:,.0f}t/s, {self._emas['images_per_sec']:.1f}i/s, "
            f"{self._format_bitrate(image_bitrate)}"
        )

    @staticmethod
    def _format_bitrate(bits_per_sec: float) -> str:
        for unit, threshold in [("Gb/s", 1e9), ("Mb/s", 1e6), ("Kb/s", 1e3)]:
            if bits_per_sec >= threshold:
                return f"{bits_per_sec / threshold:.1f}{unit}"
        return f"{bits_per_sec:.0f}b/s"


@dataclass
class FSLTransformConfig:
    """Configuration for FSL transform."""

    load_images: bool = True
    mcap_root_directory: Optional[str] = None
    image_processor: Any = None
    pad_token_id: int = 0


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
        self.stat_logger = FSLStatLogger()

    def __call__(self, batch):
        """Transform batch for FSL stage."""
        return self.transform_batch(batch)

    def transform_batch(self, batch):
        """Transform batch - handles image loading on-the-fly."""
        batch_size = len(batch["input_ids"])
        # NOTE: these are native lists, need to be converted to tensors
        results = {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "texts": batch["texts"],
            "images": [],
        }

        # Track metrics for logging
        total_tokens = 0
        total_images = 0
        total_image_bits = 0

        for i in range(batch_size):
            image_msgs_json = batch["images"][i]
            episode_path = resolve_episode_path(batch["episode_path"][i], self.config.mcap_root_directory)

            # Count tokens for this sample (exclude padding tokens)
            sample_tokens = len([token for token in batch["input_ids"][i] if token != self.config.pad_token_id])
            total_tokens += sample_tokens

            # Deserialize ScreenCaptured messages
            image_msgs = [
                ScreenCaptured.model_validate_json(img_json).resolve_relative_path(episode_path)
                for img_json in image_msgs_json
            ]
            total_images += len(image_msgs)

            if not self.config.load_images:
                results["images"].append(image_msgs)
                continue

            # Preload images in parallel if decoding server is available
            if self.is_decoding_server_available and image_msgs:
                self._preload_images_parallel(image_msgs)

            # Convert to PIL images
            all_images = [img.to_pil_image(keep_av_open=True) for img in image_msgs]

            # Calculate image bits
            image_bits = sum(image.width * image.height * 3 for image in all_images)
            total_image_bits += image_bits

            # Process with image processor if available
            if self.config.image_processor is not None:
                pixel_values = []
                for image in all_images:
                    processed = self.config.image_processor(image, return_tensors="pt")
                    pixel_value = processed["pixel_values"].squeeze(0).squeeze(0)
                    pixel_values.append(pixel_value)
                # NOTE: SmolVLM2-256M-Video-Instruct expects [num_images, 3, 512, 512]
                results["images"].append(torch.stack(pixel_values) if pixel_values else torch.empty(0, 3, 512, 512))
            else:
                results["images"].append(all_images)

        # Update statistics
        self.stat_logger.update(batch_size, total_tokens, total_images, total_image_bits)

        return results

    def _preload_images_parallel(self, image_msgs: List[ScreenCaptured]) -> None:
        """Preload images in parallel with error handling."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(img.to_pil_image) for img in image_msgs]
            for idx, future in enumerate(futures):
                try:
                    future.result(timeout=30)
                except Exception as e:
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
