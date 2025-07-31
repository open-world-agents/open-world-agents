import concurrent.futures
import os
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset as TorchDataset

from owa.msgs.desktop.screen import ScreenCaptured

from .config import DatasetStage
from .dataset import Dataset
from .transforms import resolve_episode_path

is_decoding_server_available = "VIDEO_DECODING_SERVER_URL" in os.environ


@dataclass
class FSLDatasetConfig:
    pad_token_id: int = 0
    max_sequence_length: int = 8192
    load_images: bool = True


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


class FSLDataset(TorchDataset):
    """Fixed Sequence Length dataset for training with tokenized data."""

    def __init__(
        self, dataset: Dataset, image_processor=None, config: FSLDatasetConfig = FSLDatasetConfig(), **kwargs
    ):
        # Validate inputs
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Expected Dataset from `owa.data.datasets`, got {type(dataset)}")
        if dataset.stage != DatasetStage.TOKENIZED:
            raise ValueError(f"Expected dataset stage to be TOKENIZED, got {dataset.stage}")
        if image_processor is not None and "Fast" not in image_processor.__class__.__name__:
            raise ValueError(
                "Image processor must be a fast image processor, "
                "make sure you pass `use_fast` directly to ImageProcessor.from_pretrained"
            )

        self.dataset = dataset
        self.image_processor = image_processor
        self.config = FSLDatasetConfig(**(config.__dict__ | kwargs))
        self.stat_logger = FSLStatLogger()

        # Cache cumulative sum for efficient token indexing
        # NOTE: Must cache whole cumsum in RAM for fast iteration
        self._cumsum = np.cumsum(self.dataset["total_token_count"])

    def __getitem__(self, idx: int) -> dict:
        start_token_index = idx * self.config.max_sequence_length
        start_event_index = int(np.searchsorted(self._cumsum, start_token_index, side="left"))

        # Collect data from events
        sequence_data = self._collect_sequence_data(start_event_index)

        # Process images if needed
        image_object, image_bits = self._process_images(sequence_data["images"])

        # Pad tokens to max sequence length
        tokens = sequence_data["tokens"]
        if len(tokens) < self.config.max_sequence_length:
            padding_length = self.config.max_sequence_length - len(tokens)
            tokens.extend([self.config.pad_token_id] * padding_length)

        # Create result
        result = {
            "texts": "".join(sequence_data["texts"]),
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if token_id != self.config.pad_token_id else 0 for token_id in tokens], dtype=torch.long
            ),
            "images": image_object,
        }

        self.stat_logger.update(1, len(tokens), len(sequence_data["images"]), image_bits)
        return result

    def _collect_sequence_data(self, start_event_index: int) -> dict:
        """Collect tokens, texts, and images from events up to max_sequence_length."""
        texts, all_token_ids, all_image_msgs = [], [], []
        tokens_so_far = 0

        for event_idx in range(start_event_index, len(self.dataset)):
            event = self.dataset[event_idx]

            # Check if adding this event would exceed max length
            if tokens_so_far + event["total_token_count"] > self.config.max_sequence_length:
                break

            # Collect event data
            texts.append(event["text"])
            all_token_ids.extend(event["token_ids"])
            tokens_so_far += event["total_token_count"]

            # Process images
            episode_path = resolve_episode_path(event["episode_path"], self.dataset.owa_config.mcap_root_directory)
            images = [
                ScreenCaptured.model_validate_json(image_json).resolve_relative_path(episode_path)
                for image_json in event["images"]
            ]
            all_image_msgs.extend(images)

        return {"texts": texts, "tokens": all_token_ids, "images": all_image_msgs}

    def _process_images(self, image_msgs: list[ScreenCaptured]) -> tuple:
        """Process images based on configuration and return (image_object, image_bits)."""
        if not self.config.load_images:
            return image_msgs, 0

        # Preload images in parallel if decoding server is available
        if is_decoding_server_available:
            self._preload_images_parallel(image_msgs)

        # Convert to PIL images
        all_images = [screen_captured.to_pil_image() for screen_captured in image_msgs]
        image_bits = sum(image.width * image.height * 3 for image in all_images)

        # Process with image processor if available
        if self.image_processor is not None:
            return self._process_with_image_processor(all_images), image_bits

        return all_images, image_bits

    def _preload_images_parallel(self, image_msgs: list[ScreenCaptured]):
        """Preload images in parallel using ThreadPoolExecutor."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(image.to_pil_image) for image in image_msgs]
            for idx, future in enumerate(futures):
                try:
                    future.result(timeout=30)
                except Exception as e:
                    # Use placeholder image on failure
                    image_msgs[idx].frame_arr = np.zeros((512, 512, 3), dtype=np.uint8)
                    warnings.warn(f"Failed to load image at index {idx}: {e}. Using placeholder image.", UserWarning)

    def _process_with_image_processor(self, images: list) -> torch.Tensor:
        """Process images with the configured image processor."""
        assert self.image_processor is not None  # Should be checked before calling this method

        pixel_values = []
        for image in images:
            processed = self.image_processor(image, return_tensors="pt")
            # (batch_size, max_num_images, 3, max_heights, max_widths) -> (3, height, width)
            pixel_value = processed["pixel_values"].squeeze(0).squeeze(0)
            assert (processed["pixel_attention_mask"] == 1).all()
            pixel_values.append(pixel_value)

        # NOTE: Assumes image_processor returns fixed size images
        return torch.stack(pixel_values) if pixel_values else torch.empty(0, 3, 224, 224)

    def take(self, n: int):
        """Yield first n samples from the dataset."""
        for i in range(n):
            yield self[i]

    def __len__(self) -> int:
        """Calculate the number of sequences based on total tokens and max_sequence_length."""
        total_tokens = self._cumsum[-1]
        return max(1, total_tokens // self.config.max_sequence_length)


def prepare_fsl(
    tokenized_dataset: Dataset,
    *,
    image_processor=None,
    config: FSLDatasetConfig = FSLDatasetConfig(),
    mcap_root_directory: Optional[str] = None,
    **kwargs,
) -> FSLDataset:
    """Prepare FSL dataset from tokenized dataset."""
    if mcap_root_directory is not None:
        tokenized_dataset.owa_config.mcap_root_directory = mcap_root_directory

    return FSLDataset(tokenized_dataset, image_processor, FSLDatasetConfig(**(config.__dict__ | kwargs)))
