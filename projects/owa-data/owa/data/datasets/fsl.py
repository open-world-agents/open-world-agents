"""FSL (Few-Shot Learning) dataset implementation with on-the-fly sequence generation."""

import concurrent.futures
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset
from datasets.utils.typing import PathLike
from loguru import logger

from .base import OWADatasetBase
from .config import FSLDatasetConfig

# Check if video decoding server is available
is_decoding_server_available = "VIDEO_DECODING_SERVER_URL" in os.environ

try:
    import numpy as np
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class FSLStatLogger:
    """Statistics logger for FSL dataset performance monitoring."""

    def __init__(self, log_every=10, decay_alpha=0.9):
        self.log_every = log_every
        self.decay_alpha = decay_alpha
        self.count = 0
        self.total_tokens = 0
        self.total_images = 0
        self.total_image_bits = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        # Recent metrics
        self.tokens_since_last_log = 0
        self.images_since_last_log = 0
        self.samples_since_last_log = 0
        self.image_bits_since_last_log = 0
        # Exponential moving averages
        self.ema_samples_per_sec = None
        self.ema_tokens_per_sec = None
        self.ema_images_per_sec = None
        self.ema_image_bitrate = None

    def update(self, count, tokens, images, image_bits):
        self.count += count
        self.total_tokens += tokens
        self.total_images += images
        self.total_image_bits += image_bits

        # Track recent metrics
        self.samples_since_last_log += count
        self.tokens_since_last_log += tokens
        self.images_since_last_log += images
        self.image_bits_since_last_log += image_bits

        if self.count % self.log_every == 0:
            current_time = time.time()
            elapsed_total = current_time - self.start_time
            elapsed_since_last = current_time - self.last_log_time

            # Calculate total metrics
            samples_per_sec_total = self.count / (elapsed_total + 1e-6)
            tokens_per_sec_total = self.total_tokens / (elapsed_total + 1e-6)
            images_per_sec_total = self.total_images / (elapsed_total + 1e-6)
            image_bitrate_total = self.total_image_bits / (elapsed_total + 1e-6)

            # Calculate recent metrics
            if elapsed_since_last > 0:
                samples_per_sec_recent = self.samples_since_last_log / elapsed_since_last
                tokens_per_sec_recent = self.tokens_since_last_log / elapsed_since_last
                images_per_sec_recent = self.images_since_last_log / elapsed_since_last
                image_bitrate_recent = self.image_bits_since_last_log / elapsed_since_last

                # Update EMAs
                if self.ema_samples_per_sec is None:
                    self.ema_samples_per_sec = samples_per_sec_recent
                    self.ema_tokens_per_sec = tokens_per_sec_recent
                    self.ema_images_per_sec = images_per_sec_recent
                    self.ema_image_bitrate = image_bitrate_recent
                else:
                    self.ema_samples_per_sec = (
                        self.decay_alpha * self.ema_samples_per_sec + (1 - self.decay_alpha) * samples_per_sec_recent
                    )
                    self.ema_tokens_per_sec = (
                        self.decay_alpha * self.ema_tokens_per_sec + (1 - self.decay_alpha) * tokens_per_sec_recent
                    )
                    self.ema_images_per_sec = (
                        self.decay_alpha * self.ema_images_per_sec + (1 - self.decay_alpha) * images_per_sec_recent
                    )
                    self.ema_image_bitrate = (
                        self.decay_alpha * self.ema_image_bitrate + (1 - self.decay_alpha) * image_bitrate_recent
                    )

            def format_bitrate(bits_per_sec):
                if bits_per_sec < 1024:
                    return f"{bits_per_sec:.1f}b/s"
                elif bits_per_sec < 1024**2:
                    return f"{bits_per_sec / 1024:.1f}Kb/s"
                elif bits_per_sec < 1024**3:
                    return f"{bits_per_sec / (1024**2):.1f}Mb/s"
                else:
                    return f"{bits_per_sec / (1024**3):.1f}Gb/s"

            # Format EMA string
            ema_str = ""
            if self.ema_samples_per_sec is not None:
                ema_str = (
                    f" | EMA: "
                    f"{self.ema_samples_per_sec:.1f}s/s, "
                    f"{self.ema_tokens_per_sec:,.0f}t/s, "
                    f"{self.ema_images_per_sec:.1f}i/s, "
                    f"{format_bitrate(self.ema_image_bitrate)}"
                )

            logger.info(
                f"FSL[{self.count}] | Total: "
                f"{samples_per_sec_total:.1f}s/s, "
                f"{tokens_per_sec_total:,.0f}t/s, "
                f"{images_per_sec_total:.1f}i/s, "
                f"{format_bitrate(image_bitrate_total)}"
                f"{ema_str}"
            )

            # Reset recent counters
            self.tokens_since_last_log = 0
            self.images_since_last_log = 0
            self.samples_since_last_log = 0
            self.image_bits_since_last_log = 0
            self.last_log_time = current_time


class FSLDataset(OWADatasetBase):
    """
    FSL Dataset that provides on-the-fly sequence generation from tokenized events.

    This dataset takes a tokenized event dataset and generates training sequences
    by concatenating events up to max_sequence_length tokens. It handles image
    loading and processing on-demand for efficient memory usage.

    The dataset expects input data with columns:
    - 'episode_path': str - Path to the episode file
    - 'token_ids': List[int] - Tokenized event representation
    - 'images': List[str] - JSON-serialized ScreenCaptured objects
    - 'total_token_count': int - Number of tokens in this event
    - 'text': str - Human-readable text representation
    """

    def __init__(
        self, dataset: HFDataset, image_processor=None, owa_config: Optional[FSLDatasetConfig] = None, **kwargs
    ):
        if not HAS_TORCH:
            raise ImportError("FSLDataset requires torch and numpy. Please install them: pip install torch numpy")

        # Extract data from input dataset
        if hasattr(dataset, "data"):
            arrow_table = dataset.data
            info = dataset.info
            split = dataset.split
            indices_table = getattr(dataset, "_indices", None)
            fingerprint = getattr(dataset, "_fingerprint", None)
        else:
            arrow_table = dataset
            info = None
            split = None
            indices_table = None
            fingerprint = None

        # Merge config with kwargs for backward compatibility
        if owa_config is None:
            owa_config = FSLDatasetConfig("/tmp", dataset_type="fsl")

        # Override config with any provided kwargs
        config_dict = owa_config.to_dict()
        config_dict.update(kwargs)
        owa_config = FSLDatasetConfig.from_dict(config_dict)

        super().__init__(
            arrow_table=arrow_table,
            info=info,
            split=split,
            indices_table=indices_table,
            fingerprint=fingerprint,
            owa_config=owa_config,
        )

        self.image_processor = image_processor
        self._prepared = False
        self.stat_logger = FSLStatLogger()

        if image_processor is not None and "Fast" not in image_processor.__class__.__name__:
            raise ValueError("Image processor must be a fast image processor")

    def prepare(self):
        """Prepare dataset for sequence learning by computing cumulative token counts."""
        # Access data directly to avoid circular dependency with __getitem__
        total_token_counts = super().__getitem__(slice(None))["total_token_count"]
        self._cumsum = np.cumsum(total_token_counts)
        self._prepared = True

    def __getitem__(self, idx):
        """
        Get a training sequence by concatenating events up to max_sequence_length.

        Returns:
            Dict with keys:
            - 'texts': str - Concatenated text representation
            - 'input_ids': torch.Tensor - Token IDs padded to max_sequence_length
            - 'attention_mask': torch.Tensor - Attention mask (1 for real tokens, 0 for padding)
            - 'images': torch.Tensor or List - Processed images (depends on image_processor)
        """
        if not self._prepared:
            raise RuntimeError("Dataset must be prepared before use. Call prepare() first.")

        start_token_index = idx * self.owa_config.max_sequence_length
        start_event_index = np.searchsorted(self._cumsum, start_token_index, side="left")

        all_token_ids = []
        all_image_msgs = []
        texts = []
        tokens_so_far = 0

        for event_idx in range(start_event_index, len(self)):
            event = super().__getitem__(event_idx)  # Get from HF Dataset
            texts.append(event["text"])
            episode_path = event["episode_path"]
            token_ids = event["token_ids"]
            images = event["images"]
            total_token_count = event["total_token_count"]

            # Check if adding this event would exceed max_sequence_length
            if tokens_so_far + total_token_count > self.owa_config.max_sequence_length:
                break

            all_token_ids.extend(token_ids)
            tokens_so_far += total_token_count

            # Process images - deserialize ScreenCaptured from JSON
            if images:
                from owa.msgs.desktop.screen import ScreenCaptured

                # Resolve episode path using config if available
                if self.owa_config.mcap_root_directory and episode_path:
                    full_episode_path = Path(self.owa_config.mcap_root_directory) / episode_path
                else:
                    full_episode_path = episode_path

                screen_images = [
                    ScreenCaptured.model_validate_json(img_json).resolve_relative_path(str(full_episode_path))
                    for img_json in images
                ]
                all_image_msgs.extend(screen_images)

        # Load images if configured
        if self.owa_config.load_images:
            # Parallel image loading if decoding server available
            if is_decoding_server_available:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(image.to_pil_image) for image in all_image_msgs]
                    for idx, future in enumerate(futures):
                        try:
                            future.result(timeout=5)
                        except Exception as e:
                            all_image_msgs[idx].frame_arr = np.zeros((512, 512, 3), dtype=np.uint8)
                            logger.error(f"Failed to load image: {e}")

            # Convert to PIL images
            all_images = [screen_captured.to_pil_image() for screen_captured in all_image_msgs]
            image_bits = sum(image.width * image.height * 3 for image in all_images)

            # Process with image processor if provided
            if self.image_processor is not None:
                pixel_values = []
                for image in all_images:
                    processed = self.image_processor(image, return_tensors="pt")
                    pixel_value = processed["pixel_values"].squeeze(0).squeeze(0)
                    assert (processed["pixel_attention_mask"] == 1).all()
                    pixel_values.append(pixel_value)
                image_result = torch.stack(pixel_values) if pixel_values else torch.empty(0, 3, 224, 224)
            else:
                image_result = all_images
        else:
            image_bits = 0
            image_result = all_image_msgs

        # Pad token_ids to max_sequence_length
        if tokens_so_far < self.owa_config.max_sequence_length:
            padding_length = self.owa_config.max_sequence_length - tokens_so_far
            all_token_ids.extend([self.owa_config.pad_token_id] * padding_length)
            tokens_so_far += padding_length

        assert len(all_token_ids) == self.owa_config.max_sequence_length == tokens_so_far

        # Create result
        result = {
            "texts": "".join(texts),
            "input_ids": torch.tensor(all_token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if token_id != self.owa_config.pad_token_id else 0 for token_id in all_token_ids], dtype=torch.long
            ),
            "images": image_result,
        }

        # Update statistics
        self.stat_logger.update(1, tokens_so_far, len(image_result), image_bits)

        return result

    def take(self, n):
        """Take n samples from the dataset."""
        for i in range(n):
            yield self[i]

    def __len__(self):
        """Calculate the number of sequences based on total tokens and max_sequence_length."""
        if not self._prepared:
            raise RuntimeError("Dataset must be prepared before use. Call prepare() first.")

        total_tokens = self._cumsum[-1]
        return max(1, total_tokens // self.owa_config.max_sequence_length)

    @staticmethod
    def load_from_disk(dataset_path: PathLike, storage_options: Optional[dict] = None, **kwargs) -> "FSLDataset":
        """Load FSLDataset from disk with remote filesystem support."""
        from .discovery import resolve_dataset_path_and_config

        # Load HF dataset with remote support
        hf_kwargs = kwargs.copy()
        if storage_options:
            hf_kwargs["storage_options"] = storage_options

        hf_dataset = HFDataset.load_from_disk(dataset_path, **hf_kwargs)

        # Try to load OWA config with remote support
        resolved_path, config_data, fs = resolve_dataset_path_and_config(dataset_path, storage_options)
        owa_config = None
        if config_data:
            try:
                owa_config = FSLDatasetConfig.from_dict(config_data)
            except Exception:
                pass

        return FSLDataset(hf_dataset, owa_config=owa_config if isinstance(owa_config, FSLDatasetConfig) else None)
