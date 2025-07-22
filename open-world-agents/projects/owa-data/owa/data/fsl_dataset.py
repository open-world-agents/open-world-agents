import concurrent.futures
import os
import time
from dataclasses import dataclass

import numpy as np
from datasets import Dataset as HFDataset
from loguru import logger
from torch.utils.data import Dataset

from owa.msgs.desktop.screen import ScreenCaptured

is_decoding_server_available = "VIDEO_DECODING_SERVER_URL" in os.environ


@dataclass
class FSLDatasetConfig:
    pad_token_id: int = 0
    max_sequence_length: int = 8192
    load_images: bool = True
    # TODO: trim_last_event: bool = True


class FSLStatLogger:
    """Every n-th sample, log the stats, if master rank."""

    def __init__(self, log_every=10, decay_alpha=0.9):
        self.log_every = log_every
        self.decay_alpha = decay_alpha
        self.count = 0
        self.total_tokens = 0
        self.total_images = 0
        self.total_episodes = 0
        self.total_image_bits = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        # Recent metrics
        self.tokens_since_last_log = 0
        self.images_since_last_log = 0
        self.samples_since_last_log = 0
        self.image_bits_since_last_log = 0
        # Exponential moving averages - initialize to None
        self.ema_samples_per_sec = None
        self.ema_tokens_per_sec = None
        self.ema_images_per_sec = None
        self.ema_image_bitrate = None

    def _is_master_rank(self) -> bool:
        """Check if current process is master rank (rank 0) in distributed training."""
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
        except ImportError:
            # torch.distributed not available, assume single process
            pass
        return True

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

        if self.count % self.log_every == 0 and self._is_master_rank():
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

                # Update EMAs - initialize on first update
                if self.ema_samples_per_sec is None:
                    self.ema_samples_per_sec = samples_per_sec_recent
                    self.ema_tokens_per_sec = tokens_per_sec_recent
                    self.ema_images_per_sec = images_per_sec_recent
                    self.ema_image_bitrate = image_bitrate_recent
                else:
                    # Simple EMA formula: new_ema = alpha * old_ema + (1-alpha) * new_value
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
                if bits_per_sec >= 1e9:
                    return f"{bits_per_sec / 1e9:.1f}Gb/s"
                if bits_per_sec >= 1e6:
                    return f"{bits_per_sec / 1e6:.1f}Mb/s"
                if bits_per_sec >= 1e3:
                    return f"{bits_per_sec / 1e3:.1f}Kb/s"
                return f"{bits_per_sec:.0f}b/s"

            # Format log message
            ema_str = ""
            if self.ema_samples_per_sec is not None:
                ema_str = (
                    f" | EMA: {self.ema_samples_per_sec:.1f}s/s, "
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


class FSLDataset(Dataset):
    def __init__(self, dataset: HFDataset, config: FSLDatasetConfig = FSLDatasetConfig(), **kwargs):
        self.dataset = dataset
        self.config = FSLDatasetConfig(**(config.__dict__ | kwargs))
        self.stat_logger = FSLStatLogger()

    def prepare(self):
        # TODO?: apply parallel scan
        self._cumsum = np.cumsum(self.dataset["total_token_count"])

    def check_prepared(self):
        if not hasattr(self, "_cumsum"):
            raise RuntimeError("Dataset must be prepared before use. Call prepare() first.")

    def __getitem__(self, idx):
        self.check_prepared()

        start_token_index = idx * self.config.max_sequence_length

        # self.cumsum[start_event_index-1] < start_token_index <= self._cumsum[start_event_index]
        start_event_index = np.searchsorted(self._cumsum, start_token_index, side="left")

        # Collect token_ids and images from events
        all_token_ids: list[int] = []
        all_images: list[ScreenCaptured] = []
        tokens_so_far: int = 0

        for event_idx in range(start_event_index, len(self.dataset)):
            event = self.dataset[event_idx]
            episode_path = event["episode_path"]
            token_ids = event["token_ids"]
            images = event["images"]
            total_token_count = event["total_token_count"]

            # If this is the last event and adding all its tokens would exceed max_sequence_length
            if tokens_so_far + total_token_count > self.config.max_sequence_length:
                break

            all_token_ids.extend(token_ids)
            tokens_so_far += total_token_count

            # Deserialize ScreenCaptured from JSON
            images = [
                ScreenCaptured.model_validate_json(image_json).resolve_relative_path(episode_path)
                for image_json in images
            ]
            all_images.extend(images)

        if self.config.load_images:
            # If we have a decoding server, use it to load all images in parallel
            if is_decoding_server_available:
                # TODO?: we may need to initialize ThreadPool once but it's initialization only takes 10.3 μs ± 275 ns per loop
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(image.to_pil_image) for image in all_images]
                    for idx, future in enumerate(futures):
                        try:
                            # screen_captured.frame_arr is cached here so that next time we call to_pil_image, it's fast
                            future.result(timeout=5)
                        except Exception as e:
                            all_images[idx].frame_arr = np.zeros((512, 512, 3), dtype=np.uint8)
                            logger.error(f"Failed to load image: {e}")

            # Now load the images
            all_images = [screen_captured.to_pil_image() for screen_captured in all_images]  # type: ignore

        # Pad token_ids to max_sequence_length if needed
        if tokens_so_far < self.config.max_sequence_length:
            padding_length = self.config.max_sequence_length - tokens_so_far
            all_token_ids.extend([self.config.pad_token_id] * padding_length)
            tokens_so_far += padding_length

        # Return dict with the processed data
        result = {
            "token_ids": all_token_ids,
            "attention_mask": [1 if token_id != self.config.pad_token_id else 0 for token_id in all_token_ids],
            "total_token_count": tokens_so_far,
            "images": all_images,
        }
        image_bits = sum(image.width * image.height * 3 for image in all_images) if self.config.load_images else 0

        self.stat_logger.update(1, tokens_so_far, len(all_images), image_bits)

        return result

    def take(self, n):
        for i in range(n):
            yield self[i]

    def __len__(self):
        """Calculate the number of sequences based on total tokens and max_sequence_length."""
        self.check_prepared()

        total_tokens = self._cumsum[-1]
        return max(1, total_tokens // self.config.max_sequence_length)
