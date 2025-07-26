import concurrent.futures
import os
import time
from dataclasses import dataclass

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
    # TODO: trim_last_event: bool = True


class FSLStatLogger:
    """Every n-th sample, log the stats."""

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


class FSLDataset(TorchDataset):
    def __init__(
        self, dataset: Dataset, image_processor=None, config: FSLDatasetConfig = FSLDatasetConfig(), **kwargs
    ):
        self.dataset = dataset
        self.image_processor = image_processor
        self.config = FSLDatasetConfig(**(config.__dict__ | kwargs))
        self.stat_logger = FSLStatLogger()

        if dataset.stage != DatasetStage.TOKENIZED:
            raise ValueError(f"Expected dataset stage to be TOKENIZED, got {dataset.stage}")

        if image_processor is not None and "Fast" not in image_processor.__class__.__name__:
            raise ValueError(
                "Image processor must be a fast image processor, make sure you pass `use_fast` directly to ImageProcessor.from_pretrained"
            )

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
        texts: list[str] = []
        all_token_ids: list[int] = []
        all_image_msgs: list[ScreenCaptured] = []
        tokens_so_far: int = 0

        for event_idx in range(start_event_index, len(self.dataset)):
            event = self.dataset[event_idx]
            texts.append(event["text"])
            episode_path = resolve_episode_path(event["episode_path"], self.dataset.owa_config.mcap_root_directory)
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
            all_image_msgs.extend(images)

        if self.config.load_images:
            # If we have a decoding server, use it to load all images in parallel
            if is_decoding_server_available:
                # TODO?: we may need to initialize ThreadPool once but it's initialization only takes 10.3 μs ± 275 ns per loop
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(image.to_pil_image) for image in all_image_msgs]
                    for idx, future in enumerate(futures):
                        try:
                            # screen_captured.frame_arr is cached here so that next time we call to_pil_image, it's fast
                            future.result(timeout=5)
                        except Exception as e:
                            all_image_msgs[idx].frame_arr = np.zeros((512, 512, 3), dtype=np.uint8)
                            logger.error(f"Failed to load image: {e}")

            # Now load the images
            all_images = [screen_captured.to_pil_image() for screen_captured in all_image_msgs]
            image_bits = sum(image.width * image.height * 3 for image in all_images)

            if self.image_processor is not None:
                pixel_values = []
                for image in all_images:
                    processed = self.image_processor(image, return_tensors="pt")
                    # (batch_size, max_num_images, 3, max_heights, max_widths) -> (3, height, width)
                    pixel_value = processed["pixel_values"].squeeze(0).squeeze(0)
                    assert (processed["pixel_attention_mask"] == 1).all()
                    pixel_values.append(pixel_value)
                # NOTE: this line assumes image_processor returns fixed size images.
                image_object_to_return = torch.stack(pixel_values) if pixel_values else torch.empty(0, 3, 224, 224)
            else:
                image_object_to_return = all_images
        else:
            image_bits = 0  # No images loaded
            image_object_to_return = all_image_msgs

        # Pad token_ids to max_sequence_length if needed
        if tokens_so_far < self.config.max_sequence_length:
            padding_length = self.config.max_sequence_length - tokens_so_far
            all_token_ids.extend([self.config.pad_token_id] * padding_length)
            tokens_so_far += padding_length

        assert len(all_token_ids) == self.config.max_sequence_length == tokens_so_far

        # Return dict with the processed data. TODO?: return `labels` also
        result = {
            "texts": "".join(texts),
            "input_ids": torch.tensor(all_token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if token_id != self.config.pad_token_id else 0 for token_id in all_token_ids], dtype=torch.long
            ),
            "images": image_object_to_return,
        }

        self.stat_logger.update(1, tokens_so_far, len(image_object_to_return), image_bits)

        return result

    def take(self, n):
        for i in range(n):
            yield self[i]

    def __len__(self):
        """Calculate the number of sequences based on total tokens and max_sequence_length."""
        self.check_prepared()

        total_tokens = self._cumsum[-1]
        return max(1, total_tokens // self.config.max_sequence_length)


def prepare_fsl(
    tokenized_dataset,
    max_sequence_length: int = 1024,
    pad_token_id: int = 0,
    load_images: bool = True,
    image_processor=None,
) -> FSLDataset:
    """Prepare FSL dataset from tokenized dataset."""
    config = FSLDatasetConfig(
        max_sequence_length=max_sequence_length, pad_token_id=pad_token_id, load_images=load_images
    )
    fsl_dataset = FSLDataset(tokenized_dataset, image_processor, config)
    fsl_dataset.prepare()
    return fsl_dataset
