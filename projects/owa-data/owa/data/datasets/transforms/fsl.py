"""FSL Transform class for modular image processing."""

import concurrent.futures
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional

import line_profiler
import numpy as np
import torch
from loguru import logger
from PIL import Image

from owa.core.io.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder  # noqa: F401
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
    pad_token_id: int = 0


class FSLTransform:
    """Clean, modular FSL transform class."""

    def __init__(self, config: Optional[FSLTransformConfig] = None, image_processor: Any = None, **kwargs):
        """Initialize FSL transform with configuration."""
        if config is None:
            config = FSLTransformConfig()

        # Override config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.image_processor = image_processor
        self.is_decoding_server_available = "VIDEO_DECODING_SERVER_URL" in os.environ
        self.stat_logger = FSLStatLogger()

    def __call__(self, batch):
        """Transform batch for FSL stage."""
        return self.transform_batch(batch)

    @line_profiler.profile
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

            # Batch decode images
            self._batch_decode_images(image_msgs)

            # Load images with error handling
            all_images = []
            for img in image_msgs:
                try:
                    pil_image = img.to_pil_image(keep_av_open=True)
                    all_images.append(pil_image)
                except Exception as e:
                    if len(all_images) == 0:
                        warnings.warn(f"Failed to load first image: {e}. Using black placeholder.")
                        placeholder = Image.new("RGB", (448, 448), color="black")
                        all_images.append(placeholder)
                    else:
                        warnings.warn(f"Failed to load image: {e}. Using previous image.")
                        all_images.append(all_images[-1])

            # Calculate image bits
            image_bits = sum(image.width * image.height * 3 for image in all_images)
            total_image_bits += image_bits

            # Process with image processor if available
            if self.image_processor is not None:
                if all_images:
                    processed = self.image_processor(all_images, return_tensors="pt")
                    pixel_values = processed["pixel_values"]
                    # NOTE: SmolVLMImageProcessor returns [batch, max_num_images, 3, height, width]
                    # while InternVL's ImageProcessor returns [num_images, 3, height, width]
                    if pixel_values.dim() == 5:  # [1, num_images, 3, height, width]
                        pixel_values = pixel_values.squeeze(0)
                else:
                    # NOTE: InternVL3 expectes (448, 448) while SmolVLM2 expects (512, 512)
                    pixel_values = torch.empty(0, 3, 448, 448)
                results["images"].append(pixel_values)
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

    def _batch_decode_images(self, image_msgs: List[ScreenCaptured]) -> None:
        """Batch decode images."""
        if not image_msgs:
            return

        # Group images by video path for efficient batch processing
        video_groups = {}
        for idx, img_msg in enumerate(image_msgs):
            if img_msg.media_ref is None or not img_msg.media_ref.is_video or img_msg.media_ref.pts_ns is None:
                continue

            video_path = img_msg.media_ref.uri
            if video_path not in video_groups:
                video_groups[video_path] = {"indices": [], "timestamps": []}

            video_groups[video_path]["indices"].append(idx)
            # Convert nanoseconds to seconds
            timestamp_sec = img_msg.media_ref.pts_ns / 1_000_000_000
            video_groups[video_path]["timestamps"].append(timestamp_sec)

        # Process each video group with batch decoding
        for video_path, group in video_groups.items():
            try:
                # Create decoder for this video
                decoder = PyAVVideoDecoder(video_path)

                # Batch decode all frames for this video
                frame_batch = decoder.get_frames_played_at(seconds=group["timestamps"])

                # Store decoded frames in the corresponding ScreenCaptured objects
                for i, frame_tensor in enumerate(frame_batch.data):
                    img_idx = group["indices"][i]
                    if isinstance(frame_tensor, np.ndarray):
                        # Convert numpy array [C, H, W] to numpy array [H, W, C] in RGB format. ndarray does not have permute
                        frame_rgb = frame_tensor.transpose(1, 2, 0).astype(np.uint8)
                    else:
                        # Convert from tensor [C, H, W] to numpy array [H, W, C] in BGRA format
                        frame_rgb = frame_tensor.data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    # Convert RGB to BGRA (add alpha channel)
                    frame_bgra = np.concatenate(
                        [
                            frame_rgb[:, :, [2, 1, 0]],  # BGR
                            np.full((frame_rgb.shape[0], frame_rgb.shape[1], 1), 255, dtype=np.uint8),  # Alpha
                        ],
                        axis=2,
                    )

                    # Store in frame_arr
                    image_msgs[img_idx].frame_arr = frame_bgra

            except Exception as e:
                # If batch decoding fails, fall back to individual decoding
                logger.warning(f"Batch decoding failed for {video_path}: {e}. Falling back to individual decoding.")


def create_fsl_transform(
    image_processor=None, load_images: bool = True, mcap_root_directory: Optional[str] = None, **kwargs
):
    """Create FSL transform - maintains backward compatibility."""
    config = FSLTransformConfig(load_images=load_images, mcap_root_directory=mcap_root_directory, **kwargs)

    transform = FSLTransform(config, image_processor=image_processor)
    return transform.transform_batch
