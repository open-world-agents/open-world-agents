# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litserve",
#     "opencv-python>=4.11.0",
#     "torch==2.7.1",
#     "torchvision==0.22.1",
#     "torchcodec==0.4.0",
# ]
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cpu", marker = "sys_platform != 'linux'" },
#   { index = "pytorch-cu126", marker = "sys_platform == 'linux'" },
# ]
# torchvision = [
#   { index = "pytorch-cpu", marker = "sys_platform != 'linux'" },
#   { index = "pytorch-cu126", marker = "sys_platform == 'linux'" },
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# ///

import base64
import gc
import threading
from pathlib import Path
from typing import Dict, Union

import cv2
import litserve as ls
from torchcodec.decoders import VideoDecoder

# os.system("ffmpeg -decoders | grep -i nvidia")
# os.system("pip list")


class VideoDecoderCache:
    """Thread-safe cache for TorchCodec VideoDecoder instances."""

    def __init__(self, max_size: int = 10, device="cpu"):
        """Initialize the decoder cache.

        Args:
            max_size: Maximum number of decoders to cache
        """
        self._cache: Dict[str, VideoDecoder] = {}
        self._lock = threading.RLock()
        self.max_size = max_size
        self.device = device

    def get_decoder(self, video_path: Union[str, Path]) -> VideoDecoder:
        """Get or create a cached decoder for the given video path.

        Args:
            video_path: Path to the video file

        Returns:
            VideoDecoder instance for the video
        """
        path_str = str(video_path)

        with self._lock:
            # Return existing decoder if available
            if path_str in self._cache:
                return self._cache[path_str]

            # Create new decoder
            decoder = VideoDecoder(path_str, device=self.device, num_ffmpeg_threads=1)

            # Evict oldest decoder if cache is full
            if len(self._cache) >= self.max_size:
                # Remove the first (oldest) entry
                oldest_path = next(iter(self._cache))
                del self._cache[oldest_path]
                gc.collect()  # Help with memory cleanup

            # Cache the new decoder
            self._cache[path_str] = decoder
            return decoder

    def clear(self):
        """Clear all cached decoders."""
        with self._lock:
            self._cache.clear()
            gc.collect()


class TorchCodecLitAPI(ls.LitAPI):
    """LitServe API for video decoding using TorchCodec."""

    def setup(self, device):
        """Initialize the API with device configuration.

        Args:
            device: Device to use for decoding (cpu/cuda)
        """
        self.device = device
        self.decoder_cache = VideoDecoderCache(max_size=10, device=device)

    def decode_request(self, request, **kwargs):
        """Decode incoming request to extract video path and timestamp.

        Args:
            request: JSON request with video_path and pts fields
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (video_path, pts)
        """
        video_path = request["video_path"]
        pts = float(request["pts"])
        return (video_path, pts)

    # def batch(self, inputs): ...

    def predict(self, x, **kwargs):
        """Decode video frame(s) with batch processing: group, batch, ungroup."""
        is_batch = isinstance(x, list)
        if not is_batch:
            x = [x]

        # 1. Group by video file
        groups = {}
        for i, (video_path, pts) in enumerate(x):
            path_str = str(video_path)
            if path_str not in groups:
                groups[path_str] = {"indices": [], "timestamps": []}
            groups[path_str]["indices"].append(i)
            groups[path_str]["timestamps"].append(pts)

        print(f"batch size: {len(x)}, unique videos: {len(groups)}, device: {self.device}")

        # 2. Batch process each video group
        results = [None] * len(x)
        for path_str, group in groups.items():
            decoder = self.decoder_cache.get_decoder(path_str)
            timestamps = group["timestamps"]

            frames_data = decoder.get_frames_played_at(seconds=timestamps)
            frames = frames_data.data

            # Convert tensors to numpy arrays
            for i, frame_tensor in enumerate(frames):
                frame_array = frame_tensor.permute(1, 2, 0).cpu().numpy()
                results[group["indices"][i]] = frame_array

        # 3. Return ungrouped results
        return results if is_batch else results[0]

    # def unbatch(self, output): ...

    def encode_response(self, output, **kwargs):
        """Encode the decoded frame as base64 BMP.

        Args:
            output: Numpy array containing the decoded frame in RGB format
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with base64-encoded BMP frame data
        """
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Encode as BMP
        success, frame_bytes = cv2.imencode(".bmp", frame_bgr)
        if not success:
            raise RuntimeError("Failed to encode frame as BMP")

        # Return base64-encoded frame
        return {"frame": base64.b64encode(frame_bytes.tobytes()).decode("utf-8")}


if __name__ == "__main__":
    api = TorchCodecLitAPI(
        max_batch_size=16,  # default: 1
        batch_timeout=0.25,  # default: 0.0
    )
    server = ls.LitServer(
        api,
        accelerator="cuda",  # default: auto
        workers_per_device=1,  # default: 1
    )
    server.run(port=8000, generate_client_file=False, num_api_servers=None)
