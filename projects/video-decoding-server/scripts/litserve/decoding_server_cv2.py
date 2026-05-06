# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litserve",
#     "opencv-python>=4.11.0",
# ]
# ///

import base64
import gc
import threading
from pathlib import Path
from typing import Dict, Union

import cv2
import litserve as ls


class VideoDecoderCache:
    """Thread-safe cache for cv2.VideoCapture instances."""

    def __init__(self, max_size: int = 10):
        """Initialize the decoder cache.

        Args:
            max_size: Maximum number of decoders to cache
        """
        self._cache: Dict[str, cv2.VideoCapture] = {}
        self._lock = threading.RLock()
        self.max_size = max_size

    def get_decoder(self, video_path: Union[str, Path]) -> cv2.VideoCapture:
        """Get or create a cached decoder for the given video path.

        Args:
            video_path: Path to the video file

        Returns:
            cv2.VideoCapture instance for the video
        """
        path_str = str(video_path)

        with self._lock:
            # Return existing decoder if available
            if path_str in self._cache:
                return self._cache[path_str]

            # Create new decoder
            cap = cv2.VideoCapture(path_str)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {path_str}")

            # Evict oldest decoder if cache is full
            if len(self._cache) >= self.max_size:
                # Remove the first (oldest) entry
                oldest_path = next(iter(self._cache))
                self._cache[oldest_path].release()
                del self._cache[oldest_path]
                gc.collect()

            # Cache the new decoder
            self._cache[path_str] = cap
            return cap

    def clear(self):
        """Clear all cached decoders."""
        with self._lock:
            for cap in self._cache.values():
                cap.release()
            self._cache.clear()
            gc.collect()


class CV2LitAPI(ls.LitAPI):
    """LitServe API for video decoding using cv2."""

    def setup(self, device):
        """Initialize the API."""
        self.decoder_cache = VideoDecoderCache(max_size=10)

    def decode_request(self, request, **kwargs):
        """Decode incoming request to extract video path and timestamp.

        Args:
            request: JSON request with video_path and pts fields

        Returns:
            Tuple of (video_path, pts)
        """
        video_path = request["video_path"]
        pts = float(request["pts"])
        return (video_path, pts)

    def predict(self, x, **kwargs):
        """Decode video frame at specified timestamp."""
        video_path, pts = x

        cap = self.decoder_cache.get_decoder(video_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(pts * fps)

        # Seek to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame at {pts}s from {video_path}")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame_rgb

    def encode_response(self, output, **kwargs):
        """Encode the decoded frame as base64 BMP.

        Args:
            output: Numpy array containing the decoded frame in RGB format

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
    api = CV2LitAPI(
        max_batch_size=1,  # default: 1
        batch_timeout=0.01,  # default: 0.0
    )
    server = ls.LitServer(
        api,
        accelerator="cpu",  # default: auto
        workers_per_device=1,  # default: 1
    )
    server.run(port=8000, generate_client_file=False, num_api_servers=None)
