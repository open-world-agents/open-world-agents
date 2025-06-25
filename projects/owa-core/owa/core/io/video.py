"""
Video I/O utilities for Open World Agents.

This module provides VideoReader and VideoWriter classes for reading and writing
video files using PyAV. Both local files and remote URLs (HTTP/HTTPS) are supported.

Examples:
    Reading from a local file:
        with VideoReader("video.mp4") as reader:
            for frame in reader.read_frames():
                # Process frame
                pass

    Reading from a remote URL:
        with VideoReader("https://example.com/video.mp4") as reader:
            frame = reader.read_frame(pts=1.5)  # Read frame at 1.5 seconds

    Writing a video:
        with VideoWriter("output.mp4", fps=30.0) as writer:
            writer.write_frame(frame_array)
"""

import atexit
import gc
import os
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Generator, Literal, Optional, Union

import av
import av.container
import numpy as np
from loguru import logger

# Type aliases
SECOND_TYPE = Union[float, Fraction]
DUPLICATE_TOLERANCE_SECOND: Fraction = Fraction(1, 120)
PTSUnit = Literal["pts", "sec"]
VideoPathType = Union[str, os.PathLike, Path]  # Supports both local paths and URLs

# Garbage collection counters for PyAV reference cycles
# Reference: https://github.com/pytorch/vision/blob/428a54c96e82226c0d2d8522e9cbfdca64283da0/torchvision/io/video.py#L53-L55
_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 10


def _normalize_video_path(video_path: VideoPathType) -> Union[str, Path]:
    """
    Normalize video path for use with PyAV.

    Args:
        video_path: Input video file path or URL

    Returns:
        str for URLs, Path for local files

    Raises:
        ValueError: If URL scheme is not supported or path is invalid
        FileNotFoundError: If local file does not exist
    """
    if isinstance(video_path, str):
        # Check if it's any kind of URL (not just http/https)
        if "://" in video_path:
            # Validate that we only support HTTP/HTTPS
            if not (video_path.startswith("http://") or video_path.startswith("https://")):
                raise ValueError(f"Unsupported URL scheme. Only http:// and https:// are supported, got: {video_path}")
            return video_path
        else:
            # It's a string path, convert to Path and validate
            local_path = Path(video_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Video file not found: {local_path}")
            return local_path
    else:
        # Convert to Path for local files and validate existence
        local_path = Path(video_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Video file not found: {local_path}")
        return local_path


class ContainerCache:
    """Thread-safe container cache with automatic cleanup."""

    def __init__(self, max_size=10, inactive_timeout=10**9):
        self._cache = {}  # path -> (container, ref_count, last_accessed)
        self._lock = threading.RLock()
        self.max_size = max_size
        self.inactive_timeout = inactive_timeout
        atexit.register(self.close_all)

    def get_container(self, video_path: Union[str, Path], mode: str = "r") -> av.container.Container:
        """Get or create cached container with reference counting."""
        path_str = str(video_path)

        with self._lock:
            self._cleanup_inactive()

            # Reuse existing read-mode container
            if path_str in self._cache and mode == "r":
                container, refs, _ = self._cache[path_str]
                self._cache[path_str] = (container, refs + 1, time.time())
                return container

            # Create new container
            container = av.open(path_str, mode)

            # Cache read-mode containers only
            if mode == "r":
                if len(self._cache) >= self.max_size:
                    self._evict_unused()
                self._cache[path_str] = (container, 1, time.time())

            return container

    def release_container(self, video_path: Union[str, Path]) -> None:
        """Decrease reference count."""
        path_str = str(video_path)

        with self._lock:
            if path_str in self._cache:
                container, refs, _ = self._cache[path_str]
                self._cache[path_str] = (container, max(0, refs - 1), time.time())

    def force_close_container(self, video_path: Union[str, Path]) -> None:
        """Immediately close and remove container."""
        path_str = str(video_path)

        with self._lock:
            if path_str in self._cache:
                container, _, _ = self._cache.pop(path_str)
                self._safe_close(container, path_str)

    def close_all(self) -> None:
        """Close all cached containers."""
        with self._lock:
            for path_str, (container, _, _) in list(self._cache.items()):
                self._safe_close(container, path_str)
            self._cache.clear()

    def _cleanup_inactive(self):
        """Remove unused containers that are too old."""
        current_time = time.time()
        to_remove = []

        for path, (container, refs, last_used) in self._cache.items():
            if refs == 0 and (current_time - last_used) > self.inactive_timeout:
                to_remove.append((path, container))

        for path, container in to_remove:
            self._cache.pop(path, None)
            self._safe_close(container, path)

    def _evict_unused(self):
        """Remove oldest unused container."""
        oldest_path = None
        oldest_time = float("inf")

        for path, (_, refs, last_used) in self._cache.items():
            if refs == 0 and last_used < oldest_time:
                oldest_path = path
                oldest_time = last_used

        if oldest_path:
            container, _, _ = self._cache.pop(oldest_path)
            self._safe_close(container, oldest_path)

    @staticmethod
    def _safe_close(container, path_str):
        """Safely close container with error handling."""
        try:
            container.close()
        except Exception as e:
            logger.warning(f"Error closing container {path_str}: {e}")


# Global cache instance
_container_cache = ContainerCache(max_size=10)


def get_video_container(video_path: Union[str, Path]) -> av.container.InputContainer:
    """Get a container from the cache or create a new one."""
    return _container_cache.get_container(video_path, mode="r")


def release_video_container(video_path: Union[str, Path]) -> None:
    """Release a container reference."""
    _container_cache.release_container(video_path)


def close_all_containers():
    """Close all cached containers."""
    _container_cache.close_all()


def force_close_video_container(video_path: Union[str, Path]) -> None:
    """Force immediate closure of a specific container."""
    _container_cache.force_close_container(video_path)


class VideoWriter:
    """VideoWriter uses PyAV to write video frames with VFR/CFR support.

    References:
        - https://stackoverflow.com/questions/65213302/how-to-write-variable-frame-rate-videos-in-python
        - https://github.com/PyAV-Org/PyAV/blob/main/examples/numpy/generate_video_with_pts.py
        - Design Reference: https://pytorch.org/vision/stable/generated/torchvision.io.read_video.html
    """

    def __init__(
        self, video_path: Union[str, os.PathLike, Path], fps: Optional[float] = None, vfr: bool = False, **kwargs
    ):
        """
        Args:
            video_path: Output video file path
            fps: Frames per second (required for CFR or when pts not provided)
            vfr: Use Variable Frame Rate
            **kwargs: Additional codec parameters
        """
        self.video_path = Path(video_path)
        self.fps = fps
        self.vfr = vfr
        self._closed = False
        self.past_pts = None

        # Setup codec parameters
        self.codec_params = {"gop_size": kwargs.get("gop_size", 30)}

        # Initialize container and stream
        self.container = av.open(str(video_path), mode="w")
        self._setup_stream()

    def _setup_stream(self):
        """Configure video stream for VFR or CFR."""
        if self.vfr:
            self.stream = self.container.add_stream("h264", rate=-1)
            self._time_base = Fraction(1, 60000)  # Fine-grained for VFR
        else:
            if not self.fps or self.fps <= 0:
                raise ValueError("fps must be positive for CFR (vfr=False)")
            self.stream = self.container.add_stream("h264", rate=int(self.fps))
            self._time_base = Fraction(1, int(self.fps))

        # Apply settings
        self.stream.pix_fmt = "yuv420p"
        self.stream.time_base = self._time_base
        self.stream.codec_context.time_base = self._time_base
        for key, value in self.codec_params.items():
            setattr(self.stream.codec_context, key, value)

    def write_frame(
        self,
        frame: Union[av.VideoFrame, np.ndarray],
        pts: Optional[Union[int, SECOND_TYPE]] = None,
        pts_unit: PTSUnit = "pts",
    ) -> Dict[str, Any]:
        """Write a frame to the video."""
        global _CALLED_TIMES
        _CALLED_TIMES += 1
        if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == 0:
            gc.collect()

        # Convert numpy to VideoFrame
        if isinstance(frame, np.ndarray):
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        elif not isinstance(frame, av.VideoFrame):
            raise TypeError("frame must be av.VideoFrame or np.ndarray")

        # Calculate PTS
        pts_as_pts = self._calculate_pts(pts, pts_unit)
        pts_as_sec = self.pts_to_sec(pts_as_pts)

        # Skip duplicate frames
        if self._is_duplicate(pts_as_pts, pts_as_sec):
            return {"source": str(self.video_path), "timestamp": float(pts_as_sec)}

        # Write frame
        frame.pts = pts_as_pts
        self.stream.width = frame.width
        self.stream.height = frame.height

        for packet in self.stream.encode(frame):
            self.container.mux(packet)

        self.past_pts = pts_as_pts
        return {"source": str(self.video_path), "timestamp": float(pts_as_sec)}

    def _calculate_pts(self, pts, pts_unit):
        """Calculate PTS value in pts units."""
        if pts is None:
            if not self.fps:
                raise ValueError("fps required when pts not provided")
            if self.past_pts is None:
                return 0
            return self.past_pts + self.sec_to_pts(Fraction(1, int(self.fps)))

        if pts_unit == "pts":
            if not isinstance(pts, int):
                raise TypeError("pts must be int when pts_unit is 'pts'")
            return pts
        elif pts_unit == "sec":
            if not isinstance(pts, (float, Fraction)):
                raise TypeError("pts must be float/Fraction when pts_unit is 'sec'")
            return self.sec_to_pts(pts)
        else:
            raise ValueError(f"Invalid pts_unit: {pts_unit}")

    def _is_duplicate(self, pts_as_pts, pts_as_sec):
        """Check if frame is duplicate within tolerance."""
        if self.past_pts is not None and pts_as_pts - self.past_pts < self.sec_to_pts(DUPLICATE_TOLERANCE_SECOND):
            logger.warning(f"Duplicate frame at {float(pts_as_sec):.2f}s in {self.video_path}. Skipping.")
            return True
        return False

    def pts_to_sec(self, pts: int) -> Fraction:
        return pts * self.stream.codec_context.time_base

    def sec_to_pts(self, sec: SECOND_TYPE) -> int:
        if not isinstance(sec, (float, Fraction)):
            raise TypeError("sec must be numeric")
        return int(sec / self.stream.codec_context.time_base)

    def close(self) -> None:
        """Finalize and close the container."""
        if self._closed:
            return

        try:
            # Flush encoder
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
        except Exception as e:
            logger.error(f"Error closing VideoWriter: {e}")
        finally:
            self._closed = True

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class VideoReader:
    """VideoReader uses PyAV to read video frames with caching support.

    Supports both local video files and remote URLs (HTTP/HTTPS).
    """

    def __init__(self, video_path: VideoPathType, force_close: bool = False):
        """
        Args:
            video_path: Input video file path or URL (HTTP/HTTPS)
            force_close: Force complete closure instead of using cache
        """
        self.video_path = _normalize_video_path(video_path)
        self.force_close = force_close
        self.container = get_video_container(self.video_path)

    def read_frames(
        self, start_pts: SECOND_TYPE = 0.0, end_pts: Optional[SECOND_TYPE] = None, fps: Optional[float] = None
    ) -> Generator[av.VideoFrame, None, None]:
        """Yield frames between start_pts and end_pts in seconds."""
        global _CALLED_TIMES
        _CALLED_TIMES += 1
        if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == 0:
            gc.collect()

        # Handle negative end_pts (Python-style indexing)
        if end_pts is not None and float(end_pts) < 0:
            if self.container.duration is None:
                raise ValueError("Video duration unavailable for negative end_pts")
            duration = self.container.duration / av.time_base
            end_pts = duration + float(end_pts)

        end_pts = float(end_pts) if end_pts is not None else float("inf")

        # Seek to start position
        timestamp_ts = int(av.time_base * float(start_pts))
        self.container.seek(timestamp_ts)

        if fps is None:
            # Yield all frames in interval
            yield from self._yield_all_frames(float(start_pts), end_pts)
        else:
            # Sample at specified fps
            if fps <= 0:
                raise ValueError("fps must be positive")
            yield from self._yield_sampled_frames(float(start_pts), end_pts, fps)

    def _yield_all_frames(self, start_pts, end_pts):
        """Yield all frames in time range."""
        for frame in self.container.decode(video=0):
            if frame.time is None:
                raise ValueError("Frame time is None")
            if frame.time < start_pts:
                continue
            if frame.time > end_pts:
                break
            yield frame

    def _yield_sampled_frames(self, start_pts, end_pts, fps):
        """Yield frames sampled at specified fps."""
        interval = 1.0 / fps
        next_time = start_pts
        video_stream = self.container.streams.video[0]

        for frame in self.container.decode(video=0):
            if frame.time is None:
                raise ValueError("Frame time is None")
            if frame.time < start_pts:
                continue
            if frame.time > end_pts:
                break

            if frame.time + 1e-8 >= next_time:
                frame.duration = interval / video_stream.time_base
                yield frame
                next_time += interval

                # Handle VFR gaps
                if frame.time > next_time:
                    missed = int((frame.time - next_time) // interval) + 1
                    next_time += missed * interval

    def read_frame(self, pts: SECOND_TYPE = 0.0) -> av.VideoFrame:
        """Read single frame at or after given timestamp."""
        for frame in self.read_frames(start_pts=pts, end_pts=None):
            return frame
        raise ValueError(f"Frame not found at {float(pts):.2f}s in {self.video_path}")

    def close(self) -> None:
        """Release container reference or force close."""
        if self.force_close:
            force_close_video_container(self.video_path)
        else:
            release_video_container(self.video_path)

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


if __name__ == "__main__":
    # Example usage
    video_path = Path("test.mp4")

    # Write a test video (VFR)
    with VideoWriter(video_path, fps=60.0, vfr=True) as writer:
        total_frames = 60
        for frame_i in range(total_frames):
            img = np.empty((48, 64, 3), dtype=np.uint8)
            img[:, :, 0] = (0.5 + 0.5 * np.sin(2 * np.pi * (0 / 3 + frame_i / total_frames))) * 255
            img[:, :, 1] = (0.5 + 0.5 * np.sin(2 * np.pi * (1 / 3 + frame_i / total_frames))) * 255
            img[:, :, 2] = (0.5 + 0.5 * np.sin(2 * np.pi * (2 / 3 + frame_i / total_frames))) * 255
            sec = Fraction(frame_i, 60)
            writer.write_frame(img, pts=sec, pts_unit="sec")

    # Write a test video (CFR)
    with VideoWriter(video_path.with_name("test_cfr.mp4"), fps=30.0, vfr=False) as writer_cfr:
        total_frames = 60
        for frame_i in range(total_frames):
            img = np.zeros((48, 64, 3), dtype=np.uint8)
            writer_cfr.write_frame(img)

    # Read back frames from local file
    with VideoReader(video_path) as reader:
        for frame in reader.read_frames(start_pts=Fraction(1, 2)):
            print(f"Local PTS: {frame.pts}, Time: {frame.time}, Shape: {frame.to_ndarray(format='rgb24').shape}")
            break  # Just show first frame
        try:
            frame = reader.read_frame(pts=Fraction(1, 2))
            print(f"Single frame at 0.5s: PTS={frame.pts}, Time={frame.time}")
        except ValueError as e:
            logger.error(e)

    # Example with remote URL (commented out to avoid network dependency in tests)
    # remote_url = "https://open-world-agents.github.io/open-world-agents/data/ocap.mkv"
    # try:
    #     with VideoReader(remote_url) as reader:
    #         frame = reader.read_frame(pts=0.0)
    #         print(f"Remote frame: PTS={frame.pts}, Time={frame.time}")
    # except Exception as e:
    #     logger.error(f"Failed to read remote video: {e}")
