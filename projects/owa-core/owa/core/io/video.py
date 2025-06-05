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

# Garbage collection counters for PyAV reference cycles
# Reference: https://github.com/pytorch/vision/blob/428a54c96e82226c0d2d8522e9cbfdca64283da0/torchvision/io/video.py#L53-L55
_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 10


# Thread-safe container cache with reference counting
class ContainerCache:
    _instance = None

    @classmethod
    def get_instance(cls, max_size=10, inactive_timeout=10**9):
        if cls._instance is None:
            cls._instance = ContainerCache(max_size, inactive_timeout)
        return cls._instance

    def __init__(self, max_size=10, inactive_timeout=10**9):
        self._containers = {}  # Maps path to (container, ref_count, last_accessed)
        self._lock = threading.RLock()
        self.max_size = max_size
        self.inactive_timeout = inactive_timeout

        # Register cleanup function
        atexit.register(self.close_all)

    def get_container(self, video_path: Path, mode: str = "r") -> av.container.Container:
        """Get or create and cache a PyAV container with reference counting."""
        path_str = str(video_path)
        current_time = time.time()

        with self._lock:
            # Clean up inactive containers first
            self._cleanup_inactive(current_time)

            # If the container exists and is in read mode (reusable)
            if path_str in self._containers and mode == "r":
                container, ref_count, _ = self._containers[path_str]
                self._containers[path_str] = (container, ref_count + 1, current_time)
                logger.debug(f"Reusing cached container for {path_str} (refs: {ref_count + 1})")
                return container

            # Create a new container
            container = av.open(path_str, mode)

            # Only cache read-mode containers
            if mode == "r":
                # Check if we need to evict
                if len(self._containers) >= self.max_size:
                    self._evict_lru()

                self._containers[path_str] = (container, 1, current_time)
                logger.debug(f"Created new cached container for {path_str}")

            return container

    def release_container(self, video_path: Path) -> None:
        """Decrease reference count for a container and close if no more references."""
        path_str = str(video_path)
        current_time = time.time()

        with self._lock:
            if path_str not in self._containers:
                logger.warning(f"Attempted to release non-cached container: {path_str}")
                return

            container, ref_count, _ = self._containers[path_str]

            if ref_count <= 1:
                # Last reference, but don't close immediately - just set ref_count to 0
                # and update timestamp for potential reuse
                self._containers[path_str] = (container, 0, current_time)
                logger.debug(f"Container for {path_str} has no more references (but kept in cache)")
            else:
                # Decrement reference count and update timestamp
                self._containers[path_str] = (container, ref_count - 1, current_time)
                logger.debug(f"Released container for {path_str} (refs: {ref_count - 1})")

    def _cleanup_inactive(self, current_time):
        """Clean up containers that haven't been accessed for a while and have ref_count = 0."""
        for path in list(self._containers.keys()):
            container, ref_count, last_accessed = self._containers[path]
            # If container is not in use and hasn't been accessed for a while
            if ref_count == 0 and (current_time - last_accessed) > self.inactive_timeout:
                try:
                    logger.debug(
                        f"Closing inactive container: {path} (inactive for {current_time - last_accessed:.1f}s)"
                    )
                    container.close()
                except Exception as e:
                    logger.warning(f"Error closing inactive container {path}: {e}")
                finally:
                    del self._containers[path]

    def _evict_lru(self):
        """Evict the least recently used container with ref_count == 0."""
        lru_path = None
        lru_time = float("inf")

        # Find the least recently used container with ref_count == 0
        for path, (_, ref_count, last_accessed) in self._containers.items():
            if ref_count == 0 and last_accessed < lru_time:
                lru_path = path
                lru_time = last_accessed

        # If we found one, close and remove it
        if lru_path:
            container, _, _ = self._containers[lru_path]
            try:
                logger.debug(f"Evicting least recently used container: {lru_path}")
                container.close()
            except Exception as e:
                logger.warning(f"Error closing container during eviction {lru_path}: {e}")
            finally:
                del self._containers[lru_path]
                return True

        logger.warning("Cache full and all containers are in use. Cannot evict any container.")
        return False

    def close_all(self) -> None:
        """Close all cached containers."""
        with self._lock:
            for path_str, (container, _, _) in list(self._containers.items()):
                try:
                    logger.debug(f"Closing container for {path_str} during cleanup")
                    container.close()
                except Exception as e:
                    logger.warning(f"Error closing container {path_str}: {e}")

            self._containers.clear()

    def force_close_container(self, video_path: Path) -> None:
        """Force immediate closure of a specific container, removing it from cache."""
        path_str = str(video_path)

        with self._lock:
            if path_str in self._containers:
                container, ref_count, _ = self._containers[path_str]
                try:
                    logger.debug(f"Force closing container for {path_str} (refs: {ref_count})")
                    container.close()
                except Exception as e:
                    logger.warning(f"Error force closing container {path_str}: {e}")
                finally:
                    del self._containers[path_str]
            else:
                logger.debug(f"Container {path_str} not in cache, nothing to force close")


# Create the singleton container cache
_container_cache = ContainerCache.get_instance(max_size=10)


def get_video_container(video_path: Path) -> av.container.InputContainer:
    """Get a container from the cache or create a new one."""
    return _container_cache.get_container(video_path, mode="r")


def release_video_container(video_path: Path) -> None:
    """Release a container reference, potentially making it eligible for eviction."""
    _container_cache.release_container(video_path)


def close_all_containers():
    """Close all cached containers."""
    _container_cache.close_all()


def force_close_video_container(video_path: Path) -> None:
    """Force immediate closure of a specific container."""
    _container_cache.force_close_container(video_path)


# Define PTSUnit as a Literal type for clarity
PTSUnit = Literal["pts", "sec"]


class VideoWriter:
    """
    VideoWriter uses PyAV to write video frames with optional support for VFR (Variable Frame Rate).
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
            video_path (Path): The path to the output video file.
            fps (float, optional): Nominal frames per second. Required if vfr is False or if pts is not provided per frame.
            vfr (bool): Whether to use Variable Frame Rate. If False, configures a constant frame rate.
            **kwargs: Additional codec parameters
        """
        self.video_path = Path(video_path)
        self.fps = fps
        self.vfr = vfr
        self._closed = False
        self.past_pts = None

        # Process optional codec parameters with defaults
        self.codec_params = {
            # "bit_rate": kwargs.get("bit_rate", 20 * (2**20)),  # 20 Mbps
            "gop_size": kwargs.get("gop_size", 30),
        }

        # Open container and setup stream
        self.container = av.open(str(video_path), mode="w")
        self._setup_stream()

    def _setup_stream(self):
        """Configure video stream based on VFR or CFR settings."""
        if self.vfr:
            # VFR: use fine-grained time_base for variable timestamps
            if self.fps is not None:
                logger.warning("fps is provided but vfr is True. Using fps for time_base but allowing variable PTS.")
            self.stream = self.container.add_stream("h264", rate=-1)
            self.stream.pix_fmt = "yuv420p"
            self._time_base = Fraction(1, 60000)  # Fine-grained timestamps for VFR
        else:
            # CFR: require fps and configure fixed time_base
            if self.fps is None:
                raise ValueError("fps must be provided for constant frame rate (vfr=False)")
            if self.fps <= 0:
                raise ValueError("fps must be a positive number")
            self.stream = self.container.add_stream("h264", rate=int(self.fps))
            self.stream.pix_fmt = "yuv420p"
            self._time_base = Fraction(1, int(self.fps))

        # Apply common stream settings
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
        """
        Write a frame to the video. If pts is None, it will be set to the next frame (using fps).

        Args:
            frame (av.VideoFrame | np.ndarray): The frame to write.
            pts (int | float | Fraction | None): The PTS value of the frame. If None, computes next PTS (requires fps).
            pts_unit ("pts" | "sec"): Unit of the provided pts.

        Returns:
            dict: A simple dict containing 'source' (file path) and 'timestamp' (in seconds).
        """
        global _CALLED_TIMES
        _CALLED_TIMES += 1
        if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == 0:
            gc.collect()

        if not isinstance(frame, (av.VideoFrame, np.ndarray)):
            raise TypeError("frame must be av.VideoFrame or np.ndarray")

        # Convert numpy array to VideoFrame if needed
        if isinstance(frame, np.ndarray):
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

        # Determine pts_as_pts (integer) based on provided pts or next sequential
        if pts is None:
            if self.fps is None:
                raise ValueError("fps must be provided if pts is not provided.")
            if self.past_pts is None:
                pts_as_pts = 0
            else:
                # For CFR and VFR, use time_base conversion of one frame interval
                pts_increment = self.sec_to_pts(Fraction(1, int(self.fps)))
                pts_as_pts = self.past_pts + pts_increment
        else:
            if pts_unit == "pts":
                if not isinstance(pts, int):
                    raise TypeError("pts must be int if pts_unit is 'pts'")
                pts_as_pts = pts
            elif pts_unit == "sec":
                if not isinstance(pts, (float, Fraction)):
                    raise TypeError("pts must be float or Fraction if pts_unit is 'sec'")
                pts_as_pts = self.sec_to_pts(pts)
            else:
                raise ValueError(f"Invalid pts_unit: {pts_unit}")

        # Convert to seconds for timestamp
        pts_as_sec = self.pts_to_sec(pts_as_pts)
        logger.debug(f"Writing frame with PTS={pts_as_pts}, Time={float(pts_as_sec):.3f}s")

        # Filter duplicate frames within tolerance
        if self.past_pts is not None and pts_as_pts - self.past_pts < self.sec_to_pts(DUPLICATE_TOLERANCE_SECOND):
            logger.warning(
                f"Duplicate frame detected at {float(pts_as_sec):.2f}s "
                f"(previous: {float(self.pts_to_sec(self.past_pts)):.2f}s) in {self.video_path}. Skipping."
            )
            return {"source": str(self.video_path), "timestamp": float(pts_as_sec)}

        # Assign pts and encode
        frame.pts = pts_as_pts
        # Set stream dimensions on first frame
        self.stream.width = frame.width
        self.stream.height = frame.height
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

        self.past_pts = pts_as_pts
        return {"source": str(self.video_path), "timestamp": float(pts_as_sec)}

    def pts_to_sec(self, pts: int) -> Fraction:
        return pts * self.stream.codec_context.time_base

    def sec_to_pts(self, sec: SECOND_TYPE) -> int:
        if not isinstance(sec, (float, Fraction)):
            raise TypeError("sec must be a numeric type (float or Fraction)")
        return int(sec / self.stream.codec_context.time_base)

    def close(self) -> None:
        """Finalize writing and close the container."""
        if self._closed:
            return

        # Flush encoder
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        except Exception as e:
            logger.error(f"Error encoding final packets: {e}")
        finally:
            try:
                self.container.close()
                self._closed = True
            except Exception as e:
                logger.error(f"Error closing container: {e}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class VideoReader:
    """
    VideoReader uses PyAV to read video frames based on start and end timestamps (in seconds).
    """

    def __init__(self, video_path: Union[str, os.PathLike, Path], force_close: bool = False):
        """
        Args:
            video_path (Union[str, os.PathLike, Path]): The path to the input video file.
            force_close (bool): If True, forces complete container closure on close() instead of using cache.
        """
        self.video_path = Path(video_path)
        self.force_close = force_close
        # Always use cached container, but handle force_close in close() method
        self.container = get_video_container(self.video_path)

    def read_frames(
        self,
        start_pts: SECOND_TYPE = 0.0,
        end_pts: Optional[SECOND_TYPE] = None,
        fps: Optional[float] = None,
    ) -> Generator[av.VideoFrame, None, None]:
        """
        Yield frames between start_pts (inclusive) and end_pts (exclusive) in seconds.

        Args:
            start_pts (float | Fraction): Start time in seconds.
            end_pts (float | Fraction | None): End time in seconds. If None, reads until end.
                Negative values indicate time relative to the end of the video (like Python indexing).
            fps (float | None): If set, yield frames sampled at this rate (Hz).
        """
        global _CALLED_TIMES
        _CALLED_TIMES += 1
        if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == 0:
            gc.collect()

        if not isinstance(start_pts, (float, int, Fraction)):
            raise TypeError("start_pts must be float, int, or Fraction")

        # Handle negative end_pts (Python-style indexing)
        if end_pts is not None and float(end_pts) < 0:
            # Get video duration in seconds
            if self.container.duration is None:
                raise ValueError("Video duration is not available, cannot handle negative end_pts.")
            else:
                duration = self.container.duration / av.time_base
                end_pts = duration + float(end_pts)
                logger.debug(f"Negative end_pts converted to {end_pts}s (video duration: {duration}s)")

        if end_pts is None:
            end_pts = float("inf")
        elif not isinstance(end_pts, (float, int, Fraction)):
            raise TypeError("end_pts must be float, Fraction, or None")

        video_stream = self.container.streams.video[0]
        logger.debug(f"Video average rate: {video_stream.average_rate}")
        timestamp_ts = int(av.time_base * float(start_pts))
        self.container.seek(timestamp_ts)

        if fps is None:
            # Yield all frames in the interval
            for frame in self.container.decode(video=0):
                if frame.time is None:
                    raise ValueError("Frame time is None, cannot read frames without valid timestamps.")
                if frame.time < float(start_pts):
                    continue
                if frame.time > float(end_pts):
                    break
                yield frame
        else:
            # Sample frames at the given fps
            if fps <= 0:
                raise ValueError("fps must be a positive number")
            interval = 1.0 / fps
            next_time = float(start_pts)
            for frame in self.container.decode(video=0):
                if frame.time is None:
                    raise ValueError("Frame time is None, cannot read frames without valid timestamps.")
                if frame.time < float(start_pts):
                    continue
                if frame.time > float(end_pts):
                    break
                # Only yield frames at or after the next_time
                if frame.time + 1e-8 >= next_time:
                    frame.duration = interval / video_stream.time_base  # Set duration for the frame
                    yield frame
                    next_time += interval
                    # Skip ahead if frame.time is much larger than next_time (e.g., VFR)
                    if frame.time > next_time:
                        # Catch up next_time to current frame.time + interval
                        missed = int((frame.time - next_time) // interval) + 1
                        next_time += missed * interval

    def read_frame(self, pts: SECOND_TYPE = 0.0) -> av.VideoFrame:
        """
        Read and return the first frame at or after the given timestamp (in seconds).

        Args:
            pts (float | Fraction): Time in seconds to seek.

        Returns:
            av.VideoFrame: The first frame found.

        Raises:
            ValueError: If no frame is found.
        """
        for frame in self.read_frames(start_pts=pts, end_pts=None):
            return frame
        raise ValueError(f"Frame not found at {float(pts):.2f}s in {self.video_path}")

    def close(self) -> None:
        """Release the container reference or close completely if force_close was set."""
        if self.force_close:
            # Force immediate closure and removal from cache
            force_close_video_container(self.video_path)
        else:
            # Use normal cached release
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

    # Read back frames starting at 0.5 seconds
    with VideoReader(video_path) as reader:
        for frame in reader.read_frames(start_pts=Fraction(1, 2)):
            print(f"PTS: {frame.pts}, Time: {frame.time}, Shape: {frame.to_ndarray(format='rgb24').shape}")
        try:
            frame = reader.read_frame(pts=Fraction(1, 2))
            print(f"Single frame at 0.5s: PTS={frame.pts}, Time={frame.time}")
        except ValueError as e:
            logger.error(e)
