import gc
import threading
from typing import Optional

import av
import numpy as np
from PIL import Image

from owa.core.message import OWAMessage
from owa.core.time import TimeUnits


# BUG: PyAV has "corrupted size vs. prev_size" error when `frame.to_ndarray(format="bgra")` is called for video "expert-jy-1.mkv"
#      This bug does not occur when format does not contain `alpha` channel, e.g. "bgr24"
#      Guessed reason is mismatch of width/height=770/512 and codec_width/codec_height=800/512.
class PyAVVideoReader:
    """Class responsible for reading video files and extracting frames at specified timestamps."""

    _GC_COLLECT_COUNT = 0
    _GC_COLLECTION_INTERVAL = 10  # Run garbage collection every 10 video opens

    _video_container_cache = {}
    _cache_lock = threading.Lock()
    _max_cache_size = 4  # Default maximum number of cached containers

    def __init__(self, max_cache_size=None):
        """Initialize VideoReader with an optional cache size."""
        if max_cache_size is not None:
            self._max_cache_size = max_cache_size

    def get_frame_at_pts(self, video_path, pts_ns):
        """
        Extract a frame from a video at a specified PTS (in nanoseconds).

        Args:
            video_path (str): Path to the video file
            pts_ns (int): Presentation timestamp in nanoseconds

        Returns:
            np.ndarray: The frame as a BGRA array

        Raises:
            FileNotFoundError: If the video file does not exist
            ValueError: If a frame at the specified PTS cannot be found
        """
        # Increment GC counter and occasionally run garbage collection
        PyAVVideoReader._GC_COLLECT_COUNT += 1
        if PyAVVideoReader._GC_COLLECT_COUNT % self._GC_COLLECTION_INTERVAL == 0:
            # Mandatory to prevent memory leaks when processing many videos
            gc.collect()

        try:
            # Get the video container from cache or open a new one
            # container = self._get_video_container(video_path)
            # TODO: lock per video file
            with av.open(video_path) as container:
                # Convert PTS from nanoseconds to seconds
                target_time = pts_ns / TimeUnits.SECOND

                # Select the first video stream
                try:
                    stream = next(s for s in container.streams if s.type == "video")
                except StopIteration:
                    raise ValueError("No video stream found in the file.")

                # Calculate the seek position in terms of stream time base
                seek_timestamp = int(target_time / stream.time_base)

                # Flush the decoder before seeking
                # container.flush_buffers()

                # Seek to the nearest keyframe before the target time
                container.seek(seek_timestamp, any_frame=False, backward=True, stream=stream)

                frame_found = False
                bgra_frame = None

                for frame in container.decode(stream):
                    frame_time = frame.pts * stream.time_base
                    if frame_time >= target_time:
                        # Convert frame to BGRA format
                        bgra_frame = frame.to_ndarray(format="rgb24")
                        frame_found = True
                        break

                if not frame_found:
                    raise ValueError(f"No frame found at PTS: {pts_ns} ns")
                return bgra_frame

        except FileNotFoundError:
            raise FileNotFoundError(f"Video file not found: {video_path}")
        except av.FFmpegError as e:
            raise ValueError(f"Error opening video file: {e}")

    def _get_video_container(self, video_path):
        """
        Get a video container from cache or create a new one.
        Thread-safe implementation with size limiting.
        """
        with self._cache_lock:
            # Check if it's already cached
            if video_path in self._video_container_cache:
                return self._video_container_cache[video_path]

            # If cache is full, remove the oldest entry
            if len(self._video_container_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._video_container_cache))
                oldest_container = self._video_container_cache.pop(oldest_key)
                oldest_container.close()

            # Open a new container and add it to the cache
            container = av.open(video_path)
            self._video_container_cache[video_path] = container
            return container

    def clear_cache(self):
        """Close and clear all cached video containers."""
        with self._cache_lock:
            for container in self._video_container_cache.values():
                container.close()
            self._video_container_cache.clear()
        gc.collect()


# Global video reader instance
_video_reader = PyAVVideoReader()


class ScreenEmitted(OWAMessage):
    _type = "owa.env.gst.msg.ScreenEmitted"

    # Path to the stream, e.g. output.mkv
    path: str
    # Time since stream start as nanoseconds.
    pts: int
    # Time since epoch as nanoseconds.
    utc_ns: int

    _frame_arr: Optional[np.ndarray] = None

    def to_pil_image(self) -> Image.Image:
        """
        Convert the frame at the specified PTS to a PIL Image in RGBA format.

        Returns:
            PIL.Image.Image: The frame as a PIL Image.
        """
        rgb_array = self.to_rgb_array()
        # Convert BGRA to RGB by rearranging the channels
        # rgb_array = bgra_array[..., [2, 1, 0]]
        return Image.fromarray(rgb_array, mode="RGB")

    def to_rgb_array(self) -> np.ndarray:
        """
        Extract the frame at the specified PTS and return it as a rgb NumPy array.

        Returns:
            np.ndarray: The frame as a rgb array with shape (H, W, 4).

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If a frame at the specified PTS cannot be found.
        """
        if self._frame_arr is not None:
            return self._frame_arr

        # Use the VideoReader to get the frame
        self._frame_arr = _video_reader.get_frame_at_pts(self.path, self.pts)
        return self._frame_arr


class FrameStamped(OWAMessage):
    _type = "owa.env.gst.msg.FrameStamped"

    model_config = {"arbitrary_types_allowed": True}

    timestamp_ns: int
    frame_arr: np.ndarray  # [W, H, BGRA]


def main():
    d = {"path": "output.mkv", "pts": 2683333333, "utc_ns": 1741608540328534500}
    d = {"path": "output.mkv", "pts": 10**9 * (0.99), "utc_ns": 1741608540328534500}
    frame = ScreenEmitted(**d)

    print(frame)
    print(frame.to_pil_image())

    # Clean up at the end
    _video_reader.clear_cache()


if __name__ == "__main__":
    main()
