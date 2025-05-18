import gc
import threading
from typing import Optional, Tuple

import av
import cv2
import numpy as np
from PIL import Image
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

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
            np.ndarray: The frame as a RGB array

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

                for frame in container.decode(stream):
                    frame_time = frame.pts * stream.time_base
                    if frame_time >= target_time:
                        # Convert frame to RGB format (avoid alpha for safety)
                        rgb_frame = frame.to_ndarray(format="rgb24")
                        return rgb_frame

                raise ValueError(f"No frame found at PTS: {pts_ns} ns")

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

    model_config = {"arbitrary_types_allowed": True}

    # Time since epoch as nanoseconds.
    utc_ns: int | None = None
    # The frame as a numpy array (optional, can be lazy-loaded)
    frame_arr: SkipJsonSchema[Optional[np.ndarray]] = Field(None, exclude=True)
    # Original shape of the frame before rescale, e.g. (width, height)
    original_shape: Optional[Tuple[int, int]] = None
    # Rescaled shape of the frame, e.g. (width, height)
    shape: Optional[Tuple[int, int]] = None

    # Path to the stream, e.g. output.mkv (optional)
    path: str | None = None
    # Time since stream start as nanoseconds.
    pts: int | None = None

    def model_post_init(self, __context):
        # At least one of frame_arr or (path and pts) must be provided
        if self.frame_arr is None:
            if self.path is None or self.pts is None:
                raise ValueError("ScreenEmitted requires either 'frame_arr' or both 'path' and 'pts' to be provided.")

    def lazy_load(self) -> np.ndarray:
        """
        Lazy load the frame data if not already set.
        This is called when the object is created and frame_arr is None.
        Returns:
            np.ndarray: the frame as a BGRA array.
        """
        if self.frame_arr is None and self.path is not None and self.pts is not None:
            # Always load as RGB for safety, then convert to BGRA for consumers.
            rgb_array = _video_reader.get_frame_at_pts(self.path, self.pts)
            self.frame_arr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)
            # Set shape based on the loaded frame (width, height)
            # Note: OpenCV shape is (height, width, channels), but width/height tuple is (width, height).
            h, w = self.frame_arr.shape[:2]
            shape_tuple = (w, h)
            self.shape = shape_tuple
            self.original_shape = shape_tuple
        return self.frame_arr

    def to_pil_image(self) -> Image.Image:
        """
        Convert the frame at the specified PTS to a PIL Image in RGB format.

        Returns:
            PIL.Image.Image: The frame as a PIL Image.
        """
        rgb_array = self.to_rgb_array()
        return Image.fromarray(rgb_array, mode="RGB")

    def to_rgb_array(self) -> np.ndarray:
        """
        Return the frame as a RGB numpy array.
        If frame_arr is not set, try to load from path and pts.
        """
        # If self.frame_arr is BGRA, convert to RGB.
        bgra_array = self.lazy_load()
        rgb_array = cv2.cvtColor(bgra_array, cv2.COLOR_BGRA2RGB)
        return rgb_array

    def __repr__(self):
        # give concise summary
        ks = ["utc_ns", "shape", "original_shape", "path", "pts"]
        return f"{self.__class__.__name__}({', '.join(f'{k}={getattr(self, k)!r}' for k in ks)})"


def main():
    d = {"path": "output.mkv", "pts": 2683333333, "utc_ns": 1741608540328534500}
    d = {"path": "output.mkv", "pts": int(10**9 * (0.99)), "utc_ns": 1741608540328534500}
    frame = ScreenEmitted(**d)

    print(frame)
    print(frame.to_pil_image())
    print(frame.shape)

    # Clean up at the end
    _video_reader.clear_cache()


if __name__ == "__main__":
    main()
