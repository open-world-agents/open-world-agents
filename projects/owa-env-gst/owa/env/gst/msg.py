"""
Legacy message definitions for backward compatibility.

DEPRECATED: These message definitions have been moved to the owa-msgs package
for better organization and centralized management. Please use the new imports:

    from owa.msgs.desktop.screen import ScreenCaptured

Or access via the message registry:

    from owa.core import MESSAGES
    ScreenCaptured = MESSAGES['desktop/ScreenCaptured']

This module provides compatibility imports and will be removed in a future version.
"""

import warnings
from fractions import Fraction
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pydantic import Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from owa.core.io.video import VideoReader
from owa.core.message import OWAMessage
from owa.core.time import TimeUnits

# Import new message classes for compatibility
try:
    from owa.msgs.desktop.screen import ScreenCaptured as _NewScreenEmitted
except ImportError:
    # Fallback if owa-msgs is not installed
    _NewScreenEmitted = None


def _deprecation_warning(old_name: str, new_import: str) -> None:
    """Issue deprecation warning for legacy message usage."""
    warnings.warn(
        f"Using {old_name} from owa.env.gst.msg is deprecated. Use: {new_import}", DeprecationWarning, stacklevel=3
    )


class ScreenEmitted:
    """Legacy ScreenEmitted - redirects to new implementation."""

    def __new__(cls, *args, **kwargs):
        _deprecation_warning("owa.env.gst.msg.ScreenEmitted", "from owa.msgs.desktop.screen import ScreenCaptured")
        if _NewScreenEmitted is not None:
            return _NewScreenEmitted(*args, **kwargs)
        else:
            return _LegacyScreenEmitted(*args, **kwargs)

    @classmethod
    def deserialize(cls, buffer):
        """Deserialize method for legacy compatibility."""
        if _NewScreenEmitted is not None:
            return _NewScreenEmitted.deserialize(buffer)
        else:
            return _LegacyScreenEmitted.deserialize(buffer)


# Fallback implementation for when owa-msgs is not available
class _LegacyScreenEmitted(OWAMessage):
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

    @model_validator(mode="after")
    def validate_screen_emitted(self) -> "ScreenEmitted":
        """Validate that either frame_arr or (path and pts) are provided."""
        # At least one of frame_arr or (path and pts) must be provided
        if self.frame_arr is None:
            if self.path is None or self.pts is None:
                raise ValueError("ScreenEmitted requires either 'frame_arr' or both 'path' and 'pts' to be provided.")

        # Validate frame_arr if provided
        if self.frame_arr is not None:
            if len(self.frame_arr.shape) < 2:
                raise ValueError("frame_arr must be at least 2-dimensional")

            # Set shape based on frame dimensions (width, height)
            h, w = self.frame_arr.shape[:2]
            self.shape = (w, h)

        # Validate pts if provided
        if self.pts is not None and self.pts < 0:
            raise ValueError("pts must be non-negative")

        return self

    def lazy_load(self, *, force_close: bool = False) -> np.ndarray:
        """
        Lazy load the frame data if not already set.

        Args:
            force_close: Force complete closure of video container instead of using cache

        Returns:
            np.ndarray: The frame as a BGRA array

        Raises:
            ValueError: If required parameters are missing or frame not found
        """
        if self.frame_arr is not None:
            return self.frame_arr

        if self.path is None or self.pts is None:
            raise ValueError("Cannot lazy load: both 'path' and 'pts' must be provided")

        # Convert PTS from nanoseconds to seconds for VideoReader
        pts_seconds = Fraction(self.pts, TimeUnits.SECOND)

        with VideoReader(self.path, force_close=force_close) as reader:
            frame = reader.read_frame(pts=pts_seconds)

            # Convert to RGB first, then to BGRA for consumers
            rgb_array = frame.to_ndarray(format="rgb24")
            self.frame_arr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)

            # Set shape based on the loaded frame (width, height)
            h, w = self.frame_arr.shape[:2]
            shape_tuple = (w, h)
            self.shape = shape_tuple
            self.original_shape = shape_tuple

        return self.frame_arr

    def to_rgb_array(self) -> np.ndarray:
        """
        Return the frame as an RGB numpy array.

        Returns:
            np.ndarray: The frame as an RGB array with shape (height, width, 3)
        """
        # Ensure frame is loaded
        bgra_array = self.lazy_load()

        # Convert BGRA to RGB
        rgb_array = cv2.cvtColor(bgra_array, cv2.COLOR_BGRA2RGB)
        return rgb_array

    def to_pil_image(self) -> Image.Image:
        """
        Convert the frame to a PIL Image in RGB format.

        Returns:
            PIL.Image.Image: The frame as a PIL Image in RGB mode
        """
        rgb_array = self.to_rgb_array()
        return Image.fromarray(rgb_array, mode="RGB")

    def is_loaded(self) -> bool:
        """
        Check if frame data is already loaded in memory.

        Returns:
            bool: True if frame_arr is loaded, False otherwise
        """
        return self.frame_arr is not None

    def has_video_reference(self) -> bool:
        """
        Check if this instance has a valid video file reference.

        Returns:
            bool: True if both path and pts are provided, False otherwise
        """
        return self.path is not None and self.pts is not None

    def get_memory_usage(self) -> int:
        """
        Estimate memory usage of the loaded frame in bytes.

        Returns:
            int: Estimated memory usage in bytes, 0 if not loaded
        """
        if self.frame_arr is None:
            return 0
        return self.frame_arr.nbytes

    def __str__(self) -> str:
        """Return a concise string representation of the ScreenEmitted instance."""
        # Core attributes to display
        attrs = ["utc_ns", "shape", "original_shape", "path", "pts"]
        attr_strs = []

        for attr in attrs:
            value = getattr(self, attr)
            if value is not None:
                attr_strs.append(f"{attr}={value!r}")

        # Add loading status
        if self.is_loaded():
            attr_strs.append("loaded=True")

        return f"{self.__class__.__name__}({', '.join(attr_strs)})"


def main():
    """Demonstration of ScreenEmitted functionality."""

    # Example 1: Create with video reference
    video_data = {
        "path": "output.mkv",
        "pts": int(10**9 * 0.99),  # 0.99 seconds in nanoseconds
        "utc_ns": 1741608540328534500,
    }
    frame = ScreenEmitted(**video_data)

    print("=== ScreenEmitted Demo ===")
    print(f"Created frame: {frame}")
    print(f"Is loaded: {frame.is_loaded()}")
    print(f"Has video reference: {frame.has_video_reference()}")
    print(f"Memory usage: {frame.get_memory_usage()} bytes")

    # Note: The following would attempt to load from video file
    # Uncomment if you have a valid video file:
    # pil_image = frame.to_pil_image()
    # print(f"PIL Image size: {pil_image.size}")
    # print(f"Shape after loading: {frame.shape}")

    # Example 2: Create with numpy array (if available)

    # Create a small test frame (BGRA format)
    test_frame = np.zeros((100, 200, 4), dtype=np.uint8)  # height=100, width=200
    test_frame[:, :, 2] = 255  # Red channel
    test_frame[:, :, 3] = 255  # Alpha channel

    frame_with_array = ScreenEmitted(utc_ns=1741608540328534500, frame_arr=test_frame)

    print(f"\nFrame with array: {frame_with_array}")
    print(f"Shape: {frame_with_array.shape}")
    print(f"Memory usage: {frame_with_array.get_memory_usage()} bytes")


if __name__ == "__main__":
    main()
