"""
Desktop screen capture message definitions.

This module contains message types for screen capture data and events,
following the domain-based message naming convention for better organization.
"""

from fractions import Fraction
from typing import Optional, Tuple

import numpy as np
from pydantic import Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from owa.core.message import OWAMessage


class ScreenEmitted(OWAMessage):
    """
    Represents a captured screen frame with optional video reference.

    This message can contain either direct frame data or a reference to video file
    with timestamp information for efficient storage and lazy loading.

    Attributes:
        utc_ns: UTC timestamp in nanoseconds since epoch
        frame_arr: Optional numpy array containing frame data (excluded from JSON)
        original_shape: Original frame dimensions before any rescaling
        shape: Current frame dimensions after rescaling
        path: Optional path to video file containing the frame
        pts: Presentation timestamp in nanoseconds from stream start
    """

    _type = "desktop/ScreenEmitted"

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

    def has_video_reference(self) -> bool:
        """
        Check if this frame has a video file reference.

        Returns:
            bool: True if both path and pts are available
        """
        return self.path is not None and self.pts is not None

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

        try:
            import cv2

            from owa.core.io.video import VideoReader
            from owa.core.time import TimeUnits
        except ImportError as e:
            raise ImportError(f"Required libraries not available for video loading: {e}")

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
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python is required for color conversion")

        # Ensure frame is loaded
        bgra_array = self.lazy_load()

        # Convert BGRA to RGB
        rgb_array = cv2.cvtColor(bgra_array, cv2.COLOR_BGRA2RGB)
        return rgb_array

    def to_pil_image(self):
        """
        Convert the frame to a PIL Image in RGB format.

        Returns:
            PIL.Image.Image: The frame as a PIL Image in RGB mode
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for PIL Image conversion")

        rgb_array = self.to_rgb_array()
        return Image.fromarray(rgb_array, mode="RGB")

    def is_loaded(self) -> bool:
        """
        Check if frame data is already loaded in memory.

        Returns:
            bool: True if frame_arr is loaded, False otherwise
        """
        return self.frame_arr is not None

    def get_memory_usage(self) -> int:
        """
        Get approximate memory usage of the frame data.

        Returns:
            int: Memory usage in bytes, 0 if not loaded
        """
        if not self.is_loaded():
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
