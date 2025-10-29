"""Desktop screen capture message definitions."""

from pathlib import Path
from typing import Optional, Self, Tuple, cast

import cv2
import numpy as np
from mediaref import DataURI, MediaRef
from pydantic import Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from owa.core.message import OWAMessage
from owa.core.time import TimeUnits


class ScreenCaptured(OWAMessage):
    """
    Screen capture message with flexible media handling.

    Creation patterns:
    - From raw image: ScreenCaptured(frame_arr=bgra_np_array).embed_as_data_uri()
    - From file path: ScreenCaptured(media_ref={"uri": "/path/to/image.png"})
    - From URL: ScreenCaptured(media_ref={"uri": "https://example.com/image.png"})
    - From data URI: ScreenCaptured(media_ref={"uri": "data:image/png;base64,..."})
    - From video frame: ScreenCaptured(media_ref={"uri": "/path/video.mp4", "pts_ns": 123456})

    Image access:
    - to_rgb_array(): Get RGB numpy array
    - to_pil_image(): Get PIL Image object

    Path resolution:
    - resolve_relative_path(base_path): Resolve relative paths against base directory

    Serialization requires media_ref (use embed_as_data_uri() for in-memory arrays).
    """

    _type = "desktop/ScreenCaptured"

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    # Essential fields only
    utc_ns: Optional[int] = Field(default=None, description="Time since epoch as nanoseconds")
    source_shape: Optional[Tuple[int, int]] = Field(
        default=None, description="Original source dimensions before any processing (width, height)"
    )
    shape: Optional[Tuple[int, int]] = Field(
        default=None, description="Current frame dimensions after any processing (width, height)"
    )
    media_ref: Optional[MediaRef] = Field(default=None, description="Structured media reference")
    frame_arr: SkipJsonSchema[Optional[np.ndarray]] = Field(
        default=None, exclude=True, description="BGRA frame as numpy array (in-memory only)"
    )

    @model_validator(mode="after")
    def validate_data(self) -> Self:
        """Validate that we have either frame_arr or media_ref."""
        if self.frame_arr is None and self.media_ref is None:
            raise ValueError("Either frame_arr or media_ref must be provided")

        # Set shape from frame_arr if available
        if self.frame_arr is not None:
            if len(self.frame_arr.shape) < 2:
                raise ValueError("frame_arr must be at least 2-dimensional")
            if self.frame_arr.shape[2] != 4:
                raise ValueError("frame_arr must be BGRA format")
            h, w = self.frame_arr.shape[:2]
            self.shape = (w, h)

        return self

    def model_dump_json(self, **kwargs) -> str:
        """Ensure media_ref exists before JSON serialization."""
        if self.media_ref is None:
            raise ValueError("Cannot serialize without media_ref. Use embed_as_data_uri() first.")
        return super().model_dump_json(**kwargs)

    # Core methods
    def load_frame_array(self, *, keep_av_open: bool = False) -> np.ndarray:
        """Load frame data from media reference as BGRA numpy array."""
        if self.frame_arr is not None:
            return self.frame_arr

        if self.media_ref is None:
            raise ValueError("No media reference available for loading")

        # Load using mediaref package and convert RGB to BGRA
        rgb_array = self.media_ref.to_rgb_array()
        self.frame_arr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)

        # Update shape
        h, w = self.frame_arr.shape[:2]
        self.shape = (w, h)
        if self.source_shape is None:
            self.source_shape = self.shape

        return self.frame_arr

    def embed_as_data_uri(self, format: str = "png", quality: Optional[int] = None) -> Self:
        """Embed current frame_arr as data URI in media_ref."""
        if self.frame_arr is None:
            raise ValueError("No frame_arr available to embed")

        # Convert BGRA to RGB for mediaref
        rgb_array = cv2.cvtColor(self.frame_arr, cv2.COLOR_BGRA2RGB)

        # Use mediaref's DataURI.from_image to create data URI
        data_uri = DataURI.from_image(rgb_array, format=format, quality=quality)
        self.media_ref = MediaRef(uri=data_uri)
        return self

    def to_rgb_array(self, *, keep_av_open: bool = False) -> np.ndarray:
        """Return frame as RGB numpy array."""
        bgra_array = self.load_frame_array(keep_av_open=keep_av_open)
        return cv2.cvtColor(bgra_array, cv2.COLOR_BGRA2RGB)

    def to_pil_image(self, *, keep_av_open: bool = False):
        """Convert frame to PIL Image."""
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("Pillow required for PIL conversion") from e

        rgb_array = self.to_rgb_array(keep_av_open=keep_av_open)
        return Image.fromarray(rgb_array)

    def resolve_relative_path(self, mcap_path: str, **kwargs) -> Self:
        """
        Resolve relative paths in media_ref against a base path.

        Args:
            mcap_path: MCAP file path or directory path to resolve against
            **kwargs: Additional arguments passed to MediaRef.resolve_relative_path()

        Returns:
            Self for method chaining
        """
        if self.media_ref is None:
            return self

        # If given path is an MCAP file, use its parent directory as base path
        base_path_obj = Path(mcap_path)
        if base_path_obj.suffix == ".mcap":
            base_path_obj = base_path_obj.parent
        base_path = base_path_obj.as_posix()

        self.media_ref = self.media_ref.resolve_relative_path(base_path, **kwargs)
        return self

    def __str__(self) -> str:
        """Simple string representation."""
        parts = []
        if self.utc_ns:
            parts.append(f"utc_ns={self.utc_ns}")
        if self.source_shape:
            parts.append(f"source_shape={self.source_shape}")
        if self.shape:
            parts.append(f"shape={self.shape}")
        if self.frame_arr is not None:
            mb = self.frame_arr.nbytes / (1024 * 1024)
            parts.append(f"loaded({mb:.1f}MB)")
        if self.media_ref:
            if self.media_ref.is_embedded:
                parts.append("embedded")
            elif self.media_ref.is_video:
                parts.append(f"video@{cast(int, self.media_ref.pts_ns) / TimeUnits.SECOND:.3f}s")
            else:
                parts.append("external")
        return f"ScreenCaptured({', '.join(parts)})"
