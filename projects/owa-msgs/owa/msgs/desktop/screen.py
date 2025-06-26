"""
Desktop screen capture message definitions.

Minimal, clean design focused on essential functionality.
"""

from pathlib import Path
from typing import Optional, Self, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from owa.core.io import encode_to_base64, load_image_as_bgra, load_video_frame_as_bgra
from owa.core.message import OWAMessage


class MediaRef(BaseModel):
    """Simple media reference."""

    uri: str = Field(
        ...,
        description="URI(data:image/png;base64,... | file:///path | http[s]://...) or file path(/absolute/path | relative/path)",
    )
    pts_ns: Optional[int] = Field(None, description="Video frame timestamp in nanoseconds")

    @property
    def is_embedded(self) -> bool:
        """True if this is embedded data (data URI)."""
        return self.uri.startswith("data:")

    @property
    def is_video(self) -> bool:
        """True if this references video media."""
        return self.pts_ns is not None

    @property
    def is_remote(self) -> bool:
        """True if this references a remote URL (http/https)."""
        return self.uri.startswith(("http://", "https://"))

    @property
    def is_local(self) -> bool:
        """True if this references a local file path (not embedded or remote)."""
        return not self.is_embedded and not self.is_remote

    @property
    def is_relative_path(self) -> bool:
        """True if this is a relative path (not absolute, not URI)."""
        if self.is_embedded or self.is_remote or self.uri.startswith("file://"):
            return False
        return not Path(self.uri).is_absolute()

    def resolve_relative_path(self, base_path: str) -> "MediaRef":
        """
        Resolve relative path against a base path.

        Args:
            base_path: Base path (typically MCAP file path) to resolve against

        Returns:
            New MediaRef with resolved absolute path
        """
        if not self.is_relative_path:
            return self  # Already absolute or not a local path

        # Determine if base_path is a file or directory
        # If it has an extension or doesn't end with '/', treat as file
        base_path_obj = Path(base_path)
        if base_path_obj.suffix or not base_path.endswith("/"):
            # Treat as file path, use parent directory
            base = base_path_obj.parent
        else:
            # Treat as directory path
            base = base_path_obj

        resolved_path = (base / self.uri).resolve()

        return MediaRef(uri=str(resolved_path), pts_ns=self.pts_ns)


class ScreenCaptured(OWAMessage):
    """
    Screen capture message with flexible media handling.

    Creation patterns:
    - From raw image: ScreenCaptured(frame_arr=numpy_array).embed_as_data_uri()
    - From file path: ScreenCaptured(media_ref={"uri": "/path/to/image.png"})
    - From URL: ScreenCaptured(media_ref={"uri": "https://example.com/image.png"})
    - From data URI: ScreenCaptured(media_ref={"uri": "data:image/png;base64,..."})
    - From video frame: ScreenCaptured(media_ref={"uri": "/path/video.mp4", "pts_ns": 123456})

    Image access:
    - to_rgb_array(): Get RGB numpy array
    - to_pil_image(): Get PIL Image object

    Path resolution:
    - resolve_external_path(base_path): Resolve relative paths against base directory

    Serialization requires media_ref (use embed_as_data_uri() for in-memory arrays).
    """

    _type = "desktop/ScreenCaptured"

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    # Essential fields only
    utc_ns: Optional[int] = Field(None, description="Time since epoch as nanoseconds")
    source_shape: Optional[Tuple[int, int]] = Field(
        None, description="Original source dimensions before any processing (width, height)"
    )
    shape: Optional[Tuple[int, int]] = Field(
        None, description="Current frame dimensions after any processing (width, height)"
    )
    media_ref: Optional[MediaRef] = Field(None, description="Structured media reference")
    frame_arr: SkipJsonSchema[Optional[np.ndarray]] = Field(
        None, exclude=True, description="Frame as numpy array (in-memory only)"
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
            h, w = self.frame_arr.shape[:2]
            self.shape = (w, h)

        return self

    def model_dump_json(self, **kwargs) -> str:
        """Ensure media_ref exists before JSON serialization."""
        if self.media_ref is None:
            raise ValueError("Cannot serialize without media_ref. Use embed_as_data_uri() first.")
        return super().model_dump_json(**kwargs)

    # Core methods
    def load_frame_array(self, force_close: bool = False) -> np.ndarray:
        """Load frame data from media reference as BGRA numpy array."""
        if self.frame_arr is not None:
            return self.frame_arr

        if self.media_ref is None:
            raise ValueError("No media reference available for loading")

        # Load based on media type
        if self.media_ref.is_video:
            self.frame_arr = load_video_frame_as_bgra(self.media_ref.uri, self.media_ref.pts_ns, force_close)
        else:
            self.frame_arr = load_image_as_bgra(self.media_ref.uri)

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

        base64_data = encode_to_base64(self.frame_arr, format, quality)
        self.media_ref = MediaRef(uri=f"data:image/{format};base64,{base64_data}")
        return self

    def to_rgb_array(self) -> np.ndarray:
        """Return frame as RGB numpy array."""
        bgra_array = self.load_frame_array()
        return cv2.cvtColor(bgra_array, cv2.COLOR_BGRA2RGB)

    def to_pil_image(self):
        """Convert frame to PIL Image."""
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("Pillow required for PIL conversion") from e

        rgb_array = self.to_rgb_array()
        return Image.fromarray(rgb_array, mode="RGB")

    def resolve_external_path(self, base_path: str) -> Self:
        """
        Resolve relative paths in media_ref against a base path.

        Args:
            base_path: Base path (typically MCAP file path) to resolve against

        Returns:
            Self for method chaining
        """
        if self.media_ref is None:
            return self

        self.media_ref = self.media_ref.resolve_relative_path(base_path)
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
                parts.append(f"video@{self.media_ref.pts_ns}ns")
            else:
                parts.append("external")
        return f"ScreenCaptured({', '.join(parts)})"
