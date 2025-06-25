"""
Desktop screen capture message definitions.

This module contains message types for screen capture data and events,
following the domain-based message naming convention for better organization.
"""

import base64
from fractions import Fraction
from pathlib import Path
from typing import Literal, Optional, Self, Tuple, Union

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from owa.core.io import load_image
from owa.core.io.video import VideoReader
from owa.core.message import OWAMessage
from owa.core.time import TimeUnits


class EmbeddedRef(BaseModel):
    """Reference to embedded compressed image data."""

    type: Literal["embedded"] = "embedded"
    format: Literal["png", "jpeg"]
    data: str  # base64 encoded image data


class ExternalRef(BaseModel):
    """Reference to external media file (image or video with timestamp)."""

    type: Literal["external"] = "external"
    path: str  # file path or URL
    pts_ns: Optional[int] = None  # timestamp in nanoseconds (for video frames), None for static images


# Union type for all media reference types
MediaRef = Union[EmbeddedRef, ExternalRef]


# ============================================================================
# Helper Functions for Media Processing
# ============================================================================


def _compress_frame_to_embedded(
    frame_arr: np.ndarray, format: Literal["png", "jpeg"] = "png", quality: Optional[int] = None
) -> EmbeddedRef:
    """
    Compress frame array to embedded reference.

    Args:
        frame_arr: BGRA frame array
        format: Compression format
        quality: JPEG quality (1-100)

    Returns:
        EmbeddedRef: Compressed embedded reference
    """

    # Use provided quality or default
    compression_quality = quality if quality is not None else 85
    if not (1 <= compression_quality <= 100):
        raise ValueError("JPEG quality must be between 1 and 100")

    # Convert BGRA to RGB for encoding
    rgb_array = cv2.cvtColor(frame_arr, cv2.COLOR_BGRA2RGB)

    # Encode based on format
    if format == "png":
        success, encoded = cv2.imencode(".png", cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
    elif format == "jpeg":
        success, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, compression_quality],
        )
    else:
        raise ValueError(f"Unsupported format: {format}")

    if not success:
        raise ValueError(f"Failed to encode image as {format}")

    # Create embedded reference
    base64_data = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return EmbeddedRef(format=format, data=base64_data)


def _load_from_embedded(embedded_ref: EmbeddedRef) -> np.ndarray:
    """Load frame from embedded data."""

    image_bytes = base64.b64decode(embedded_ref.data)

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr_array is None:
        raise ValueError(f"Failed to decode embedded {embedded_ref.format} data")

    # Convert BGR to BGRA
    return cv2.cvtColor(bgr_array, cv2.COLOR_BGR2BGRA)


def _load_from_external(external_ref: ExternalRef, *, force_close: bool = False) -> np.ndarray:
    """Load frame from external media reference."""
    path = external_ref.path
    pts_ns = external_ref.pts_ns

    # Validate file exists for local files only
    if not path.startswith(("http://", "https://")):
        media_path = Path(path)
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")

    # Determine if this is a video frame or static image
    if pts_ns is not None:
        return _load_video_frame(path, pts_ns, force_close)
    else:
        return _load_static_image(path)


def _load_video_frame(path: str, pts_ns: int, force_close: bool) -> np.ndarray:
    """Load a specific frame from video."""
    # Convert nanoseconds to Fraction for VideoReader
    pts_fraction = Fraction(pts_ns, TimeUnits.SECOND)

    try:
        with VideoReader(path, force_close=force_close) as reader:
            frame = reader.read_frame(pts=pts_fraction)

            # Convert to RGB first, then to BGRA
            rgb_array = frame.to_ndarray(format="rgb24")
            return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)

    except Exception as e:
        source_type = "remote" if path.startswith(("http://", "https://")) else "local"
        pts_seconds = pts_ns / 1_000_000_000
        raise ValueError(f"Failed to load frame at {pts_seconds:.3f}s from {source_type} video {path}: {e}") from e


def _load_static_image(path: str) -> np.ndarray:
    """Load a static image file."""
    try:
        # Load image using owa.core.io.load_image
        pil_image = load_image(path)

        # Convert PIL image to numpy array (RGB format)
        rgb_array = np.array(pil_image)

        # Convert RGB to BGRA
        bgra_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)

        return bgra_array

    except Exception as e:
        source_type = "remote" if path.startswith(("http://", "https://")) else "local"
        raise ValueError(f"Failed to load {source_type} image from {path}: {e}") from e


def _get_media_info(media_ref: Optional[MediaRef]) -> dict:
    """Get information about the media reference."""
    if media_ref is None:
        return {"type": None}

    if media_ref.type == "embedded":
        size_bytes = len(base64.b64decode(media_ref.data))
        return {
            "type": "embedded",
            "format": media_ref.format,
            "size_bytes": size_bytes,
        }

    if media_ref.type == "external":
        is_remote = media_ref.path.startswith(("http://", "https://"))

        info = {
            "type": "external",
            "path": media_ref.path,
            "is_local": not is_remote,
            "is_remote": is_remote,
            "media_type": "video" if media_ref.pts_ns is not None else "image",
        }

        if media_ref.pts_ns is not None:
            info["pts_ns"] = media_ref.pts_ns
            info["pts_seconds"] = media_ref.pts_ns / 1_000_000_000  # Also provide seconds for convenience

        return info

    return {"type": "unknown"}


def _format_media_display(media_ref: MediaRef) -> str:
    """Format media reference for display in string representation."""
    if media_ref.type == "embedded":
        size_kb = len(base64.b64decode(media_ref.data)) / 1024
        return f"embedded_{media_ref.format}({size_kb:.1f}KB)"

    elif media_ref.type == "external":
        path_display = (
            Path(media_ref.path).name if not media_ref.path.startswith(("http://", "https://")) else media_ref.path
        )

        if media_ref.pts_ns is not None:
            # Video frame
            prefix = "remote" if media_ref.path.startswith(("http://", "https://")) else "local"
            pts_seconds = media_ref.pts_ns / 1_000_000_000
            return f"{prefix}_video({path_display}@{pts_seconds:.3f}s)"
        else:
            # Static image
            prefix = "remote" if media_ref.path.startswith(("http://", "https://")) else "local"
            return f"{prefix}_image({path_display})"

    return "unknown_media"


# ============================================================================
# Main Message Class
# ============================================================================


class ScreenCaptured(OWAMessage):
    """
    Represents a captured screen frame with structured media reference system.

    This message can contain frame data in several formats:
    1. Structured media reference (media_ref) - typed system for both embedded and external data
    2. Direct numpy array (frame_arr) - in-memory only, excluded from serialization
    """

    _type = "desktop/ScreenCaptured"

    model_config = {"arbitrary_types_allowed": True}

    # Time since epoch as nanoseconds.
    utc_ns: Optional[int] = None
    # Original source(commonly monitor or window) dimensions before any processing, e.g. (width, height)
    source_shape: Optional[Tuple[int, int]] = None
    # Current frame dimensions after any processing, e.g. (width, height)
    shape: Optional[Tuple[int, int]] = None

    # Structured media reference
    media_ref: Optional[MediaRef] = None

    # The frame as a numpy array (optional, can be lazy-loaded) - excluded from serialization
    frame_arr: SkipJsonSchema[Optional[np.ndarray]] = Field(None, exclude=True)

    @model_validator(mode="after")
    def validate_screen_emitted(self) -> "ScreenCaptured":
        """Validate that at least one frame data source is provided."""
        # At least one source must be provided
        if self.frame_arr is None and self.media_ref is None:
            raise ValueError("ScreenCaptured requires either 'frame_arr' or 'media_ref' to be provided")

        # Validate frame_arr if provided and set shape
        if self.frame_arr is not None:
            if len(self.frame_arr.shape) < 2:
                raise ValueError("frame_arr must be at least 2-dimensional")

            # Always set shape based on actual frame dimensions (width, height)
            h, w = self.frame_arr.shape[:2]
            self.shape = (w, h)

        return self

    def model_dump_json(self, **kwargs) -> str:
        """Override model_dump_json to ensure media_ref exists before JSON serialization."""
        if self.media_ref is None:
            raise ValueError(
                "Cannot serialize ScreenCaptured to JSON without media_ref. "
                "Use embed_from_array() or set_external_reference() to create a media reference first."
            )
        return super().model_dump_json(**kwargs)

    # Reference type checking methods
    def has_embedded_data(self) -> bool:
        """Check if this frame has embedded data."""
        return self.media_ref is not None and self.media_ref.type == "embedded"

    def has_external_reference(self) -> bool:
        """Check if this frame has external media reference."""
        return self.media_ref is not None and self.media_ref.type == "external"

    # Media reference creation methods
    def embed_from_array(self, format: Literal["png", "jpeg"] = "png", *, quality: Optional[int] = None) -> Self:
        """Compress and embed the current frame_arr data."""
        if self.frame_arr is None:
            raise ValueError("No frame_arr available to embed")

        self.media_ref = _compress_frame_to_embedded(self.frame_arr, format, quality)
        return self

    def set_external_reference(self, path: Union[str, Path], pts_ns: Optional[int] = None) -> None:
        """Set an external media reference."""
        if isinstance(path, Path):
            path = str(path)

        self.media_ref = ExternalRef(path=path, pts_ns=pts_ns)

    # Frame loading and conversion methods
    def lazy_load(self, *, force_close: bool = False) -> np.ndarray:
        """Lazy load the frame data from any available source."""
        if self.frame_arr is not None:
            return self.frame_arr

        if self.media_ref is None:
            raise ValueError("No frame data sources available for loading")

        # Load based on reference type
        if self.media_ref.type == "embedded":
            self.frame_arr = _load_from_embedded(self.media_ref)
        elif self.media_ref.type == "external":
            self.frame_arr = _load_from_external(self.media_ref, force_close=force_close)
        else:
            raise ValueError(f"Unsupported media reference type: {self.media_ref.type}")

        # Update shape information based on loaded frame
        h, w = self.frame_arr.shape[:2]
        shape_tuple = (w, h)
        self.shape = shape_tuple
        if self.source_shape is None:
            self.source_shape = shape_tuple

        return self.frame_arr

    def to_rgb_array(self) -> np.ndarray:
        """Return the frame as an RGB numpy array."""
        # Ensure frame is loaded
        bgra_array = self.lazy_load()

        # Convert BGRA to RGB
        return cv2.cvtColor(bgra_array, cv2.COLOR_BGRA2RGB)

    def to_pil_image(self):
        """Convert the frame to a PIL Image in RGB format."""
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("Pillow is required for PIL Image conversion") from e

        rgb_array = self.to_rgb_array()
        return Image.fromarray(rgb_array, mode="RGB")

    def get_media_info(self) -> dict:
        """Get information about the media reference."""
        return _get_media_info(self.media_ref)

    def __str__(self) -> str:
        """Return a concise string representation of the ScreenCaptured instance."""
        # Core attributes to display
        attrs = ["utc_ns", "source_shape", "shape"]
        attr_strs = []

        for attr in attrs:
            value = getattr(self, attr)
            if value is not None:
                attr_strs.append(f"{attr}={value!r}")

        # Add memory info if loaded
        if self.frame_arr is not None:
            memory_mb = self.frame_arr.nbytes / (1024 * 1024)
            attr_strs.append(f"loaded({memory_mb:.1f}MB)")

        # Add media info
        if self.media_ref:
            attr_strs.append(_format_media_display(self.media_ref))

        return f"{self.__class__.__name__}({', '.join(attr_strs)})"

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return self.__str__()
