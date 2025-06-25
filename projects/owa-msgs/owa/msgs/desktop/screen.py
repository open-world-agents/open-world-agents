"""
Desktop screen capture message definitions.

This module contains message types for screen capture data and events,
following the domain-based message naming convention for better organization.
"""

import base64
import re
from fractions import Fraction
from pathlib import Path
from typing import Literal, Optional, Self, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from owa.core.io import load_image
from owa.core.io.video import VideoReader
from owa.core.message import OWAMessage
from owa.core.time import TimeUnits


class MediaRef(BaseModel):
    """Unified media reference supporting data URIs, file paths, and URLs."""

    uri: str = Field(..., description="URI: data:image/png;base64,... | file:///path | http[s]://... | /path/to/file")
    pts_ns: Optional[int] = Field(None, description="Video frame timestamp in nanoseconds")

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, v: str) -> str:
        if not v:
            raise ValueError("URI cannot be empty")

        # Validate data URIs
        if v.startswith("data:"):
            if not re.match(r"^data:[^;,]+(?:;[^,]*)?,[A-Za-z0-9+/]*={0,2}$", v):
                raise ValueError("Invalid data URI format")

        # Validate URLs
        elif v.startswith(("http://", "https://")):
            parsed = urlparse(v)
            if not parsed.netloc:
                raise ValueError("Invalid URL format")

        # Validate file URIs
        elif v.startswith("file://"):
            parsed = urlparse(v)
            if not parsed.path:
                raise ValueError("Invalid file URI format")

        # Local paths are accepted as-is (validation happens at load time)

        return v

    @computed_field
    @property
    def scheme(self) -> str:
        """URI scheme: 'data', 'file', 'http', 'https', or 'path'."""
        if self.uri.startswith("data:"):
            return "data"
        elif self.uri.startswith("file://"):
            return "file"
        elif self.uri.startswith(("http://", "https://")):
            return urlparse(self.uri).scheme
        else:
            return "path"

    @computed_field
    @property
    def is_embedded(self) -> bool:
        """True if this is embedded data (data URI)."""
        return self.scheme == "data"

    @computed_field
    @property
    def is_external(self) -> bool:
        """True if this references external media."""
        return self.scheme in ("file", "http", "https", "path")

    @computed_field
    @property
    def is_video(self) -> bool:
        """True if this references video media."""
        return self.pts_ns is not None or (self.is_embedded and self.mime_type and self.mime_type.startswith("video/"))

    @computed_field
    @property
    def mime_type(self) -> Optional[str]:
        """MIME type for data URIs, None for external references."""
        if not self.is_embedded:
            return None

        match = re.match(r"^data:([^;,]+)", self.uri)
        return match.group(1) if match else None

    @computed_field
    @property
    def format(self) -> Optional[str]:
        """Image format for embedded data ('png', 'jpeg'), None for external."""
        if not self.is_embedded or not self.mime_type:
            return None

        if self.mime_type == "image/png":
            return "png"
        elif self.mime_type in ("image/jpeg", "image/jpg"):
            return "jpeg"
        return None

    @computed_field
    @property
    def data(self) -> Optional[str]:
        """Base64 data for data URIs, None for external references."""
        if not self.is_embedded:
            return None

        comma_idx = self.uri.find(",")
        return self.uri[comma_idx + 1 :] if comma_idx > 0 else None

    @computed_field
    @property
    def path(self) -> Optional[str]:
        """File path or URL for external references, None for embedded data."""
        if self.is_embedded:
            return None

        if self.scheme == "file":
            return urlparse(self.uri).path
        elif self.scheme in ("http", "https"):
            return self.uri
        else:  # scheme == 'path'
            return self.uri

    # Factory methods for easy creation
    @classmethod
    def from_embedded(cls, format: Literal["png", "jpeg"], data: str, pts_ns: Optional[int] = None) -> "MediaRef":
        """Create MediaRef from embedded image data."""
        mime_type = f"image/{format}"
        uri = f"data:{mime_type};base64,{data}"
        return cls(uri=uri, pts_ns=pts_ns)

    @classmethod
    def from_path(cls, path: str, pts_ns: Optional[int] = None) -> "MediaRef":
        """Create MediaRef from file path or URL."""
        return cls(uri=path, pts_ns=pts_ns)

    @classmethod
    def from_file_uri(cls, file_path: str, pts_ns: Optional[int] = None) -> "MediaRef":
        """Create MediaRef from file:// URI."""
        if not file_path.startswith("/"):
            raise ValueError("File path must be absolute")
        uri = f"file://{file_path}"
        return cls(uri=uri, pts_ns=pts_ns)


# ============================================================================
# Helper Functions for Media Processing
# ============================================================================


def _compress_frame_to_embedded(
    frame_arr: np.ndarray, format: Literal["png", "jpeg"] = "png", quality: Optional[int] = None
) -> MediaRef:
    """
    Compress frame array to embedded reference.

    Args:
        frame_arr: BGRA frame array
        format: Compression format
        quality: JPEG quality (1-100)

    Returns:
        EmbeddedRef: Compressed embedded reference
    """
    # Convert BGRA to BGR for cv2 encoding
    bgr_array = cv2.cvtColor(frame_arr, cv2.COLOR_BGRA2BGR)

    # Encode based on format
    if format == "png":
        success, encoded = cv2.imencode(".png", bgr_array)
    elif format == "jpeg":
        if quality is None:
            quality = 85
        if not (1 <= quality <= 100):
            raise ValueError("JPEG quality must be between 1 and 100")
        success, encoded = cv2.imencode(".jpg", bgr_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        raise ValueError(f"Unsupported format: {format}")

    if not success:
        raise ValueError(f"Failed to encode image as {format}")

    # Create embedded reference
    base64_data = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return MediaRef.from_embedded(format=format, data=base64_data)


def _load_from_embedded(media_ref: MediaRef) -> np.ndarray:
    """Load frame from embedded data."""
    if not media_ref.is_embedded:
        raise ValueError("MediaRef is not embedded data")

    if not media_ref.data:
        raise ValueError("No embedded data found")

    image_bytes = base64.b64decode(media_ref.data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr_array is None:
        raise ValueError(f"Failed to decode embedded {media_ref.format} data")

    return cv2.cvtColor(bgr_array, cv2.COLOR_BGR2BGRA)


def _load_from_external_image(media_ref: MediaRef) -> np.ndarray:
    """Load frame from external image reference."""
    if not media_ref.is_external:
        raise ValueError("MediaRef is not external reference")

    if not media_ref.path:
        raise ValueError("No path found in MediaRef")

    path = media_ref.path

    # Validate file exists for local files only
    if not path.startswith(("http://", "https://")):
        media_path = Path(path)
        if not media_path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

    return _load_static_image(path)


def _load_from_external_video(media_ref: MediaRef, *, force_close: bool = False) -> np.ndarray:
    """Load frame from external video reference."""
    if not media_ref.is_external:
        raise ValueError("MediaRef is not external reference")

    if not media_ref.path:
        raise ValueError("No path found in MediaRef")

    if media_ref.pts_ns is None:
        raise ValueError("No timestamp found for video MediaRef")

    path = media_ref.path
    pts_ns = media_ref.pts_ns

    # Validate file exists for local files only
    if not path.startswith(("http://", "https://")):
        media_path = Path(path)
        if not media_path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

    return _load_video_frame(path, pts_ns, force_close)


def _load_video_frame(path: str, pts_ns: int, force_close: bool) -> np.ndarray:
    """Load a specific frame from video."""
    pts_fraction = Fraction(pts_ns, TimeUnits.SECOND)

    try:
        with VideoReader(path, force_close=force_close) as reader:
            frame = reader.read_frame(pts=pts_fraction)
            rgb_array = frame.to_ndarray(format="rgb24")
            return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)
    except Exception as e:
        source_type = "remote" if path.startswith(("http://", "https://")) else "local"
        pts_seconds = pts_ns / 1_000_000_000
        raise ValueError(f"Failed to load frame at {pts_seconds:.3f}s from {source_type} video {path}: {e}") from e


def _load_static_image(path: str) -> np.ndarray:
    """Load a static image file."""
    try:
        pil_image = load_image(path)
        rgb_array = np.array(pil_image)
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)
    except Exception as e:
        source_type = "remote" if path.startswith(("http://", "https://")) else "local"
        raise ValueError(f"Failed to load {source_type} image from {path}: {e}") from e


def _get_media_info(media_ref: Optional[MediaRef]) -> dict:
    """Get information about the media reference."""
    if media_ref is None:
        return {"type": None}

    if media_ref.is_embedded:
        if not media_ref.data:
            return {"type": "embedded", "error": "No data found"}

        return {
            "type": "embedded",
            "format": media_ref.format,
            "size_bytes": len(base64.b64decode(media_ref.data)),
        }

    if media_ref.is_external:
        if not media_ref.path:
            return {"type": "external", "error": "No path found"}

        is_remote = media_ref.path.startswith(("http://", "https://"))
        info = {
            "type": "external_video" if media_ref.is_video else "external_image",
            "path": media_ref.path,
            "is_local": not is_remote,
            "is_remote": is_remote,
            "media_type": "video" if media_ref.is_video else "image",
        }

        if media_ref.is_video and media_ref.pts_ns is not None:
            info.update(
                {
                    "pts_ns": media_ref.pts_ns,
                    "pts_seconds": media_ref.pts_ns / 1_000_000_000,
                }
            )

        return info

    return {"type": "unknown"}


def _format_media_display(media_ref: MediaRef) -> str:
    """Format media reference for display in string representation."""
    if media_ref.is_embedded:
        if not media_ref.data or not media_ref.format:
            return "embedded_unknown"
        size_kb = len(base64.b64decode(media_ref.data)) / 1024
        return f"embedded_{media_ref.format}({size_kb:.1f}KB)"

    if media_ref.is_external:
        if not media_ref.path:
            return "external_unknown"

        is_remote = media_ref.path.startswith(("http://", "https://"))
        path_display = media_ref.path if is_remote else Path(media_ref.path).name
        prefix = "remote" if is_remote else "local"
        media_type = "video" if media_ref.is_video else "image"

        if media_ref.is_video and media_ref.pts_ns is not None:
            pts_seconds = media_ref.pts_ns / 1_000_000_000
            return f"{prefix}_{media_type}({path_display}@{pts_seconds:.3f}s)"
        else:
            return f"{prefix}_{media_type}({path_display})"

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
        """Validate frame data and set shape information."""
        # Require either frame_arr or media_ref
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
                "Use embed_from_array() to create a media reference first."
            )
        return super().model_dump_json(**kwargs)

    # Reference type checking methods
    def has_embedded_data(self) -> bool:
        """Check if this frame has embedded data."""
        return self.media_ref is not None and self.media_ref.is_embedded

    def has_external_reference(self) -> bool:
        """Check if this frame has any external media reference (image or video)."""
        return self.media_ref is not None and self.media_ref.is_external

    def has_external_image_reference(self) -> bool:
        """Check if this frame has external image reference."""
        return self.media_ref is not None and self.media_ref.is_external and not self.media_ref.is_video

    def has_external_video_reference(self) -> bool:
        """Check if this frame has external video reference."""
        return self.media_ref is not None and self.media_ref.is_external and self.media_ref.is_video

    def is_loaded(self) -> bool:
        """Check if the frame data is currently loaded in memory."""
        return self.frame_arr is not None

    # Media reference creation methods
    def embed_from_array(self, format: Literal["png", "jpeg"] = "png", *, quality: Optional[int] = None) -> Self:
        """Compress and embed the current frame_arr data."""
        if self.frame_arr is None:
            raise ValueError("No frame_arr available to embed")

        self.media_ref = _compress_frame_to_embedded(self.frame_arr, format, quality)
        return self

    def resolve_external_path(self, mcap_path: Union[str, Path]) -> Self:
        """
        Resolve relative external path using MCAP file location.

        This method is needed during data read operations when external references
        contain relative paths that need to be resolved relative to the MCAP file.
        Absolute paths and non-external references are left unchanged.

        Args:
            mcap_path: Path to the MCAP file used as base for relative path resolution

        Returns:
            Self for method chaining
        """
        if self.media_ref is None or not self.has_external_reference():
            return self

        current_path = self.media_ref.path

        if not current_path:
            return self

        # Skip if path is already absolute or is a URL
        if Path(current_path).is_absolute() or current_path.startswith(("http://", "https://")):
            return self

        # Resolve relative path relative to MCAP directory
        mcap_dir = Path(mcap_path).parent
        resolved_path = mcap_dir / current_path

        # Update the path in the media reference
        self.media_ref = MediaRef.from_path(str(resolved_path), pts_ns=self.media_ref.pts_ns)

        return self

    @classmethod
    def from_external_image(
        cls,
        path: Union[str, Path],
        *,
        mcap_path: Optional[Union[str, Path]] = None,
        utc_ns: Optional[int] = None,
        source_shape: Optional[Tuple[int, int]] = None,
        shape: Optional[Tuple[int, int]] = None,
    ) -> "ScreenCaptured":
        """
        Create ScreenCaptured instance with external image reference.

        The path can be absolute or relative to the MCAP file. When saving to MCAP,
        paths are stored as provided (absolute or relative to MCAP). During data read,
        relative paths need to be resolved using resolve_external_path().

        Args:
            path: Path to the image file (absolute or relative to MCAP)
            mcap_path: Optional MCAP file path for immediate relative path resolution
            utc_ns: UTC timestamp in nanoseconds
            source_shape: Original source dimensions (width, height)
            shape: Current frame dimensions (width, height)

        Returns:
            ScreenCaptured instance with external image reference
        """
        path_str = str(path)

        # If mcap_path is provided and path is relative, resolve it immediately
        if mcap_path is not None and not Path(path_str).is_absolute():
            mcap_dir = Path(mcap_path).parent
            resolved_path = mcap_dir / path_str
            path_str = str(resolved_path)

        return cls(
            utc_ns=utc_ns,
            source_shape=source_shape,
            shape=shape,
            media_ref=MediaRef.from_path(path=path_str),
        )

    @classmethod
    def from_external_video(
        cls,
        path: Union[str, Path],
        pts_ns: int,
        *,
        mcap_path: Optional[Union[str, Path]] = None,
        utc_ns: Optional[int] = None,
        source_shape: Optional[Tuple[int, int]] = None,
        shape: Optional[Tuple[int, int]] = None,
    ) -> "ScreenCaptured":
        """
        Create ScreenCaptured instance with external video reference.

        The path can be absolute or relative to the MCAP file. When saving to MCAP,
        paths are stored as provided (absolute or relative to MCAP). During data read,
        relative paths need to be resolved using resolve_external_path().

        Args:
            path: Path to the video file (absolute or relative to MCAP)
            pts_ns: Timestamp in nanoseconds for the specific frame
            mcap_path: Optional MCAP file path for immediate relative path resolution
            utc_ns: UTC timestamp in nanoseconds
            source_shape: Original source dimensions (width, height)
            shape: Current frame dimensions (width, height)

        Returns:
            ScreenCaptured instance with external video reference
        """
        path_str = str(path)

        # If mcap_path is provided and path is relative, resolve it immediately
        if mcap_path is not None and not Path(path_str).is_absolute():
            mcap_dir = Path(mcap_path).parent
            resolved_path = mcap_dir / path_str
            path_str = str(resolved_path)

        return cls(
            utc_ns=utc_ns,
            source_shape=source_shape,
            shape=shape,
            media_ref=MediaRef.from_path(path=path_str, pts_ns=pts_ns),
        )

    # Frame loading and conversion methods
    def lazy_load(self, *, force_close: bool = False) -> np.ndarray:
        """Lazy load the frame data from any available source."""
        if self.frame_arr is not None:
            return self.frame_arr

        if self.media_ref is None:
            raise ValueError("No frame data sources available for loading")

        # Load based on reference type
        if self.media_ref.is_embedded:
            self.frame_arr = _load_from_embedded(self.media_ref)
        elif self.media_ref.is_external and not self.media_ref.is_video:
            self.frame_arr = _load_from_external_image(self.media_ref)
        elif self.media_ref.is_external and self.media_ref.is_video:
            self.frame_arr = _load_from_external_video(self.media_ref, force_close=force_close)
        else:
            raise ValueError(f"Unsupported media reference: {self.media_ref.scheme}")

        # Update shape information
        h, w = self.frame_arr.shape[:2]
        self.shape = (w, h)
        if self.source_shape is None:
            self.source_shape = self.shape

        return self.frame_arr

    def to_rgb_array(self) -> np.ndarray:
        """Return the frame as an RGB numpy array."""
        bgra_array = self.lazy_load()
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
        attr_strs = []

        # Add core attributes
        for attr in ["utc_ns", "source_shape", "shape"]:
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
