"""
Unified media loading utilities for Open World Agents.

This module provides high-level functions and classes for loading images and video frames
from various sources (embedded base64, local files, remote URLs) and converting
between different formats (PIL, numpy, base64).

The module is designed with the following principles:
- Source-agnostic loading (embedded, local, remote)
- Format-aware processing (images vs videos)
- Consistent output format (BGRA numpy arrays)
- Composable operations (small, focused functions)
- Error context preservation (maintain source information)
"""

import base64
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal, Optional, Union
from urllib.parse import urlparse

import cv2
import numpy as np
import PIL.Image
import PIL.ImageOps

from ..time import TimeUnits
from .image import load_image
from .video import VideoReader


@dataclass
class MediaSource:
    """Represents a media source with metadata."""

    uri: str
    source_type: Literal["embedded", "local", "remote"]
    media_type: Literal["image", "video"]
    format: Optional[str] = None

    @classmethod
    def from_mediaref(cls, media_ref) -> "MediaSource":
        """Create MediaSource from MediaRef."""
        if media_ref.is_embedded:
            return cls(
                uri=media_ref.uri,
                source_type="embedded",
                media_type="video" if media_ref.is_video else "image",
                format=media_ref.format,
            )
        else:
            # Determine if remote or local
            if media_ref.path and media_ref.path.startswith(("http://", "https://")):
                source_type = "remote"
            else:
                source_type = "local"

            return cls(
                uri=media_ref.uri,
                source_type=source_type,
                media_type="video" if media_ref.is_video else "image",
                format=media_ref.format,
            )

    @classmethod
    def from_path(cls, path: str) -> "MediaSource":
        """Create MediaSource from file path or URL."""
        # Check if it's a URL with scheme
        if "://" in path:
            if path.startswith(("http://", "https://")):
                source_type = "remote"
            else:
                # Other schemes like ftp:// are treated as local for validation purposes
                # This will cause FileNotFoundError which is the expected behavior
                source_type = "local"
        else:
            source_type = "local"

        # Simple format detection from extension
        path_lower = path.lower()
        if any(path_lower.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]):
            media_type = "video"
        else:
            media_type = "image"

        return cls(
            uri=path,
            source_type=source_type,
            media_type=media_type,
            format=None,  # Will be detected during loading
        )

    @classmethod
    def from_embedded(cls, data: str, format: str) -> "MediaSource":
        """Create MediaSource from embedded data."""
        mime_type = f"image/{format}"
        uri = f"data:{mime_type};base64,{data}"

        return cls(
            uri=uri,
            source_type="embedded",
            media_type="image",  # Embedded videos not supported yet
            format=format,
        )


class FormatConverter:
    """Handle conversions between different image formats."""

    @staticmethod
    def pil_to_bgra_array(pil_image: PIL.Image.Image) -> np.ndarray:
        """Convert PIL image to BGRA numpy array."""
        # Ensure image is in RGB mode
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert to numpy array and then to BGRA
        rgb_array = np.array(pil_image)
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)

    @staticmethod
    def bgra_array_to_pil(array: np.ndarray) -> PIL.Image.Image:
        """Convert BGRA numpy array to PIL image."""
        # Convert BGRA to RGB
        rgb_array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
        return PIL.Image.fromarray(rgb_array)

    @staticmethod
    def encode_to_base64(array: np.ndarray, format: Literal["png", "jpeg"], quality: Optional[int] = None) -> str:
        """Encode numpy array to base64 string."""
        # Convert BGRA to BGR for cv2 encoding
        bgr_array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)

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

        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    @staticmethod
    def decode_from_base64(data: str, format: str) -> np.ndarray:
        """Decode base64 string to numpy array."""
        try:
            image_bytes = base64.b64decode(data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            bgr_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if bgr_array is None:
                raise ValueError(f"Failed to decode embedded {format} data")

            return cv2.cvtColor(bgr_array, cv2.COLOR_BGR2BGRA)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 {format} data: {e}") from e


class MediaValidator:
    """Validate media sources and provide metadata."""

    @staticmethod
    def validate_source(source: MediaSource) -> bool:
        """Check if media source is accessible."""
        try:
            if source.source_type == "embedded":
                # Try to decode the embedded data
                parsed = urlparse(source.uri)
                if parsed.scheme == "data":
                    data_part = parsed.path.split(",", 1)[1]
                    FormatConverter.decode_from_base64(data_part, source.format or "png")
                return True
            elif source.source_type == "local":
                return Path(source.uri).exists()
            elif source.source_type == "remote":
                # Quick HEAD request to check if URL is accessible
                import requests

                response = requests.head(source.uri, timeout=5)
                return response.status_code == 200
        except Exception:
            return False

        return False

    @staticmethod
    def get_metadata(source: MediaSource) -> dict:
        """Get media metadata (dimensions, format, size, etc.)."""
        metadata = {
            "source_type": source.source_type,
            "media_type": source.media_type,
            "format": source.format,
            "uri": source.uri,
        }

        try:
            if source.source_type == "embedded":
                parsed = urlparse(source.uri)
                if parsed.scheme == "data":
                    data_part = parsed.path.split(",", 1)[1]
                    metadata["size_bytes"] = len(base64.b64decode(data_part))
            elif source.source_type == "local":
                path = Path(source.uri)
                if path.exists():
                    metadata["size_bytes"] = path.stat().st_size
                    metadata["exists"] = True
                else:
                    metadata["exists"] = False
            elif source.source_type == "remote":
                metadata["is_remote"] = True

        except Exception as e:
            metadata["error"] = str(e)

        return metadata

    @staticmethod
    def is_supported_format(path: str) -> bool:
        """Check if file format is supported."""
        path_lower = path.lower()

        # Supported image formats
        image_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        # Supported video formats
        video_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]

        return any(path_lower.endswith(ext) for ext in image_formats + video_formats)


class MediaLoader:
    """Unified media loader supporting multiple sources and formats."""

    def __init__(self, timeout: float = 60.0, cache_enabled: bool = True):
        """
        Initialize MediaLoader.

        Args:
            timeout: Timeout for network operations in seconds
            cache_enabled: Whether to enable caching (for video containers)
        """
        self.timeout = timeout
        self.cache_enabled = cache_enabled

    def load_image(self, source: MediaSource) -> np.ndarray:
        """
        Load image from any source, return BGRA array.

        Args:
            source: MediaSource describing the image location

        Returns:
            BGRA numpy array

        Raises:
            ValueError: If source is invalid or loading fails
            FileNotFoundError: If local file doesn't exist
        """
        if source.media_type != "image":
            raise ValueError(f"Expected image source, got {source.media_type}")

        try:
            if source.source_type == "embedded":
                return self._load_from_embedded(source)
            elif source.source_type in ("local", "remote"):
                return self._load_from_path(source)
            else:
                raise ValueError(f"Unsupported source type: {source.source_type}")
        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is for proper test handling
            raise
        except Exception as e:
            source_desc = f"{source.source_type} image"
            raise ValueError(f"Failed to load {source_desc} from {source.uri}: {e}") from e

    def load_video_frame(self, source: MediaSource, pts_ns: int, force_close: bool = False) -> np.ndarray:
        """
        Load video frame from any source, return BGRA array.

        Args:
            source: MediaSource describing the video location
            pts_ns: Presentation timestamp in nanoseconds
            force_close: Force complete closure instead of using cache

        Returns:
            BGRA numpy array

        Raises:
            ValueError: If source is invalid or loading fails
            FileNotFoundError: If local file doesn't exist
        """
        if source.media_type != "video":
            raise ValueError(f"Expected video source, got {source.media_type}")

        if source.source_type == "embedded":
            raise ValueError("Embedded video sources are not supported")

        try:
            return self._load_video_frame_from_path(source, pts_ns, force_close)
        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is for proper test handling
            raise
        except Exception as e:
            source_desc = f"{source.source_type} video"
            pts_seconds = pts_ns / 1_000_000_000
            raise ValueError(f"Failed to load frame at {pts_seconds:.3f}s from {source_desc} {source.uri}: {e}") from e

    def _load_from_embedded(self, source: MediaSource) -> np.ndarray:
        """Load image from embedded base64 data."""
        parsed = urlparse(source.uri)
        if parsed.scheme != "data":
            raise ValueError(f"Invalid embedded URI scheme: {parsed.scheme}")

        try:
            # Extract base64 data from data URI
            data_part = parsed.path.split(",", 1)[1]
            return FormatConverter.decode_from_base64(data_part, source.format or "png")
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid embedded data format: {e}") from e

    def _load_from_path(self, source: MediaSource) -> np.ndarray:
        """Load image from local file or remote URL."""
        path = source.uri

        # Validate local file exists
        if source.source_type == "local":
            if not Path(path).exists():
                raise FileNotFoundError(f"Image file not found: {path}")

        # Use existing load_image function and convert to BGRA
        pil_image = load_image(path)
        return FormatConverter.pil_to_bgra_array(pil_image)

    def _load_video_frame_from_path(self, source: MediaSource, pts_ns: int, force_close: bool) -> np.ndarray:
        """Load specific frame from video file or URL."""
        path = source.uri

        # Validate local file exists
        if source.source_type == "local":
            if not Path(path).exists():
                raise FileNotFoundError(f"Video file not found: {path}")

        # Load frame using VideoReader
        pts_fraction = Fraction(pts_ns, TimeUnits.SECOND)

        with VideoReader(path, force_close=force_close) as reader:
            frame = reader.read_frame(pts=pts_fraction)
            rgb_array = frame.to_ndarray(format="rgb24")
            return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)


# ============================================================================
# High-Level Factory Functions
# ============================================================================

# Global default loader instance
default_loader = MediaLoader()


def load_media_as_array(source: Union[str, MediaSource], pts_ns: Optional[int] = None, **kwargs) -> np.ndarray:
    """
    High-level function to load any media as BGRA array.

    Args:
        source: File path, URL, or MediaSource object
        pts_ns: Presentation timestamp for video (required for video sources)
        **kwargs: Additional arguments passed to loader methods

    Returns:
        BGRA numpy array

    Examples:
        # Load image
        array = load_media_as_array("image.jpg")
        array = load_media_as_array("https://example.com/image.png")

        # Load video frame
        array = load_media_as_array("video.mp4", pts_ns=1000000000)  # 1 second
        array = load_media_as_array("https://example.com/video.mp4", pts_ns=500000000)  # 0.5 seconds
    """
    if isinstance(source, str):
        source = MediaSource.from_path(source)

    if source.media_type == "image":
        return default_loader.load_image(source)
    elif source.media_type == "video":
        if pts_ns is None:
            raise ValueError("pts_ns is required for video sources")
        force_close = kwargs.get("force_close", False)
        return default_loader.load_video_frame(source, pts_ns, force_close)
    else:
        raise ValueError(f"Unsupported media type: {source.media_type}")


def create_embedded_from_file(file_path: str, format: Literal["png", "jpeg"] = "png") -> str:
    """
    Load file and return base64 encoded data.

    Args:
        file_path: Path to local image file
        format: Target format for encoding

    Returns:
        Base64 encoded string

    Example:
        base64_data = create_embedded_from_file("image.jpg", format="png")
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    source = MediaSource.from_path(file_path)
    array = default_loader.load_image(source)
    return FormatConverter.encode_to_base64(array, format)


def create_embedded_from_url(url: str, format: Literal["png", "jpeg"] = "png") -> str:
    """
    Download and return base64 encoded data.

    Args:
        url: Remote image URL
        format: Target format for encoding

    Returns:
        Base64 encoded string

    Example:
        base64_data = create_embedded_from_url("https://example.com/image.jpg", format="png")
    """
    source = MediaSource.from_path(url)
    array = default_loader.load_image(source)
    return FormatConverter.encode_to_base64(array, format)


def validate_media_accessibility(source: Union[str, MediaSource]) -> bool:
    """
    Quick check if media is accessible.

    Args:
        source: File path, URL, or MediaSource object

    Returns:
        True if accessible, False otherwise

    Example:
        if validate_media_accessibility("https://example.com/image.jpg"):
            # Safe to load
            array = load_media_as_array("https://example.com/image.jpg")
    """
    if isinstance(source, str):
        source = MediaSource.from_path(source)

    return MediaValidator.validate_source(source)


def get_media_metadata(source: Union[str, MediaSource]) -> dict:
    """
    Get metadata about media source.

    Args:
        source: File path, URL, or MediaSource object

    Returns:
        Dictionary with metadata information

    Example:
        metadata = get_media_metadata("image.jpg")
        print(f"Size: {metadata.get('size_bytes')} bytes")
    """
    if isinstance(source, str):
        source = MediaSource.from_path(source)

    return MediaValidator.get_metadata(source)
