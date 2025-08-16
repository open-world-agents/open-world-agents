import base64
import os
from fractions import Fraction
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import PIL.Image

from ..time import TimeUnits
from .image import load_image
from .video import VideoReader

VIDEO_DECODING_SERVER_URL = os.environ.get("VIDEO_DECODING_SERVER_URL")  # e.g. 127.0.0.1:8000

# ============================================================================
# Core Image/Video Loading Functions
# ============================================================================


def load_image_as_bgra(path_or_uri: str) -> np.ndarray:
    """
    Load image from any source and return as BGRA numpy array.

    Args:
        path_or_uri: File path, URL, or data URI

    Returns:
        BGRA numpy array

    Raises:
        ValueError: If loading fails
        FileNotFoundError: If local file doesn't exist
    """
    try:
        if path_or_uri.startswith("data:"):
            return _load_from_data_uri(path_or_uri)
        else:
            # Use existing load_image function and convert to BGRA
            pil_image = load_image(path_or_uri)
            return _pil_to_bgra_array(pil_image)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load image from {path_or_uri}: {e}") from e


def load_video_frame_as_bgra(path_or_url: str, pts_ns: int, *, keep_av_open: bool = False) -> np.ndarray:
    """
    Load video frame and return as BGRA numpy array.

    Args:
        path_or_url: File path or URL to video
        pts_ns: Presentation timestamp in nanoseconds
        keep_av_open: Keep AV container open in cache instead of forcing closure

    Returns:
        BGRA numpy array

    Raises:
        ValueError: If loading fails
        FileNotFoundError: If local file doesn't exist
    """
    try:
        # Validate local file exists
        if not path_or_url.startswith(("http://", "https://")):
            if not Path(path_or_url).exists():
                raise FileNotFoundError(f"Video file not found: {path_or_url}")

        # Use Triton decoding server if available
        if VIDEO_DECODING_SERVER_URL:
            rgb_array = extract_frame(path_or_url, pts_ns / TimeUnits.SECOND, server_url=VIDEO_DECODING_SERVER_URL)
            return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)

        # Use VideoReader
        pts_fraction = Fraction(pts_ns, TimeUnits.SECOND)

        with VideoReader(path_or_url, keep_av_open=keep_av_open) as reader:
            frame = reader.read_frame(pts=pts_fraction)
            rgb_array = frame.to_ndarray(format="rgb24")
            return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)
    except FileNotFoundError:
        raise
    except Exception as e:
        pts_seconds = pts_ns / 1_000_000_000
        raise ValueError(f"Failed to load frame at {pts_seconds:.3f}s from {path_or_url}: {e}") from e


# ============================================================================
# Triton Decoding Server Support
# ============================================================================


def extract_frame(video_path: str, time_sec: float, server_url: str = "127.0.0.1:8000") -> np.ndarray:
    """
    Extract a frame from video at specified time.

    Args:
        video_path: Path to video file
        time_sec: Time in seconds
        server_url: Triton server URL

    Returns:
        Frame as numpy array (H, W, 3)
    """
    import tritonclient.http as httpclient

    client = httpclient.InferenceServerClient(url=server_url)

    inputs = [httpclient.InferInput("video_path", [1], "BYTES"), httpclient.InferInput("time_sec", [1], "FP32")]
    inputs[0].set_data_from_numpy(np.array([str(video_path).encode()], dtype=np.object_))
    inputs[1].set_data_from_numpy(np.array([time_sec], dtype=np.float32))

    outputs = [httpclient.InferRequestedOutput("frame")]
    response = client.infer("video_decoder", inputs=inputs, outputs=outputs)

    frame = response.as_numpy("frame")
    if frame is None:
        raise RuntimeError("Failed to extract frame from server response")

    return frame


# ============================================================================
# Format Conversion Functions
# ============================================================================


def encode_to_base64(array: np.ndarray, format: Literal["png", "jpeg", "bmp"], quality: Optional[int] = None) -> str:
    """
    Encode BGRA numpy array to base64 string.

    Args:
        array: BGRA numpy array
        format: Output format ('png', 'jpeg', or 'bmp')
        quality: JPEG quality (1-100), ignored for PNG and BMP

    Returns:
        Base64 encoded string
    """
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
    elif format == "bmp":
        success, encoded = cv2.imencode(".bmp", bgr_array)
    else:
        raise ValueError(f"Unsupported format: {format}")

    if not success:
        raise ValueError(f"Failed to encode image as {format}")

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def decode_from_base64(data: str) -> np.ndarray:
    """
    Decode base64 string to BGRA numpy array.

    Args:
        data: Base64 encoded image data

    Returns:
        BGRA numpy array
    """
    try:
        image_bytes = base64.b64decode(data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if bgr_array is None:
            raise ValueError("Failed to decode base64 image data")

        return cv2.cvtColor(bgr_array, cv2.COLOR_BGR2BGRA)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 data: {e}") from e


def bgra_array_to_pil(array: np.ndarray) -> PIL.Image.Image:
    """Convert BGRA numpy array to PIL image."""
    rgb_array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
    return PIL.Image.fromarray(rgb_array)


# ============================================================================
# Utility Functions
# ============================================================================


def validate_media_path(path: str) -> bool:
    """
    Check if media path is accessible.

    Args:
        path: File path or URL

    Returns:
        True if accessible, False otherwise
    """
    try:
        if path.startswith(("http://", "https://")):
            # Quick HEAD request to check if URL is accessible
            import requests

            response = requests.head(path, timeout=5)
            return response.status_code == 200
        else:
            return Path(path).exists()
    except Exception:
        return False


# ============================================================================
# Helper Functions
# ============================================================================


def _load_from_data_uri(data_uri: str) -> np.ndarray:
    """Load image from data URI."""
    parsed = urlparse(data_uri)
    if parsed.scheme != "data":
        raise ValueError(f"Invalid data URI scheme: {parsed.scheme}")

    try:
        # Extract base64 data from data URI
        data_part = parsed.path.split(",", 1)[1]
        return decode_from_base64(data_part)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid data URI format: {e}") from e


def _pil_to_bgra_array(pil_image: PIL.Image.Image) -> np.ndarray:
    """Convert PIL image to BGRA numpy array."""
    # Ensure image is in RGB mode
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Convert to numpy array and then to BGRA
    rgb_array = np.array(pil_image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)
