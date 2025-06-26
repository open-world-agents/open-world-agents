"""
Minimal tests for screen capture message with new clean API.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from owa.core.io.video import VideoWriter, force_close_video_container
from owa.core.time import TimeUnits
from owa.msgs.desktop.screen import MediaRef, ScreenCaptured


@pytest.fixture
def sample_bgra_frame():
    """Create a sample BGRA frame for testing."""
    # Create a 64x48 BGRA frame with gradient pattern
    height, width = 48, 64
    frame = np.zeros((height, width, 4), dtype=np.uint8)

    # Create gradient pattern for easy identification
    for y in range(height):
        for x in range(width):
            frame[y, x] = [x * 4, y * 5, (x + y) * 2, 255]  # BGRA

    return frame


@pytest.fixture
def sample_video_file():
    """Create a temporary video file with known frames for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / "test_video.mp4"

        # Create test video with 5 frames at different timestamps
        timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]  # 5 frames at 100ms intervals

        with VideoWriter(video_path, fps=10.0, vfr=True) as writer:
            for i, timestamp in enumerate(timestamps):
                # Create distinct frames with different colors
                frame = np.full((48, 64, 3), i * 50, dtype=np.uint8)  # RGB
                writer.write_frame(frame, pts=timestamp, pts_unit="sec")

            # Add a final frame to ensure the last intended frame has duration
            final_timestamp = timestamps[-1] + 0.1  # 100ms after last frame
            final_frame = np.zeros((48, 64, 3), dtype=np.uint8)  # Black frame as end marker
            writer.write_frame(final_frame, pts=final_timestamp, pts_unit="sec")

        yield video_path, timestamps

        force_close_video_container(video_path)


class TestMediaRef:
    """Test MediaRef minimal design."""

    def test_create_embedded_ref(self):
        """Test creating MediaRef with embedded data URI."""
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        ref = MediaRef(uri=data_uri)

        assert ref.is_embedded == True
        assert ref.is_video == False

    def test_create_video_ref(self):
        """Test creating MediaRef for video."""
        ref = MediaRef(uri="test.mp4", pts_ns=1000000000)

        assert ref.is_embedded == False
        assert ref.is_video == True
        assert ref.pts_ns == 1000000000

    def test_create_external_image_ref(self):
        """Test creating MediaRef for external image."""
        ref = MediaRef(uri="image.png")

        assert ref.is_embedded == False
        assert ref.is_video == False

    def test_create_external_url_ref(self):
        """Test creating MediaRef for remote URL."""
        ref = MediaRef(uri="https://example.com/image.jpg")

        assert ref.is_embedded == False
        assert ref.is_video == False


class TestScreenCaptured:
    """Test ScreenCaptured minimal design."""

    def test_create_with_frame_arr(self, sample_bgra_frame):
        """Test creating ScreenCaptured with direct frame array."""
        utc_ns = 1741608540328534500

        screen_msg = ScreenCaptured(utc_ns=utc_ns, frame_arr=sample_bgra_frame)

        assert screen_msg.utc_ns == utc_ns
        assert np.array_equal(screen_msg.frame_arr, sample_bgra_frame)
        assert screen_msg.shape == (64, 48)  # (width, height)
        assert screen_msg.media_ref is None  # No media reference initially

    def test_load_frame_array_with_existing_frame(self, sample_bgra_frame):
        """Test that load_frame_array returns existing frame when frame_arr is already set."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        loaded_frame = screen_msg.load_frame_array()
        assert np.array_equal(loaded_frame, sample_bgra_frame)
        assert loaded_frame is screen_msg.frame_arr  # Should return same object

    def test_to_rgb_array(self, sample_bgra_frame):
        """Test conversion from BGRA to RGB array."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        rgb_array = screen_msg.to_rgb_array()

        # Verify shape and conversion
        assert rgb_array.shape == (48, 64, 3)  # RGB has 3 channels
        assert rgb_array.dtype == np.uint8

        # Verify color conversion (BGRA -> RGB)
        expected_rgb = cv2.cvtColor(sample_bgra_frame, cv2.COLOR_BGRA2RGB)
        assert np.array_equal(rgb_array, expected_rgb)

    def test_to_pil_image(self, sample_bgra_frame):
        """Test conversion to PIL Image."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        pil_image = screen_msg.to_pil_image()

        assert isinstance(pil_image, Image.Image)
        assert pil_image.mode == "RGB"
        assert pil_image.size == (64, 48)  # PIL size is (width, height)

        # Verify content matches RGB conversion
        rgb_array = screen_msg.to_rgb_array()
        pil_array = np.array(pil_image)
        assert np.array_equal(pil_array, rgb_array)

    def test_embed_as_data_uri_png(self, sample_bgra_frame):
        """Test embedding frame data as PNG."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        # Initially no embedded data
        assert screen_msg.media_ref is None

        # Embed the frame
        screen_msg.embed_as_data_uri(format="png")

        # Now should have embedded data
        assert screen_msg.media_ref is not None
        assert screen_msg.media_ref.is_embedded == True
        assert "data:image/png;base64," in screen_msg.media_ref.uri

    def test_embed_as_data_uri_jpeg(self, sample_bgra_frame):
        """Test embedding frame data as JPEG with quality setting."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        # Embed as JPEG with specific quality
        screen_msg.embed_as_data_uri(format="jpeg", quality=95)

        assert screen_msg.media_ref is not None
        assert screen_msg.media_ref.is_embedded == True
        assert "data:image/jpeg;base64," in screen_msg.media_ref.uri

    def test_embedded_roundtrip(self, sample_bgra_frame):
        """Test embedding and loading back gives similar results."""
        # Original message
        original_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        # Embed as PNG
        original_msg.embed_as_data_uri(format="png")

        # Create new message from embedded data
        embedded_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=original_msg.media_ref)

        # Load back
        loaded_frame = embedded_msg.load_frame_array()

        # Should have same shape and similar content (allowing for compression)
        assert loaded_frame.shape == sample_bgra_frame.shape
        assert loaded_frame.dtype == sample_bgra_frame.dtype

    def test_create_with_embedded_ref(self, sample_bgra_frame):
        """Test creating ScreenCaptured with embedded reference."""
        # First create an embedded reference
        screen_msg_temp = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)
        screen_msg_temp.embed_as_data_uri(format="png")
        embedded_ref = screen_msg_temp.media_ref

        # Create new message with embedded reference
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=embedded_ref)

        assert screen_msg.utc_ns == 1741608540328534500
        assert screen_msg.frame_arr is None  # Should not be loaded yet
        assert screen_msg.media_ref.is_embedded == True

    def test_load_from_embedded(self, sample_bgra_frame):
        """Test loading from embedded data."""
        # Create embedded reference
        screen_msg_temp = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)
        screen_msg_temp.embed_as_data_uri(format="png")

        # Create new message with just embedded data
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=screen_msg_temp.media_ref)

        # Initially no frame loaded
        assert screen_msg.frame_arr is None

        # Load should work
        loaded_frame = screen_msg.load_frame_array()

        assert loaded_frame is not None
        assert screen_msg.frame_arr is not None
        assert loaded_frame.shape[2] == 4  # BGRA format
        assert screen_msg.shape is not None

    def test_create_with_external_video_ref(self, sample_video_file):
        """Test creating ScreenCaptured with external video reference."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[2] * TimeUnits.SECOND)  # Third frame (0.2s)

        media_ref = MediaRef(uri=str(video_path), pts_ns=pts_ns)
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        assert screen_msg.utc_ns == 1741608540328534500
        assert screen_msg.frame_arr is None  # Should not be loaded yet
        assert screen_msg.media_ref.is_video == True
        assert screen_msg.shape is None  # Not set until loading

    def test_load_from_video(self, sample_video_file):
        """Test loading from external video file."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[1] * TimeUnits.SECOND)  # Second frame (0.1s)

        media_ref = MediaRef(uri=str(video_path), pts_ns=pts_ns)
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        # Initially, frame should not be loaded
        assert screen_msg.frame_arr is None
        assert screen_msg.shape is None

        # Trigger loading
        loaded_frame = screen_msg.load_frame_array()

        # After loading, frame should be available
        assert loaded_frame is not None
        assert screen_msg.frame_arr is not None
        assert np.array_equal(loaded_frame, screen_msg.frame_arr)
        assert loaded_frame.shape[2] == 4  # BGRA format
        assert screen_msg.shape is not None
        assert screen_msg.source_shape is not None

    def test_validation_requires_frame_or_media_ref(self):
        """Test that either frame_arr or media_ref is required."""
        with pytest.raises(ValueError, match="Either frame_arr or media_ref must be provided"):
            ScreenCaptured(utc_ns=1741608540328534500)

    def test_embed_without_frame_arr(self):
        """Test that embed_as_data_uri requires frame_arr."""
        media_ref = MediaRef(uri="test.mp4", pts_ns=1000000000)
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        with pytest.raises(ValueError, match="No frame_arr available to embed"):
            screen_msg.embed_as_data_uri()

    def test_json_serialization_without_media_ref(self, sample_bgra_frame):
        """Test that JSON serialization requires media_ref."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        with pytest.raises(ValueError, match="Cannot serialize without media_ref"):
            screen_msg.model_dump_json()

    def test_string_representation(self, sample_bgra_frame):
        """Test string representation."""
        # Test with frame_arr only
        screen_msg1 = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)
        repr_str1 = str(screen_msg1)
        assert "ScreenCaptured" in repr_str1
        assert "utc_ns=1741608540328534500" in repr_str1
        assert "shape=(64, 48)" in repr_str1

        # Test with embedded ref
        screen_msg1.embed_as_data_uri(format="png")
        repr_str2 = str(screen_msg1)
        assert "embedded" in repr_str2

        # Test with video ref
        media_ref = MediaRef(uri="test.mp4", pts_ns=2000000000)
        screen_msg2 = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)
        repr_str3 = str(screen_msg2)
        assert "video@2000000000ns" in repr_str3
