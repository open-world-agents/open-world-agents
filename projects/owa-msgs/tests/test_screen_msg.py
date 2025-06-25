import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from owa.core.io.video import VideoWriter, force_close_video_container
from owa.core.time import TimeUnits
from owa.msgs.desktop.screen import EmbeddedRef, ExternalImageRef, ExternalVideoRef, ScreenCaptured


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


class TestEmbeddedRef:
    """Test EmbeddedRef creation and operations."""

    def test_create_embedded_ref(self):
        """Test creating EmbeddedRef directly."""
        data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        ref = EmbeddedRef(format="png", data=data)

        assert ref.type == "embedded"
        assert ref.format == "png"
        assert ref.data == data

    def test_create_embedded_ref_jpeg(self):
        """Test creating EmbeddedRef with JPEG format."""
        data = "base64encodeddata"
        ref = EmbeddedRef(format="jpeg", data=data)

        assert ref.type == "embedded"
        assert ref.format == "jpeg"
        assert ref.data == data


class TestExternalImageRef:
    """Test ExternalImageRef creation and operations."""

    def test_create_external_image_ref(self):
        """Test creating ExternalImageRef for static image."""
        ref = ExternalImageRef(path="image.png")

        assert ref.type == "external_image"
        assert ref.path == "image.png"

    def test_create_external_image_ref_remote_url(self):
        """Test creating ExternalImageRef for remote URL."""
        ref = ExternalImageRef(path="https://example.com/image.jpg")

        assert ref.type == "external_image"
        assert ref.path == "https://example.com/image.jpg"


class TestExternalVideoRef:
    """Test ExternalVideoRef creation and operations."""

    def test_create_external_video_ref(self):
        """Test creating ExternalVideoRef for video with timestamp."""
        ref = ExternalVideoRef(path="test.mp4", pts_ns=1000000000)

        assert ref.type == "external_video"
        assert ref.path == "test.mp4"
        assert ref.pts_ns == 1000000000


class TestScreenCapturedWithFrameArray:
    """Test ScreenCaptured with direct frame data (frame_arr only)."""

    def test_create_with_frame_arr(self, sample_bgra_frame):
        """Test creating ScreenCaptured with direct frame array."""
        utc_ns = 1741608540328534500

        screen_msg = ScreenCaptured(utc_ns=utc_ns, frame_arr=sample_bgra_frame)

        assert screen_msg.utc_ns == utc_ns
        assert np.array_equal(screen_msg.frame_arr, sample_bgra_frame)
        assert screen_msg.shape == (64, 48)  # (width, height)
        assert screen_msg.source_shape is None  # Not set automatically for direct frames
        assert screen_msg.media_ref is None  # No media reference initially

    def test_lazy_load_with_existing_frame(self, sample_bgra_frame):
        """Test that lazy_load returns existing frame when frame_arr is already set."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        loaded_frame = screen_msg.lazy_load()
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

    def test_embed_from_array_png(self, sample_bgra_frame):
        """Test embedding frame data as PNG."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        # Initially no embedded data
        assert screen_msg.has_embedded_data() is False

        # Embed the frame
        screen_msg.embed_from_array(format="png")

        # Now should have embedded data
        assert screen_msg.has_embedded_data() is True
        assert screen_msg.media_ref.type == "embedded"
        assert screen_msg.media_ref.format == "png"
        assert len(screen_msg.media_ref.data) > 0  # Should have base64 data

    def test_embed_from_array_jpeg(self, sample_bgra_frame):
        """Test embedding frame data as JPEG with quality setting."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        # Embed as JPEG with specific quality
        screen_msg.embed_from_array(format="jpeg", quality=95)

        assert screen_msg.has_embedded_data() is True
        assert screen_msg.media_ref.format == "jpeg"


class TestScreenCapturedWithEmbeddedRef:
    """Test ScreenCaptured with EmbeddedRef."""

    def test_create_with_embedded_ref(self, sample_bgra_frame):
        """Test creating ScreenCaptured with embedded reference."""
        # First create an embedded reference
        screen_msg_temp = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)
        screen_msg_temp.embed_from_array(format="png")
        embedded_ref = screen_msg_temp.media_ref

        # Create new message with embedded reference
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=embedded_ref)

        assert screen_msg.utc_ns == 1741608540328534500
        assert screen_msg.frame_arr is None  # Should not be loaded yet
        assert screen_msg.has_embedded_data() is True
        assert screen_msg.has_external_reference() is False
        assert screen_msg.media_ref.format == "png"

    def test_lazy_load_from_embedded(self, sample_bgra_frame):
        """Test lazy loading from embedded data."""
        # Create embedded reference
        screen_msg_temp = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)
        screen_msg_temp.embed_from_array(format="png")

        # Create new message with just embedded data
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=screen_msg_temp.media_ref)

        # Initially no frame loaded
        assert screen_msg.frame_arr is None

        # Lazy load should work
        loaded_frame = screen_msg.lazy_load()

        assert loaded_frame is not None
        assert screen_msg.frame_arr is not None
        assert loaded_frame.shape[2] == 4  # BGRA format
        assert screen_msg.shape is not None

    def test_embedded_roundtrip(self, sample_bgra_frame):
        """Test embedding and loading back gives similar results."""
        # Original message
        original_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        # Embed as PNG
        original_msg.embed_from_array(format="png")

        # Create new message from embedded data
        embedded_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=original_msg.media_ref)

        # Load back
        loaded_frame = embedded_msg.lazy_load()

        # Should have same shape and similar content (allowing for compression)
        assert loaded_frame.shape == sample_bgra_frame.shape
        assert loaded_frame.dtype == sample_bgra_frame.dtype

    def test_embedded_jpeg_quality(self, sample_bgra_frame):
        """Test JPEG embedding with different quality settings."""
        # High quality
        msg_high = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame.copy())
        msg_high.embed_from_array(format="jpeg", quality=95)

        # Low quality
        msg_low = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame.copy())
        msg_low.embed_from_array(format="jpeg", quality=20)

        # High quality should have larger data size
        high_size = len(msg_high.media_ref.data)
        low_size = len(msg_low.media_ref.data)
        assert high_size > low_size


class TestScreenCapturedWithExternalRef:
    """Test ScreenCaptured with ExternalImageRef and ExternalVideoRef."""

    def test_create_with_external_video_ref(self, sample_video_file):
        """Test creating ScreenCaptured with external video reference."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[2] * TimeUnits.SECOND)  # Third frame (0.2s)

        media_ref = ExternalVideoRef(path=str(video_path), pts_ns=pts_ns)
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        assert screen_msg.utc_ns == 1741608540328534500
        assert screen_msg.frame_arr is None  # Should not be loaded yet
        assert screen_msg.media_ref.path == str(video_path)
        assert screen_msg.media_ref.pts_ns == pts_ns
        assert screen_msg.shape is None  # Not set until lazy loading
        assert screen_msg.source_shape is None
        assert screen_msg.has_external_reference() is True
        assert screen_msg.has_embedded_data() is False

    def test_create_with_external_image_ref(self):
        """Test creating ScreenCaptured with external image reference."""
        media_ref = ExternalImageRef(path="test_image.png")
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        assert screen_msg.media_ref.path == "test_image.png"
        # ExternalImageRef doesn't have pts_ns field
        assert screen_msg.has_external_reference() is True

    def test_lazy_loading_from_video(self, sample_video_file):
        """Test lazy loading from external video file."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[1] * TimeUnits.SECOND)  # Second frame (0.1s)

        media_ref = ExternalVideoRef(path=str(video_path), pts_ns=pts_ns)
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        # Initially, frame should not be loaded
        assert screen_msg.frame_arr is None
        assert screen_msg.shape is None

        # Trigger lazy loading
        loaded_frame = screen_msg.lazy_load()

        # After lazy loading, frame should be available
        assert loaded_frame is not None
        assert screen_msg.frame_arr is not None
        assert np.array_equal(loaded_frame, screen_msg.frame_arr)
        assert loaded_frame.shape[2] == 4  # BGRA format
        assert screen_msg.shape is not None
        assert screen_msg.source_shape is not None
        assert screen_msg.shape == screen_msg.source_shape

    def test_lazy_loading_sets_correct_shape(self, sample_video_file):
        """Test that lazy loading sets the correct shape information."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[0] * TimeUnits.SECOND)  # First frame

        media_ref = ExternalVideoRef(path=str(video_path), pts_ns=pts_ns)
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        # Trigger lazy loading
        screen_msg.lazy_load()

        # Check that shape is set correctly (width, height)
        assert screen_msg.shape == (64, 48)
        assert screen_msg.source_shape == (64, 48)

    def test_multiple_lazy_load_calls(self, sample_video_file):
        """Test that multiple calls to lazy_load don't reload the frame."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[3] * TimeUnits.SECOND)  # Fourth frame

        media_ref = ExternalVideoRef(path=str(video_path), pts_ns=pts_ns)
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        # First lazy load
        frame1 = screen_msg.lazy_load()

        # Second lazy load should return the same object
        frame2 = screen_msg.lazy_load()

        assert frame1 is frame2  # Same object reference
        assert np.array_equal(frame1, frame2)

    def test_video_frame_content_verification(self, sample_video_file):
        """Test that loaded frames have expected content based on timestamp."""
        video_path, timestamps = sample_video_file

        for i, timestamp in enumerate(timestamps[:3]):  # Test first 3 frames
            pts_ns = int(timestamp * TimeUnits.SECOND)

            media_ref = ExternalVideoRef(path=str(video_path), pts_ns=pts_ns)
            screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

            rgb_array = screen_msg.to_rgb_array()

            # Each frame should have distinct color values (i * 50)
            # Check the dominant color in the frame
            mean_color = np.mean(rgb_array[:, :, 0])  # Red channel average
            expected_color = i * 50

            # Allow some tolerance for encoding/decoding variations
            assert abs(mean_color - expected_color) < 10, (
                f"Frame {i} should have color ~{expected_color}, got {mean_color:.1f}"
            )

    def test_to_rgb_array_triggers_lazy_loading(self, sample_video_file):
        """Test that to_rgb_array triggers lazy loading when needed."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[2] * TimeUnits.SECOND)

        media_ref = ExternalVideoRef(path=str(video_path), pts_ns=pts_ns)
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        # Initially no frame loaded
        assert screen_msg.frame_arr is None

        # to_rgb_array should trigger lazy loading
        rgb_array = screen_msg.to_rgb_array()

        # After conversion, frame should be loaded
        assert screen_msg.frame_arr is not None
        assert rgb_array.shape == (48, 64, 3)  # RGB format
        assert rgb_array.dtype == np.uint8

    def test_to_pil_image_triggers_lazy_loading(self, sample_video_file):
        """Test that to_pil_image triggers lazy loading when needed."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[3] * TimeUnits.SECOND)  # Fourth frame (0.3s) instead of last frame

        media_ref = ExternalVideoRef(path=str(video_path), pts_ns=pts_ns)
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        # Initially no frame loaded
        assert screen_msg.frame_arr is None

        # to_pil_image should trigger lazy loading
        pil_image = screen_msg.to_pil_image()

        # After conversion, frame should be loaded
        assert screen_msg.frame_arr is not None
        assert isinstance(pil_image, Image.Image)
        assert pil_image.mode == "RGB"
        assert pil_image.size == (64, 48)


class TestScreenCapturedValidation:
    """Test validation and error cases across all usage patterns."""

    def test_requires_frame_or_media_ref(self):
        """Test that either frame_arr or media_ref is required."""
        with pytest.raises(
            ValueError, match="ScreenCaptured requires either 'frame_arr' or 'media_ref' to be provided"
        ):
            ScreenCaptured(utc_ns=1741608540328534500)

    def test_embed_without_frame_arr(self):
        """Test that embed_from_array requires frame_arr."""
        media_ref = ExternalVideoRef(path="test.mp4", pts_ns=1000000000)
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        with pytest.raises(ValueError, match="No frame_arr available to embed"):
            screen_msg.embed_from_array()

    def test_lazy_load_with_invalid_file(self):
        """Test lazy loading with non-existent file."""
        media_ref = ExternalVideoRef(path="non_existent_file.mp4", pts_ns=1000000000)
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        with pytest.raises((FileNotFoundError, ValueError)):
            screen_msg.lazy_load()

    def test_lazy_load_with_invalid_pts(self, sample_video_file):
        """Test lazy loading with PTS that doesn't exist in video."""
        video_path, _ = sample_video_file
        invalid_pts_ns = int(10.0 * TimeUnits.SECOND)  # Way beyond video duration

        media_ref = ExternalVideoRef(path=str(video_path), pts_ns=invalid_pts_ns)
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        with pytest.raises(ValueError):
            screen_msg.lazy_load()

    def test_lazy_load_without_sources(self, sample_bgra_frame):
        """Test lazy loading when no data sources are available."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)
        screen_msg.media_ref = None  # Remove media ref
        screen_msg.frame_arr = None  # Remove frame arr

        with pytest.raises(ValueError, match="No frame data sources available for loading"):
            screen_msg.lazy_load()

    def test_invalid_jpeg_quality(self, sample_bgra_frame):
        """Test embed_from_array with invalid JPEG quality."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        # Quality too low
        with pytest.raises(ValueError, match="JPEG quality must be between 1 and 100"):
            screen_msg.embed_from_array(format="jpeg", quality=0)

        # Quality too high
        with pytest.raises(ValueError, match="JPEG quality must be between 1 and 100"):
            screen_msg.embed_from_array(format="jpeg", quality=101)

    def test_json_serialization_without_media_ref(self, sample_bgra_frame):
        """Test that JSON serialization requires media_ref."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        with pytest.raises(ValueError, match="Cannot serialize ScreenCaptured to JSON without media_ref"):
            screen_msg.model_dump_json()

    def test_repr_method_all_patterns(self, sample_bgra_frame):
        """Test __repr__ method for all usage patterns."""
        # Test with frame_arr only
        screen_msg1 = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)
        repr_str1 = repr(screen_msg1)
        assert "ScreenCaptured" in repr_str1
        assert "utc_ns=1741608540328534500" in repr_str1
        assert "shape=(64, 48)" in repr_str1

        # Test with external video ref
        media_ref2 = ExternalVideoRef(path="test.mp4", pts_ns=2000000000)
        screen_msg2 = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref2)
        repr_str2 = repr(screen_msg2)
        assert "local_video(test.mp4@2.000s)" in repr_str2

        # Test with external image ref
        media_ref3 = ExternalImageRef(path="image.png")
        screen_msg3 = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref3)
        repr_str3 = repr(screen_msg3)
        assert "local_image(image.png)" in repr_str3

        # Test with embedded ref
        screen_msg4 = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame.copy())
        screen_msg4.embed_from_array(format="png")
        repr_str4 = repr(screen_msg4)
        assert "embedded_png" in repr_str4


class TestScreenCapturedIntegration:
    """Integration tests for multi-pattern workflows and transitions."""

    def test_external_to_frame_array_workflow(self, sample_video_file):
        """Test workflow: ExternalVideoRef → load → FrameArray."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[1] * TimeUnits.SECOND)

        # Start with external reference
        media_ref = ExternalVideoRef(path=str(video_path), pts_ns=pts_ns)
        external_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        # Load the frame
        frame_data = external_msg.lazy_load()

        # Create frame-array-based message with the same frame
        frame_msg = ScreenCaptured(utc_ns=external_msg.utc_ns, frame_arr=frame_data.copy())

        # Both should produce identical outputs
        assert np.array_equal(external_msg.to_rgb_array(), frame_msg.to_rgb_array())

        external_pil = external_msg.to_pil_image()
        frame_pil = frame_msg.to_pil_image()
        assert np.array_equal(np.array(external_pil), np.array(frame_pil))

    def test_frame_array_to_embedded_workflow(self, sample_bgra_frame):
        """Test workflow: FrameArray → embed → EmbeddedRef → load."""
        # Start with frame array
        original_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        # Embed the frame
        original_msg.embed_from_array(format="png")

        # Create new message with just the embedded data
        embedded_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=original_msg.media_ref)

        # Load the frame back
        loaded_frame = embedded_msg.lazy_load()

        # Should be very similar (allowing for compression artifacts)
        assert loaded_frame.shape == sample_bgra_frame.shape
        assert loaded_frame.dtype == sample_bgra_frame.dtype

    def test_external_to_embedded_workflow(self, sample_video_file):
        """Test workflow: ExternalVideoRef → load → embed → EmbeddedRef → load."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[0] * TimeUnits.SECOND)

        # Start with external reference
        media_ref = ExternalVideoRef(path=str(video_path), pts_ns=pts_ns)
        external_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)

        # Load from external
        external_frame = external_msg.lazy_load()

        # Create frame array message and embed
        frame_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=external_frame.copy())
        frame_msg.embed_from_array(format="jpeg", quality=90)

        # Create embedded message and load
        embedded_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=frame_msg.media_ref)
        embedded_frame = embedded_msg.lazy_load()

        # Should have same shape
        assert embedded_frame.shape == external_frame.shape
        assert embedded_frame.dtype == external_frame.dtype

    def test_cross_pattern_consistency(self, sample_bgra_frame):
        """Test that all patterns produce consistent RGB/PIL outputs."""
        # Frame array pattern
        frame_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame.copy())

        # Embedded pattern
        embedded_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame.copy())
        embedded_msg.embed_from_array(format="png")
        embedded_only_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref=embedded_msg.media_ref)

        # All should produce very similar RGB arrays (allowing for compression)
        frame_rgb = frame_msg.to_rgb_array()
        embedded_rgb = embedded_only_msg.to_rgb_array()

        assert frame_rgb.shape == embedded_rgb.shape
        assert frame_rgb.dtype == embedded_rgb.dtype

        # PIL images should also be consistent
        frame_pil = frame_msg.to_pil_image()
        embedded_pil = embedded_only_msg.to_pil_image()

        assert frame_pil.size == embedded_pil.size
        assert frame_pil.mode == embedded_pil.mode


class TestMediaUtils:
    """Test utility functions for media reference operations."""

    def test_get_media_info_none(self):
        """Test get_media_info with None reference."""
        from owa.msgs.desktop.screen import _get_media_info

        info = _get_media_info(None)
        assert info["type"] is None

    def test_get_media_info_embedded(self, sample_bgra_frame):
        """Test get_media_info with embedded reference."""
        from owa.msgs.desktop.screen import _get_media_info

        # Create embedded reference
        msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)
        msg.embed_from_array(format="jpeg", quality=90)

        info = _get_media_info(msg.media_ref)
        assert info["type"] == "embedded"
        assert info["format"] == "jpeg"
        assert info["size_bytes"] > 0

    def test_get_media_info_external_video(self):
        """Test get_media_info with external video reference."""
        from owa.msgs.desktop.screen import _get_media_info

        media_ref = ExternalVideoRef(path="test.mp4", pts_ns=1000000000)
        info = _get_media_info(media_ref)

        assert info["type"] == "external_video"
        assert info["path"] == "test.mp4"
        assert info["is_local"] is True
        assert info["is_remote"] is False
        assert info["media_type"] == "video"
        assert info["pts_ns"] == 1000000000
        assert info["pts_seconds"] == 1.0

    def test_get_media_info_external_image(self):
        """Test get_media_info with external image reference."""
        from owa.msgs.desktop.screen import _get_media_info

        media_ref = ExternalImageRef(path="https://example.com/image.png")
        info = _get_media_info(media_ref)

        assert info["type"] == "external_image"
        assert info["path"] == "https://example.com/image.png"
        assert info["is_local"] is False
        assert info["is_remote"] is True
        assert info["media_type"] == "image"

    def test_format_media_display_embedded(self, sample_bgra_frame):
        """Test format_media_display with embedded reference."""
        from owa.msgs.desktop.screen import _format_media_display

        # Create embedded reference
        msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)
        msg.embed_from_array(format="png")

        display = _format_media_display(msg.media_ref)
        assert display.startswith("embedded_png(")
        assert display.endswith("KB)")

    def test_format_media_display_external_video(self):
        """Test format_media_display with external video reference."""
        from owa.msgs.desktop.screen import _format_media_display

        media_ref = ExternalVideoRef(path="/path/to/video.mp4", pts_ns=2500000000)
        display = _format_media_display(media_ref)

        assert display == "local_video(video.mp4@2.500s)"

    def test_format_media_display_external_image_remote(self):
        """Test format_media_display with remote image reference."""
        from owa.msgs.desktop.screen import _format_media_display

        media_ref = ExternalImageRef(path="https://example.com/image.jpg")
        display = _format_media_display(media_ref)

        assert display == "remote_image(https://example.com/image.jpg)"

    def test_screen_captured_get_media_info_integration(self, sample_bgra_frame):
        """Test ScreenCaptured.get_media_info() method integration."""
        # Test with frame array (no media ref)
        msg1 = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)
        info1 = msg1.get_media_info()
        assert info1["type"] is None

        # Test with embedded data
        msg1.embed_from_array(format="jpeg", quality=90)
        info2 = msg1.get_media_info()
        assert info2["type"] == "embedded"
        assert info2["format"] == "jpeg"
        assert info2["size_bytes"] > 0

        # Test with external reference
        media_ref = ExternalVideoRef(path="test.mp4", pts_ns=1000000000)
        msg2 = ScreenCaptured(utc_ns=1741608540328534500, media_ref=media_ref)
        info3 = msg2.get_media_info()
        assert info3["type"] == "external_video"
        assert info3["path"] == "test.mp4"
        assert info3["media_type"] == "video"
        assert info3["pts_ns"] == 1000000000
        assert info3["pts_seconds"] == 1.0
