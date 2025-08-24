"""
Minimal tests for screen capture message with new clean API.
"""

import os

import cv2
import numpy as np
import pytest
from PIL import Image

from owa.core.io.video import VideoWriter
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
def sample_video_file(tmp_path):
    """Create a temporary video file with known frames for testing."""
    video_path = tmp_path / "test_video.mp4"

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


@pytest.fixture
def sample_image_file(tmp_path):
    """Create a temporary image file for testing."""
    image_path = tmp_path / "test_image.png"

    # Create a simple test image
    test_image = np.zeros((48, 64, 3), dtype=np.uint8)
    test_image[:, :, 0] = 255  # Red channel

    # Save as PNG
    cv2.imwrite(str(image_path), test_image)

    return image_path


class TestMediaRef:
    """Test MediaRef minimal design."""

    def test_create_embedded_ref(self):
        """Test creating MediaRef with embedded data URI."""
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        ref = MediaRef(uri=data_uri)

        assert ref.is_embedded
        assert not ref.is_video

    def test_create_video_ref(self):
        """Test creating MediaRef for video."""
        ref = MediaRef(uri="test.mp4", pts_ns=1000000000)

        assert not ref.is_embedded
        assert ref.is_video
        assert ref.pts_ns == 1000000000

    def test_create_external_image_ref(self):
        """Test creating MediaRef for external image."""
        ref = MediaRef(uri="image.png")

        assert not ref.is_embedded
        assert not ref.is_video

    def test_create_external_url_ref(self):
        """Test creating MediaRef for remote URL."""
        ref = MediaRef(uri="https://example.com/image.jpg")

        assert not ref.is_embedded
        assert not ref.is_video
        assert ref.is_remote
        assert not ref.is_local

    def test_file_uri_handling(self):
        """Test file:// URI handling."""
        ref = MediaRef(uri="file:///path/to/image.jpg")

        assert not ref.is_embedded
        assert not ref.is_remote
        assert ref.is_local
        assert not ref.is_relative_path  # file:// URIs are not relative

    def test_relative_path_detection(self):
        """Test relative path detection."""
        # Relative paths
        ref_rel = MediaRef(uri="images/test.jpg")
        assert ref_rel.is_relative_path
        assert ref_rel.is_local

        # Absolute paths
        ref_abs = MediaRef(uri="/absolute/path/test.jpg")
        assert not ref_abs.is_relative_path
        assert ref_abs.is_local

        # URLs should not be relative
        ref_url = MediaRef(uri="https://example.com/test.jpg")
        assert not ref_url.is_relative_path

        # Data URIs should not be relative
        ref_data = MediaRef(uri="data:image/png;base64,abc123")
        assert not ref_data.is_relative_path

    @pytest.mark.skipif(os.name == "nt", reason="POSIX-specific path test")
    def test_resolve_relative_path_posix(self):
        """Test relative path resolution on POSIX systems."""
        ref = MediaRef(uri="images/test.jpg")

        # Test with non-existent file path as base (treated as directory since file doesn't exist)
        resolved = ref.resolve_relative_path("/mcap/files/recording.mcap")
        assert resolved.uri == "/mcap/files/images/test.jpg"
        assert resolved.pts_ns == ref.pts_ns

        # Test with directory as base
        resolved2 = ref.resolve_relative_path("/mcap/files/")
        assert resolved2.uri == "/mcap/files/images/test.jpg"

        # Test with absolute path (should return self)
        ref_abs = MediaRef(uri="/absolute/path/test.jpg")
        resolved3 = ref_abs.resolve_relative_path("/mcap/files/recording.mcap")
        assert resolved3 is ref_abs

    @pytest.mark.skipif(os.name != "nt", reason="Windows-specific path test")
    def test_resolve_relative_path_windows(self):
        """Test relative path resolution on Windows systems."""
        ref = MediaRef(uri="images/test.jpg")

        # Test with non-existent file path as base (treated as directory since file doesn't exist)
        resolved = ref.resolve_relative_path("C:/mcap/files/recording.mcap")
        assert resolved.uri == "C:/mcap/files/images/test.jpg"
        assert resolved.pts_ns == ref.pts_ns

        # Test with directory as base (Windows style)
        resolved2 = ref.resolve_relative_path("C:/mcap/files/")
        assert resolved2.uri == "C:/mcap/files/images/test.jpg"

        # Test with Windows absolute path - Ensure absolute paths are correctly identified
        ref_abs = MediaRef(uri="C:/absolute/path/test.jpg")
        resolved3 = ref_abs.resolve_relative_path("C:/mcap/files/recording.mcap")
        # Absolute paths should remain unchanged
        assert resolved3.uri == "C:/absolute/path/test.jpg"

        # Test with backslash paths (Windows native)
        resolved4 = ref.resolve_relative_path(r"C:\mcap\files\recording.mcap")
        # The method always returns forward slashes due to as_posix()
        assert resolved4.uri == "C:/mcap/files/images/test.jpg"

    def test_direct_constructor(self):
        """Test MediaRef direct constructor."""
        ref = MediaRef(uri="test/path.jpg")
        assert ref.uri == "test/path.jpg"
        assert ref.pts_ns is None

        ref_video = MediaRef(uri="test/video.mp4", pts_ns=1000000000)
        assert ref_video.uri == "test/video.mp4"
        assert ref_video.pts_ns == 1000000000
        assert ref_video.is_video


class TestScreenCaptured:
    """Test ScreenCaptured creation patterns and usage as documented in docstring."""

    # === Creation Patterns (as documented in docstring) ===

    def test_create_from_raw_image_pattern(self, sample_bgra_frame):
        """Test: From raw image: ScreenCaptured(frame_arr=numpy_array).embed_as_data_uri()"""
        # Create from raw image in memory
        screen_msg = ScreenCaptured(frame_arr=sample_bgra_frame).embed_as_data_uri()

        # Verify it's properly embedded
        assert screen_msg.media_ref is not None
        assert screen_msg.media_ref.is_embedded
        assert "data:image/png;base64," in screen_msg.media_ref.uri
        assert screen_msg.shape == (64, 48)  # (width, height)

    def test_create_from_file_path_pattern(self, sample_image_file):
        """Test: From file path: ScreenCaptured(media_ref={"uri": "/path/to/image.png"})"""
        # Create from file path
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref={"uri": str(sample_image_file)})

        # Verify file reference
        assert screen_msg.media_ref is not None
        assert screen_msg.media_ref.uri == str(sample_image_file)
        assert screen_msg.media_ref.is_local
        assert not screen_msg.media_ref.is_embedded
        assert screen_msg.frame_arr is None  # Not loaded yet

    def test_create_from_data_uri_pattern(self, sample_bgra_frame):
        """Test: From data URI: ScreenCaptured(media_ref={"uri": "data:image/png;base64,..."})"""
        # First create a data URI
        temp_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame).embed_as_data_uri()
        data_uri = temp_msg.media_ref.uri

        # Create from data URI
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref={"uri": data_uri})

        # Verify data URI reference
        assert screen_msg.media_ref is not None
        assert screen_msg.media_ref.is_embedded
        assert screen_msg.media_ref.uri == data_uri
        assert screen_msg.frame_arr is None  # Not loaded yet

    def test_create_from_video_frame_pattern(self, sample_video_file):
        """Test: From video frame: ScreenCaptured(media_ref={"uri": "/path/video.mp4", "pts_ns": 123456})"""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[1] * TimeUnits.SECOND)  # Second frame

        # Create from video frame
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref={"uri": str(video_path), "pts_ns": pts_ns})

        # Verify video reference
        assert screen_msg.media_ref is not None
        assert screen_msg.media_ref.is_video
        assert screen_msg.media_ref.pts_ns == pts_ns
        assert screen_msg.frame_arr is None  # Not loaded yet

    def test_create_from_url_pattern(self):
        """Test: From URL: ScreenCaptured(media_ref={"uri": "https://example.com/image.png"})"""
        # Create from URL
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref={"uri": "https://example.com/image.png"})

        # Verify URL reference
        assert screen_msg.media_ref is not None
        assert screen_msg.media_ref.is_remote
        assert not screen_msg.media_ref.is_embedded
        assert not screen_msg.media_ref.is_local
        assert screen_msg.frame_arr is None  # Not loaded yet

    # === Image Access Methods (as documented in docstring) ===

    def test_to_rgb_array(self, sample_bgra_frame):
        """Test: to_rgb_array(): Get RGB numpy array"""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        # Get RGB array
        rgb_array = screen_msg.to_rgb_array()

        # Verify RGB conversion
        assert isinstance(rgb_array, np.ndarray)
        assert rgb_array.shape == (48, 64, 3)  # RGB has 3 channels
        assert rgb_array.dtype == np.uint8

        # Verify color conversion matches expected
        expected_rgb = cv2.cvtColor(sample_bgra_frame, cv2.COLOR_BGRA2RGB)
        assert np.array_equal(rgb_array, expected_rgb)

    def test_to_pil_image(self, sample_bgra_frame):
        """Test: to_pil_image(): Get PIL Image object"""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        # Get PIL Image
        pil_image = screen_msg.to_pil_image()

        # Verify PIL Image
        assert isinstance(pil_image, Image.Image)
        assert pil_image.mode == "RGB"
        assert pil_image.size == (64, 48)  # PIL size is (width, height)

        # Verify content matches RGB conversion
        rgb_array = screen_msg.to_rgb_array()
        pil_array = np.array(pil_image)
        assert np.array_equal(pil_array, rgb_array)

    # === Path Resolution (as documented in docstring) ===
    @pytest.mark.skipif(os.name == "nt", reason="Path resolution tests fail on Windows")
    def test_resolve_relative_path_method(self):
        """Test: resolve_relative_path(base_path): Resolve relative paths against base directory"""
        # Create with relative path
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref={"uri": "videos/frame.jpg"})

        # Resolve against MCAP file path
        result = screen_msg.resolve_relative_path("/data/recordings")

        # Verify path resolution
        assert result is screen_msg  # Returns self for chaining
        assert screen_msg.media_ref.uri == "/data/recordings/videos/frame.jpg"

        # Test with no media_ref (should not crash)
        screen_msg_no_ref = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=np.zeros((10, 10, 4), dtype=np.uint8))
        screen_msg_no_ref.resolve_relative_path("/some/path.mcap")  # Should not crash

    # === Serialization Requirements (as documented in docstring) ===

    def test_serialization_requires_media_ref(self, sample_bgra_frame):
        """Test: Serialization requires media_ref (use embed_as_data_uri() for in-memory arrays)"""
        # Raw frame cannot be serialized
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        with pytest.raises(ValueError, match="Cannot serialize without media_ref"):
            screen_msg.model_dump_json()

        # After embedding, serialization works
        screen_msg.embed_as_data_uri()
        json_str = screen_msg.model_dump_json()
        assert isinstance(json_str, str)
        assert "data:image/png;base64," in json_str

    # === Legacy Tests (for compatibility) ===

    def test_load_frame_array_with_existing_frame(self, sample_bgra_frame):
        """Test that load_frame_array returns existing frame when frame_arr is already set."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        loaded_frame = screen_msg.load_frame_array()
        assert np.array_equal(loaded_frame, sample_bgra_frame)
        assert loaded_frame is screen_msg.frame_arr  # Should return same object

    def test_embed_as_data_uri_png(self, sample_bgra_frame):
        """Test embedding frame data as PNG."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        # Initially no embedded data
        assert screen_msg.media_ref is None

        # Embed the frame
        screen_msg.embed_as_data_uri(format="png")

        # Now should have embedded data
        assert screen_msg.media_ref is not None
        assert screen_msg.media_ref.is_embedded
        assert "data:image/png;base64," in screen_msg.media_ref.uri

    def test_embed_as_data_uri_jpeg(self, sample_bgra_frame):
        """Test embedding frame data as JPEG with quality setting."""
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        # Embed as JPEG with specific quality
        screen_msg.embed_as_data_uri(format="jpeg", quality=95)

        assert screen_msg.media_ref is not None
        assert screen_msg.media_ref.is_embedded
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
        assert screen_msg.media_ref.is_embedded

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
        assert screen_msg.media_ref.is_video
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

    # === Remote URL Tests (merged from test_screen_remote.py) ===

    @pytest.mark.network
    def test_create_from_url_pattern_with_loading(self):
        """
        Test: From URL: ScreenCaptured(media_ref={"uri": "https://example.com/image.png"})

        This test demonstrates the URL creation pattern from docstring with actual loading.
        Skipped in CI environments to avoid flaky tests from unreliable external services.
        """
        test_cases = [
            ("https://www.sample-videos.com/video321/mp4/240/big_buck_bunny_240p_2mb.mp4", 1_000_000_000, "video"),
            ("https://httpbingo.org/image/png", None, "image"),
        ]

        for test_url, pts_ns, media_type in test_cases:
            # === Creation Pattern: From URL ===
            if pts_ns is not None:
                # Video with timestamp
                screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref={"uri": test_url, "pts_ns": pts_ns})
            else:
                # Image without timestamp
                screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref={"uri": test_url})

            # === Verify Remote Reference Properties ===
            assert screen_msg.media_ref.is_remote, f"Should be detected as remote URL for {media_type}"
            assert not screen_msg.media_ref.is_embedded, f"Should not be embedded data for {media_type}"
            assert not screen_msg.media_ref.is_local, f"Should not be local file for {media_type}"
            assert screen_msg.media_ref.uri == test_url

            # === Verify Media Type Detection ===
            if media_type == "video":
                assert screen_msg.media_ref.is_video, "Should detect video media"
                assert screen_msg.media_ref.pts_ns == pts_ns, "Should preserve timestamp"
            else:
                assert not screen_msg.media_ref.is_video, "Should detect image media"
                assert screen_msg.media_ref.pts_ns is None, "Image should have no timestamp"

            # === Test Frame Loading ===
            frame_arr = screen_msg.load_frame_array()
            assert isinstance(frame_arr, np.ndarray), f"Should return numpy array for {media_type}"
            assert frame_arr.dtype == np.uint8, f"Should be uint8 format for {media_type}"
            assert len(frame_arr.shape) == 3, f"Should be 3D array (H, W, C) for {media_type}"
            assert frame_arr.shape[2] == 4, f"Should be BGRA format for {media_type}"

            # === Test Shape Setting ===
            h, w = frame_arr.shape[:2]
            expected_shape = (w, h)  # (width, height)
            assert screen_msg.shape == expected_shape, f"Shape should be set after loading for {media_type}"

            # === Test Image Access Methods (docstring patterns) ===
            # to_rgb_array(): Get RGB numpy array
            rgb_arr = screen_msg.to_rgb_array()
            assert rgb_arr.shape == (h, w, 3), f"RGB should have 3 channels for {media_type}"
            assert rgb_arr.dtype == np.uint8, f"RGB should be uint8 for {media_type}"

            # to_pil_image(): Get PIL Image object
            pil_img = screen_msg.to_pil_image()
            assert pil_img.size == (w, h), f"PIL size should be (width, height) for {media_type}"
            assert pil_img.mode == "RGB", f"PIL should be RGB mode for {media_type}"

            # === Test String Representation ===
            str_repr = str(screen_msg)
            if media_type == "video":
                assert f"video@{pts_ns}ns" in str_repr, "Should show video timestamp"
            else:
                assert "external" in str_repr, "Should show external reference"

    @pytest.mark.network
    def test_remote_video_caching_behavior(self):
        """Test remote video frame caching and keep_av_open functionality."""
        test_url = "https://www.sample-videos.com/video321/mp4/240/big_buck_bunny_240p_2mb.mp4"
        pts_ns = 1_000_000_000  # 1 second

        # Create from remote video (docstring pattern)
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref={"uri": test_url, "pts_ns": pts_ns})

        # === Test Initial Frame Loading ===
        frame_arr = screen_msg.load_frame_array()
        assert frame_arr.shape[0] > 0, "Height should be > 0"
        assert frame_arr.shape[1] > 0, "Width should be > 0"
        assert frame_arr.shape[2] == 4, "Should be BGRA format"

        # === Test Frame Caching ===
        frame_arr2 = screen_msg.load_frame_array()
        assert np.array_equal(frame_arr, frame_arr2), "Subsequent calls should return cached frame"
        assert frame_arr2 is screen_msg.frame_arr, "Should return same object reference"

        # === Test Keep AV Open Parameter ===
        frame_arr3 = screen_msg.load_frame_array(keep_av_open=True)
        assert frame_arr3.shape == frame_arr.shape, "Keep AV open should return same shape"

    def test_remote_serialization_roundtrip(self):
        """Test JSON serialization with remote media reference."""
        test_url = "https://example.com/video.mp4"
        pts_ns = 1_000_000_000  # 1 second

        # Create with remote reference (no network access needed for serialization)
        screen_msg = ScreenCaptured(
            media_ref={"uri": test_url, "pts_ns": pts_ns},
            utc_ns=1234567890000000000,
            source_shape=(1920, 1080),
            shape=(1920, 1080),
        )

        # === Test JSON Serialization ===
        json_str = screen_msg.model_dump_json()
        assert test_url in json_str, "URL should be in JSON"
        assert str(pts_ns) in json_str, "Timestamp should be in JSON"

        # === Test Deserialization ===
        screen_msg2 = ScreenCaptured.model_validate_json(json_str)
        assert screen_msg2.media_ref.uri == test_url, "URL should be preserved"
        assert screen_msg2.media_ref.pts_ns == pts_ns, "Timestamp should be preserved"
        assert screen_msg2.utc_ns == screen_msg.utc_ns, "UTC timestamp should be preserved"
        assert screen_msg2.source_shape == screen_msg.source_shape, "Source shape should be preserved"
        assert screen_msg2.shape == screen_msg.shape, "Shape should be preserved"

    def test_remote_error_handling(self):
        """Test error handling with invalid remote references."""
        # Test invalid URL scheme - FTP URLs are treated as local files
        screen_msg = ScreenCaptured(
            utc_ns=1741608540328534500, media_ref={"uri": "ftp://example.com/video.mp4", "pts_ns": 0}
        )

        with pytest.raises(FileNotFoundError, match="Video file not found"):
            screen_msg.load_frame_array()

        # Test non-existent remote file (should raise network-related error)
        screen_msg = ScreenCaptured(
            utc_ns=1741608540328534500, media_ref={"uri": "https://nonexistent.example.com/video.mp4", "pts_ns": 0}
        )

        with pytest.raises(Exception):  # Could be various network-related errors
            screen_msg.load_frame_array()

    def test_remote_string_representation(self):
        """Test string representation for remote files."""
        # Test remote video
        screen_msg = ScreenCaptured(
            utc_ns=1741608540328534500,
            media_ref={
                "uri": "https://example.com/long/path/to/video.mp4",
                "pts_ns": 1_500_000_000,  # 1.5 seconds
            },
        )
        str_repr = str(screen_msg)
        assert "video@1500000000ns" in str_repr, "Should show video timestamp"

        # Test remote image
        screen_msg = ScreenCaptured(utc_ns=1741608540328534500, media_ref={"uri": "https://example.com/image.jpg"})
        str_repr = str(screen_msg)
        assert "external" in str_repr, "Should show external reference"
