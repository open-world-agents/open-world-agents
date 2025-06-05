import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from owa.core.io.video import VideoWriter, force_close_video_container
from owa.core.time import TimeUnits
from owa.env.gst.msg import ScreenEmitted


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


class TestScreenEmittedOnRAM:
    """Test ScreenEmitted with direct frame data (on-RAM scenarios)."""

    def test_create_with_frame_arr(self, sample_bgra_frame):
        """Test creating ScreenEmitted with direct frame array."""
        utc_ns = 1741608540328534500

        screen_msg = ScreenEmitted(utc_ns=utc_ns, frame_arr=sample_bgra_frame)

        assert screen_msg.utc_ns == utc_ns
        assert np.array_equal(screen_msg.frame_arr, sample_bgra_frame)
        assert screen_msg.shape == (64, 48)  # (width, height)
        assert screen_msg.original_shape is None  # Not set in post_init for direct frames
        assert screen_msg.path is None
        assert screen_msg.pts is None

    def test_lazy_load_with_existing_frame(self, sample_bgra_frame):
        """Test that lazy_load returns existing frame when frame_arr is already set."""
        screen_msg = ScreenEmitted(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        loaded_frame = screen_msg.lazy_load()
        assert np.array_equal(loaded_frame, sample_bgra_frame)
        assert loaded_frame is screen_msg.frame_arr  # Should return same object

    def test_to_rgb_array(self, sample_bgra_frame):
        """Test conversion from BGRA to RGB array."""
        screen_msg = ScreenEmitted(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        rgb_array = screen_msg.to_rgb_array()

        # Verify shape and conversion
        assert rgb_array.shape == (48, 64, 3)  # RGB has 3 channels
        assert rgb_array.dtype == np.uint8

        # Verify color conversion (BGRA -> RGB)
        expected_rgb = cv2.cvtColor(sample_bgra_frame, cv2.COLOR_BGRA2RGB)
        assert np.array_equal(rgb_array, expected_rgb)

    def test_to_pil_image(self, sample_bgra_frame):
        """Test conversion to PIL Image."""
        screen_msg = ScreenEmitted(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        pil_image = screen_msg.to_pil_image()

        assert isinstance(pil_image, Image.Image)
        assert pil_image.mode == "RGB"
        assert pil_image.size == (64, 48)  # PIL size is (width, height)

        # Verify content matches RGB conversion
        rgb_array = screen_msg.to_rgb_array()
        pil_array = np.array(pil_image)
        assert np.array_equal(pil_array, rgb_array)


class TestScreenEmittedOnDisk:
    """Test ScreenEmitted with path/PTS data (on-disk scenarios with lazy loading)."""

    def test_create_with_path_and_pts(self, sample_video_file):
        """Test creating ScreenEmitted with path and PTS."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[2] * TimeUnits.SECOND)  # Third frame (0.2s)

        screen_msg = ScreenEmitted(utc_ns=1741608540328534500, path=str(video_path), pts=pts_ns)

        assert screen_msg.utc_ns == 1741608540328534500
        assert screen_msg.frame_arr is None  # Should not be loaded yet
        assert screen_msg.path == str(video_path)
        assert screen_msg.pts == pts_ns
        assert screen_msg.shape is None  # Not set until lazy loading
        assert screen_msg.original_shape is None

    def test_lazy_loading_functionality(self, sample_video_file):
        """Test that lazy loading works correctly."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[1] * TimeUnits.SECOND)  # Second frame (0.1s)

        screen_msg = ScreenEmitted(utc_ns=1741608540328534500, path=str(video_path), pts=pts_ns)

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
        assert screen_msg.original_shape is not None
        assert screen_msg.shape == screen_msg.original_shape

    def test_lazy_loading_sets_correct_shape(self, sample_video_file):
        """Test that lazy loading sets the correct shape information."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[0] * TimeUnits.SECOND)  # First frame

        screen_msg = ScreenEmitted(utc_ns=1741608540328534500, path=str(video_path), pts=pts_ns)

        # Trigger lazy loading
        screen_msg.lazy_load()

        # Check that shape is set correctly (width, height)
        assert screen_msg.shape == (64, 48)
        assert screen_msg.original_shape == (64, 48)

    def test_multiple_lazy_load_calls(self, sample_video_file):
        """Test that multiple calls to lazy_load don't reload the frame."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[3] * TimeUnits.SECOND)  # Fourth frame

        screen_msg = ScreenEmitted(utc_ns=1741608540328534500, path=str(video_path), pts=pts_ns)

        # First lazy load
        frame1 = screen_msg.lazy_load()

        # Second lazy load should return the same object
        frame2 = screen_msg.lazy_load()

        assert frame1 is frame2  # Same object reference
        assert np.array_equal(frame1, frame2)

    def test_to_rgb_array_triggers_lazy_loading(self, sample_video_file):
        """Test that to_rgb_array triggers lazy loading when needed."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[2] * TimeUnits.SECOND)

        screen_msg = ScreenEmitted(utc_ns=1741608540328534500, path=str(video_path), pts=pts_ns)

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

        screen_msg = ScreenEmitted(utc_ns=1741608540328534500, path=str(video_path), pts=pts_ns)

        # Initially no frame loaded
        assert screen_msg.frame_arr is None

        # to_pil_image should trigger lazy loading
        pil_image = screen_msg.to_pil_image()

        # After conversion, frame should be loaded
        assert screen_msg.frame_arr is not None
        assert isinstance(pil_image, Image.Image)
        assert pil_image.mode == "RGB"
        assert pil_image.size == (64, 48)


class TestScreenEmittedValidation:
    """Test validation and error cases."""

    def test_requires_frame_or_path_pts(self):
        """Test that either frame_arr or (path + pts) is required."""
        with pytest.raises(ValueError, match="ScreenEmitted requires either 'frame_arr' or both 'path' and 'pts'"):
            ScreenEmitted(utc_ns=1741608540328534500)

    def test_requires_both_path_and_pts(self):
        """Test that both path and pts are required when using disk-based loading."""
        # Only path, no pts
        with pytest.raises(ValueError, match="ScreenEmitted requires either 'frame_arr' or both 'path' and 'pts'"):
            ScreenEmitted(utc_ns=1741608540328534500, path="test.mp4")

        # Only pts, no path
        with pytest.raises(ValueError, match="ScreenEmitted requires either 'frame_arr' or both 'path' and 'pts'"):
            ScreenEmitted(utc_ns=1741608540328534500, pts=1000000000)

    def test_lazy_load_with_invalid_file(self):
        """Test lazy loading with non-existent file."""
        screen_msg = ScreenEmitted(utc_ns=1741608540328534500, path="non_existent_file.mp4", pts=1000000000)

        with pytest.raises((FileNotFoundError, ValueError)):
            screen_msg.lazy_load()

    def test_lazy_load_with_invalid_pts(self, sample_video_file):
        """Test lazy loading with PTS that doesn't exist in video."""
        video_path, _ = sample_video_file
        invalid_pts_ns = int(10.0 * TimeUnits.SECOND)  # Way beyond video duration

        screen_msg = ScreenEmitted(utc_ns=1741608540328534500, path=str(video_path), pts=invalid_pts_ns)

        with pytest.raises(ValueError, match="Frame not found"):
            screen_msg.lazy_load()

    def test_repr_method(self, sample_bgra_frame):
        """Test the __repr__ method."""
        # Test with frame_arr
        screen_msg = ScreenEmitted(utc_ns=1741608540328534500, frame_arr=sample_bgra_frame)

        repr_str = repr(screen_msg)
        assert "ScreenEmitted" in repr_str
        assert "utc_ns=1741608540328534500" in repr_str
        assert "shape=(64, 48)" in repr_str

        # Test with path/pts
        screen_msg2 = ScreenEmitted(utc_ns=1741608540328534500, path="test.mp4", pts=2000000000)

        repr_str2 = repr(screen_msg2)
        assert "path='test.mp4'" in repr_str2
        assert "pts=2000000000" in repr_str2


class TestScreenEmittedIntegration:
    """Integration tests combining multiple features."""

    def test_disk_to_ram_conversion(self, sample_video_file):
        """Test converting from disk-based to RAM-based ScreenEmitted."""
        video_path, timestamps = sample_video_file
        pts_ns = int(timestamps[1] * TimeUnits.SECOND)

        # Start with disk-based
        disk_msg = ScreenEmitted(utc_ns=1741608540328534500, path=str(video_path), pts=pts_ns)

        # Load the frame
        frame_data = disk_msg.lazy_load()

        # Create RAM-based with the same frame
        ram_msg = ScreenEmitted(utc_ns=disk_msg.utc_ns, frame_arr=frame_data.copy())

        # Both should produce identical outputs
        assert np.array_equal(disk_msg.to_rgb_array(), ram_msg.to_rgb_array())

        disk_pil = disk_msg.to_pil_image()
        ram_pil = ram_msg.to_pil_image()
        assert np.array_equal(np.array(disk_pil), np.array(ram_pil))

    def test_frame_content_verification(self, sample_video_file):
        """Test that loaded frames have expected content based on timestamp."""
        video_path, timestamps = sample_video_file

        for i, timestamp in enumerate(timestamps[:3]):  # Test first 3 frames
            pts_ns = int(timestamp * TimeUnits.SECOND)

            screen_msg = ScreenEmitted(utc_ns=1741608540328534500, path=str(video_path), pts=pts_ns)

            rgb_array = screen_msg.to_rgb_array()

            # Each frame should have distinct color values (i * 50)
            # Check the dominant color in the frame
            mean_color = np.mean(rgb_array[:, :, 0])  # Red channel average
            expected_color = i * 50

            # Allow some tolerance for encoding/decoding variations
            assert abs(mean_color - expected_color) < 10, (
                f"Frame {i} should have color ~{expected_color}, got {mean_color:.1f}"
            )
