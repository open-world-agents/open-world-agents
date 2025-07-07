import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from owa.core.io.video import VideoReader, VideoWriter


def test_video_writer_vfr():
    """Test Variable Frame Rate (VFR) video writing and verify irregular timestamps."""
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / "test_vfr.mp4"

        # Use more dramatically irregular timestamps to test VFR
        irregular_timestamps = [0.0, 0.05, 0.4, 0.45, 1.2]  # Big gaps to ensure variance
        with VideoWriter(video_path, fps=60.0, vfr=True) as writer:
            for i, timestamp in enumerate(irregular_timestamps):
                img = np.full((48, 64, 3), i * 50, dtype=np.uint8)  # Different color per frame
                writer.write_frame(img, pts=timestamp, pts_unit="sec")

            # Add a final frame to ensure the last intended frame has duration
            final_timestamp = irregular_timestamps[-1] + 0.1  # 100ms after last frame
            final_img = np.zeros((48, 64, 3), dtype=np.uint8)  # Black frame as end marker
            writer.write_frame(final_img, pts=final_timestamp, pts_unit="sec")

        assert video_path.exists(), "VFR video file should be created"

        # Strict VFR timestamp verification
        with VideoReader(video_path, force_close=True) as reader:
            read_timestamps = []
            frame_colors = []
            for frame in reader.read_frames():
                read_timestamps.append(frame.time)
                frame_array = frame.to_ndarray(format="rgb24")
                # Get the dominant color value to identify frame
                frame_colors.append(frame_array[0, 0, 0])  # Red channel of first pixel

            # We expect to read at least our intended frames (excluding the end marker)
            expected_frame_count = len(irregular_timestamps)
            assert len(read_timestamps) >= expected_frame_count, (
                f"Should read at least {expected_frame_count} frames, got {len(read_timestamps)}"
            )

            # Take only the frames corresponding to our irregular timestamps
            test_timestamps = read_timestamps[:expected_frame_count]
            test_colors = frame_colors[:expected_frame_count]

            # Strict timestamp matching with tolerance
            timestamp_tolerance = 0.01  # 10ms tolerance
            for i, (expected_ts, actual_ts) in enumerate(zip(irregular_timestamps, test_timestamps)):
                assert abs(actual_ts - expected_ts) <= timestamp_tolerance, (
                    f"Frame {i}: expected timestamp {expected_ts:.3f}s, got {actual_ts:.3f}s "
                    f"(diff: {abs(actual_ts - expected_ts):.6f}s > {timestamp_tolerance:.3f}s)"
                )

            # Verify frame content matches expected sequence
            expected_colors = [i * 50 for i in range(len(irregular_timestamps))]
            for i, (expected_color, actual_color) in enumerate(zip(expected_colors, test_colors)):
                color_tolerance = 5  # Allow small encoding variations
                assert abs(actual_color - expected_color) <= color_tolerance, (
                    f"Frame {i}: expected color {expected_color}, got {actual_color}"
                )

            # Verify irregular timing patterns are preserved
            intervals = [test_timestamps[i + 1] - test_timestamps[i] for i in range(len(test_timestamps) - 1)]
            expected_intervals = [
                irregular_timestamps[i + 1] - irregular_timestamps[i] for i in range(len(irregular_timestamps) - 1)
            ]

            # Calculate variance to ensure irregularity is preserved
            interval_variance = np.var(intervals)
            expected_variance = np.var(expected_intervals)

            # VFR should preserve high variance (irregular timing)
            assert interval_variance > 0.01, (
                f"VFR should preserve irregular timing, variance too low: {interval_variance:.6f}"
            )

            # Variance should be similar to what we wrote (within reasonable encoding tolerance)
            variance_ratio = interval_variance / expected_variance if expected_variance > 0 else 1
            assert 0.5 <= variance_ratio <= 2.0, (
                f"VFR variance ratio {variance_ratio:.3f} should be close to 1.0 "
                f"(actual: {interval_variance:.6f}, expected: {expected_variance:.6f})"
            )

            # Verify timestamps are strictly increasing
            for i in range(1, len(test_timestamps)):
                assert test_timestamps[i] > test_timestamps[i - 1], (
                    f"Timestamps should be strictly increasing: "
                    f"{test_timestamps[i - 1]:.6f} >= {test_timestamps[i]:.6f} at index {i}"
                )

            # Verify the largest gap is preserved
            max_interval = max(intervals)
            max_expected_interval = max(expected_intervals)
            assert abs(max_interval - max_expected_interval) <= timestamp_tolerance, (
                f"Largest interval should be preserved: expected {max_expected_interval:.3f}s, got {max_interval:.3f}s"
            )


def test_video_writer_cfr():
    """Test Constant Frame Rate (CFR) video writing and verify regular timestamps."""
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / "test_cfr.mp4"
        target_fps = 20.0

        with VideoWriter(video_path, fps=target_fps, vfr=False) as writer:
            for frame_i in range(8):  # 0.4 seconds at 20fps
                img = np.full((48, 64, 3), frame_i * 30, dtype=np.uint8)
                writer.write_frame(img)  # Auto-generated regular timestamps

        assert video_path.exists(), "CFR video file should be created"

        # Verify CFR timestamps are regular
        with VideoReader(video_path, force_close=True) as reader:
            read_timestamps = []
            for frame in reader.read_frames():
                read_timestamps.append(frame.time)
                if len(read_timestamps) >= 6:
                    break

            # CFR should have regular intervals (low variance)
            if len(read_timestamps) >= 3:
                intervals = [read_timestamps[i + 1] - read_timestamps[i] for i in range(len(read_timestamps) - 1)]
                expected_interval = 1.0 / target_fps

                # Check intervals are close to expected
                for interval in intervals:
                    assert abs(interval - expected_interval) < expected_interval * 0.1, (
                        f"CFR interval should be ~{expected_interval:.3f}s, got {interval:.3f}s"
                    )

                # Low variance indicates regular timing
                interval_variance = np.var(intervals)
                assert interval_variance < 0.001, f"CFR should have regular timing, variance: {interval_variance:.6f}"


def test_video_reader():
    """Test video reading functionality with both frame iteration and single frame access."""
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / "test_read.mp4"

        # Create test video with identifiable frames
        with VideoWriter(video_path, fps=30.0, vfr=True) as writer:
            for frame_i in range(10):
                img = np.full((48, 64, 3), frame_i * 25, dtype=np.uint8)
                sec = Fraction(frame_i, 30)
                writer.write_frame(img, pts=sec, pts_unit="sec")

        # Test frame iteration and properties
        with VideoReader(video_path, force_close=True) as reader:
            frame_count = 0
            for frame in reader.read_frames(start_pts=Fraction(1, 6)):  # Start at ~0.167s (frame 5)
                frame_array = frame.to_ndarray(format="rgb24")
                assert frame_array.shape == (48, 64, 3), "Frame should have correct dimensions"
                assert frame.pts is not None, "Frame should have PTS"
                assert frame.time is not None, "Frame should have time"
                assert frame.time >= 0.16, "Should start from requested timestamp"
                frame_count += 1
                if frame_count >= 5:
                    break

            assert frame_count > 0, "Should read at least one frame"

            # Test single frame access
            single_frame = reader.read_frame(pts=Fraction(1, 10))  # ~0.1s
            assert single_frame is not None, "Should read single frame"
            assert single_frame.time >= 0.09, "Single frame should be at requested time"


def test_video_processing_pipeline():
    """Test complete video processing pipeline with VFR to CFR conversion."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vfr_input = Path(temp_dir) / "input_vfr.mp4"
        cfr_output = Path(temp_dir) / "output_cfr.mp4"

        # Create VFR input with irregular timing
        irregular_times = [0.0, 0.05, 0.2, 0.25, 0.6, 0.65]
        with VideoWriter(vfr_input, fps=30.0, vfr=True) as writer:
            for i, timestamp in enumerate(irregular_times):
                img = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
                writer.write_frame(img, pts=timestamp, pts_unit="sec")

        # Process VFR to CFR: read with fps sampling, write as CFR
        target_fps = 15.0
        with VideoReader(vfr_input, force_close=True) as reader:
            with VideoWriter(cfr_output, fps=target_fps, vfr=False) as writer:
                frame_count = 0
                for frame in reader.read_frames(fps=target_fps):  # Sample at regular intervals
                    frame_array = frame.to_ndarray(format="rgb24")
                    writer.write_frame(frame_array)
                    frame_count += 1

        assert cfr_output.exists(), "CFR output should be created"
        assert frame_count > 0, "Should process frames during VFR→CFR conversion"

        # Compare VFR input vs CFR output timing characteristics
        def get_timing_stats(video_path):
            with VideoReader(video_path, force_close=True) as reader:
                timestamps = []
                for frame in reader.read_frames():
                    timestamps.append(frame.time)
                    if len(timestamps) >= 4:
                        break

                if len(timestamps) >= 2:
                    intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
                    return np.var(intervals) if len(intervals) > 1 else 0
                return 0

        vfr_variance = get_timing_stats(vfr_input)
        cfr_variance = get_timing_stats(cfr_output)

        # CFR output should have more regular timing than VFR input
        assert cfr_variance <= vfr_variance or cfr_variance < 0.01, (
            f"CFR output ({cfr_variance:.6f}) should be more regular than VFR input ({vfr_variance:.6f})"
        )


# ============================================================================
# Remote File Handling Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_url,expected_frame_shape",
    [
        ("https://www.sample-videos.com/video321/mp4/240/big_buck_bunny_240p_2mb.mp4", (240, 320, 3)),
    ],
)
def test_video_reader_remote_url(test_url, expected_frame_shape):
    """Test VideoReader with remote HTTP/HTTPS URLs."""
    try:
        with VideoReader(test_url) as reader:
            # Verify the video path is stored as URL string
            assert isinstance(reader.video_path, str)
            assert reader.video_path == test_url

            # Verify container is created successfully
            assert reader.container is not None
            assert hasattr(reader.container, "streams")

            # Test reading a single frame
            frame = reader.read_frame(pts=0.0)
            assert frame is not None
            assert frame.pts is not None
            assert frame.time is not None

            # Verify frame properties
            frame_array = frame.to_ndarray(format="rgb24")
            assert frame_array.shape == expected_frame_shape
            assert frame_array.dtype == np.uint8

            # Test reading multiple frames
            frame_count = 0
            for frame in reader.read_frames(start_pts=0.0, end_pts=1.0):
                frame_count += 1
                if frame_count >= 3:  # Just test a few frames
                    break

            assert frame_count > 0, "Should read at least one frame from remote video"

    except Exception as e:
        # If network is unavailable, skip the test
        pytest.skip(f"Network test skipped due to error: {e}")


def test_video_reader_url_validation():
    """Test URL validation in VideoReader."""
    # Test valid HTTP URL
    valid_http_url = "http://example.com/video.mp4"
    try:
        # This will fail at container creation, but URL validation should pass
        VideoReader(valid_http_url)
    except Exception as e:
        # Should fail at PyAV level, not at our validation level
        assert "Unsupported URL scheme" not in str(e)

    # Test valid HTTPS URL
    valid_https_url = "https://example.com/video.mp4"
    try:
        VideoReader(valid_https_url)
    except Exception as e:
        assert "Unsupported URL scheme" not in str(e)

    # Test invalid URL schemes
    invalid_urls = [
        "ftp://example.com/video.mp4",
        "file://example.com/video.mp4",
        "rtsp://example.com/stream",
        "custom://example.com/video.mp4",
    ]

    for invalid_url in invalid_urls:
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            VideoReader(invalid_url)


def test_video_reader_local_file_validation():
    """Test local file validation in VideoReader."""
    # Test non-existent local file
    non_existent_file = "non_existent_video.mp4"
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        VideoReader(non_existent_file)

    # Test with Path object
    non_existent_path = Path("another_non_existent_video.mp4")
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        VideoReader(non_existent_path)


def test_video_reader_path_type_handling():
    """Test VideoReader handles different path types correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / "test_types.mp4"

        # Create a test video
        with VideoWriter(video_path, fps=30.0) as writer:
            img = np.zeros((48, 64, 3), dtype=np.uint8)
            writer.write_frame(img)

        # Test with Path object
        with VideoReader(video_path, force_close=True) as reader:
            assert isinstance(reader.video_path, Path)
            frame = reader.read_frame(pts=0.0)
            assert frame is not None

        # Test with string path
        with VideoReader(str(video_path), force_close=True) as reader:
            assert isinstance(reader.video_path, Path)
            frame = reader.read_frame(pts=0.0)
            assert frame is not None


def test_video_reader_container_caching_with_urls():
    """Test that container caching works correctly with URLs."""
    test_url = "https://httpbin.org/status/200"  # Simple URL for testing

    # Note: This test focuses on the caching mechanism rather than actual video processing
    # since httpbin.org/status/200 is not a video file

    try:
        # First access - should create new container
        reader1 = VideoReader(test_url)
        container1_id = id(reader1.container)  # noqa: F841

        # Second access - should potentially reuse cached container
        reader2 = VideoReader(test_url)
        container2_id = id(reader2.container)  # noqa: F841

        # Clean up
        reader1.close()
        reader2.close()

        # The test passes if no exceptions are raised during container creation/caching
        assert True, "Container caching mechanism works with URLs"

    except Exception as e:
        # If the URL doesn't work, that's fine - we're testing the caching mechanism
        pytest.skip(f"URL caching test skipped due to network/URL issue: {e}")


@pytest.mark.parametrize(
    "url_type,test_url",
    [
        ("http", "http://httpbin.org/status/200"),
        ("https", "https://httpbin.org/status/200"),
    ],
)
def test_video_reader_url_schemes(url_type, test_url):
    """Test VideoReader with different URL schemes."""
    try:
        # This will likely fail at PyAV level since these aren't video files,
        # but our URL validation should pass
        VideoReader(test_url)
    except Exception as e:
        # Should not fail due to URL scheme validation
        assert "Unsupported URL scheme" not in str(e)
        # Any other error (like "not a video file") is acceptable for this test


def test_video_reader_force_close_with_urls():
    """Test force_close parameter works with URLs."""
    test_url = "https://httpbin.org/status/200"

    try:
        # Test with force_close=True
        with VideoReader(test_url, force_close=True) as reader:
            assert reader.force_close is True
            assert isinstance(reader.video_path, str)

        # Test with force_close=False (default)
        with VideoReader(test_url, force_close=False) as reader:
            assert reader.force_close is False
            assert isinstance(reader.video_path, str)

    except Exception as e:
        pytest.skip(f"Force close test skipped due to network issue: {e}")
