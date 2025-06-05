import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from owa.core.io.video import VideoReader, VideoWriter


def test_video_writer_vfr():
    """Test Variable Frame Rate (VFR) video writing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / "test_vfr.mp4"

        with VideoWriter(video_path, fps=60.0, vfr=True) as writer:
            total_frames = 60
            for frame_i in range(total_frames):
                img = np.empty((48, 64, 3), dtype=np.uint8)
                img[:, :, 0] = (0.5 + 0.5 * np.sin(2 * np.pi * (0 / 3 + frame_i / total_frames))) * 255
                img[:, :, 1] = (0.5 + 0.5 * np.sin(2 * np.pi * (1 / 3 + frame_i / total_frames))) * 255
                img[:, :, 2] = (0.5 + 0.5 * np.sin(2 * np.pi * (2 / 3 + frame_i / total_frames))) * 255
                sec = Fraction(frame_i, 60)
                writer.write_frame(img, pts=sec, pts_unit="sec")

        assert video_path.exists(), "VFR video file should be created"


def test_video_writer_cfr():
    """Test Constant Frame Rate (CFR) video writing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / "test_cfr.mp4"

        with VideoWriter(video_path, fps=30.0, vfr=False) as writer:
            total_frames = 60
            for frame_i in range(total_frames):
                img = np.zeros((48, 64, 3), dtype=np.uint8)
                writer.write_frame(img)

        assert video_path.exists(), "CFR video file should be created"


def test_video_reader():
    """Test video reading functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / "test_read.mp4"

        # First create a test video
        with VideoWriter(video_path, fps=60.0, vfr=True) as writer:
            total_frames = 60
            for frame_i in range(total_frames):
                img = np.empty((48, 64, 3), dtype=np.uint8)
                img[:, :, 0] = (0.5 + 0.5 * np.sin(2 * np.pi * (0 / 3 + frame_i / total_frames))) * 255
                img[:, :, 1] = (0.5 + 0.5 * np.sin(2 * np.pi * (1 / 3 + frame_i / total_frames))) * 255
                img[:, :, 2] = (0.5 + 0.5 * np.sin(2 * np.pi * (2 / 3 + frame_i / total_frames))) * 255
                sec = Fraction(frame_i, 60)
                writer.write_frame(img, pts=sec, pts_unit="sec")

        # Now read it back
        with VideoReader(video_path, force_close=True) as reader:
            frame_count = 0
            for frame in reader.read_frames(start_pts=Fraction(1, 2)):
                frame_array = frame.to_ndarray(format="rgb24")
                assert frame_array.shape == (48, 64, 3), "Frame should have correct dimensions"
                assert frame.pts is not None, "Frame should have PTS"
                assert frame.time is not None, "Frame should have time"
                frame_count += 1
                if frame_count >= 10:  # Limit frames to avoid long test
                    break

            assert frame_count > 0, "Should read at least one frame"

            # Test reading a single frame
            single_frame = reader.read_frame(pts=Fraction(1, 2))
            assert single_frame is not None, "Should be able to read single frame"
            assert single_frame.pts is not None, "Single frame should have PTS"
            assert single_frame.time is not None, "Single frame should have time"


def test_video_processing_pipeline():
    """Test a complete video processing pipeline (read and write)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input.mp4"
        output_path = Path(temp_dir) / "output.mp4"

        # Create input video
        with VideoWriter(input_path, fps=30.0, vfr=False) as writer:
            for frame_i in range(30):  # 1 second of video at 30fps
                img = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
                writer.write_frame(img)

        # Process video (read and write)
        with (
            VideoReader(input_path, force_close=True) as reader,
            VideoWriter(output_path, fps=30.0) as writer,
        ):
            frame_count = 0
            for frame in reader.read_frames():
                frame_array = frame.to_ndarray(format="rgb24")
                writer.write_frame(frame_array)
                frame_count += 1
                if frame_count >= 15:  # Process only half the frames
                    break

        assert output_path.exists(), "Output video should be created"
        assert frame_count == 15, "Should process exactly 15 frames"
