from fractions import Fraction
from pathlib import Path

import numpy as np
from loguru import logger

from owa.core.io.video import VideoReader, VideoWriter

if __name__ == "__main__":
    video_path = Path("test.mp4")

    # Write a test video (VFR)
    with VideoWriter(video_path, fps=60.0, vfr=True) as writer:
        total_frames = 60
        for frame_i in range(total_frames):
            img = np.empty((48, 64, 3), dtype=np.uint8)
            img[:, :, 0] = (0.5 + 0.5 * np.sin(2 * np.pi * (0 / 3 + frame_i / total_frames))) * 255
            img[:, :, 1] = (0.5 + 0.5 * np.sin(2 * np.pi * (1 / 3 + frame_i / total_frames))) * 255
            img[:, :, 2] = (0.5 + 0.5 * np.sin(2 * np.pi * (2 / 3 + frame_i / total_frames))) * 255
            sec = Fraction(frame_i, 60)
            writer.write_frame(img, pts=sec, pts_unit="sec")

    # Write a test video (CFR)
    with VideoWriter(video_path.with_name("test_cfr.mp4"), fps=30.0, vfr=False) as writer_cfr:
        total_frames = 60
        for frame_i in range(total_frames):
            img = np.zeros((48, 64, 3), dtype=np.uint8)
            writer_cfr.write_frame(img)

    # Read back frames from local file
    with VideoReader(video_path) as reader:
        for frame in reader.read_frames(start_pts=Fraction(1, 2)):
            print(f"Local PTS: {frame.pts}, Time: {frame.time}, Shape: {frame.to_ndarray(format='rgb24').shape}")
            break  # Just show first frame
        try:
            frame = reader.read_frame(pts=Fraction(1, 2))
            print(f"Single frame at 0.5s: PTS={frame.pts}, Time={frame.time}")
        except ValueError as e:
            logger.error(e)

    # Example with remote URL (commented out to avoid network dependency in tests)
    # remote_url = "https://open-world-agents.github.io/open-world-agents/data/ocap.mkv"
    # try:
    #     with VideoReader(remote_url) as reader:
    #         frame = reader.read_frame(pts=0.0)
    #         print(f"Remote frame: PTS={frame.pts}, Time={frame.time}")
    # except Exception as e:
    #     logger.error(f"Failed to read remote video: {e}")
