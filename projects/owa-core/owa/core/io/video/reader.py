import enum
import gc
from dataclasses import dataclass
from fractions import Fraction
from typing import Generator, Optional

import av

from ...utils.typing import PathLike
from .. import cached_av
from .typing import SECOND_TYPE

# Garbage collection counters for PyAV reference cycles
# Reference: https://github.com/pytorch/vision/blob/428a54c96e82226c0d2d8522e9cbfdca64283da0/torchvision/io/video.py#L53-L55
_CALLED_TIMES = 0
GC_COLLECTION_INTERVAL = 10


@dataclass
class VideoStreamMetadata:
    """Video stream metadata container."""

    num_frames: int
    duration_seconds: Fraction
    average_rate: Fraction
    width: int
    height: int


class BatchDecodingStrategy(enum.StrEnum):
    SEPARATE = "separate"  # Decode each frame separately. Best at sparse query.
    SEQUENTIAL_PER_KEYFRAME_BLOCK = "sequential_per_keyframe_block"  # Decode frames in batches per keyframe block. Better at dense query then separate, better at sparse query then sequential.
    SEQUENTIAL = "sequential"  # Decode frames in batches. Best at dense query.


class VideoReader:
    """PyAV-based video reader with caching support for local files and URLs."""

    def __init__(self, video_path: PathLike, *, keep_av_open: bool = False):
        """
        Initialize video reader.

        Args:
            video_path: Input video file path or URL (HTTP/HTTPS)
            keep_av_open: Keep AV container open in cache
        """
        self.video_path = video_path
        self.container = cached_av.open(self.video_path, "r", keep_av_open=keep_av_open)
        self._metadata = self._extract_metadata()

    def _extract_metadata(self) -> VideoStreamMetadata:
        """Extract video stream metadata from container."""
        container = self.container
        if not container.streams.video:
            raise ValueError(f"No video streams found in {self.video_path}")
        stream = container.streams.video[0]

        # Determine video duration
        if stream.duration and stream.time_base:
            duration_seconds = stream.duration * stream.time_base
        elif container.duration:
            duration_seconds = container.duration * Fraction(1, av.time_base)
        else:
            raise ValueError("Failed to determine duration")

        # Determine frame rate
        if stream.average_rate:
            average_rate = stream.average_rate
        else:
            raise ValueError("Failed to determine average rate")

        # Determine frame count
        if stream.frames:
            num_frames = stream.frames
        else:
            num_frames = int(duration_seconds * average_rate)

        return VideoStreamMetadata(
            num_frames=num_frames,
            duration_seconds=duration_seconds,
            average_rate=average_rate,
            width=stream.width,
            height=stream.height,
        )

    @property
    def metadata(self) -> VideoStreamMetadata:
        """Access video stream metadata."""
        return self._metadata

    def read_frames(
        self, start_pts: SECOND_TYPE = 0.0, end_pts: Optional[SECOND_TYPE] = None, fps: Optional[float] = None
    ) -> Generator[av.VideoFrame, None, None]:
        """Yield frames between start_pts and end_pts in seconds."""
        global _CALLED_TIMES
        _CALLED_TIMES += 1
        if _CALLED_TIMES % GC_COLLECTION_INTERVAL == 0:
            gc.collect()

        # Handle negative end_pts (Python-style indexing)
        if end_pts is not None and float(end_pts) < 0:
            if self.container.duration is None:
                raise ValueError("Video duration unavailable for negative end_pts")
            duration = self.container.duration / av.time_base
            end_pts = duration + float(end_pts)

        end_pts = float(end_pts) if end_pts is not None else float("inf")

        if fps is None:
            # Yield all frames in interval
            yield from self._yield_frame_range(float(start_pts), end_pts)
        else:
            # Sample at specified fps
            if fps <= 0:
                raise ValueError("fps must be positive")
            yield from self._yield_frame_rated(float(start_pts), end_pts, fps)

    def get_frames_played_at(
        self,
        seconds: list[float],
        *,
        strategy: BatchDecodingStrategy = BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK,
    ) -> list[av.VideoFrame]:
        """Return frames at specific time points using keyframe-aware reading."""
        if not seconds:
            return []

        if max(seconds) > self.metadata.duration_seconds:
            raise ValueError(f"Requested time {max(seconds)}s exceeds video duration {self.metadata.duration_seconds}")

        # Decode each frame separately
        if strategy == BatchDecodingStrategy.SEPARATE:
            return [self.read_frame(pts=s) for s in seconds]

        queries = sorted([(s, i) for i, s in enumerate(seconds)])
        frames: list[av.VideoFrame] = [None] * len(queries)  # type: ignore

        # Read all frames in one go
        if strategy == BatchDecodingStrategy.SEQUENTIAL:
            start_pts = queries[0][0]
            found = 0

            for frame in self.read_frames(start_pts):  # do not specify end_pts to avoid early termination
                while found < len(queries) and frame.time >= queries[found][0]:
                    frames[queries[found][1]] = frame
                    found += 1
                if found >= len(queries):
                    break

        # Restart-on-keyframe logic:
        #    This method uses a two-loop approach to read frames in segments:
        #    - Outer loop: Manages seeking to different video segments
        #    - Inner loop: Reads frames until keyframe is detected or all targets found
        #    This approach is efficient both for inter-GOP and intra-GOP queries.
        elif strategy == BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK:
            query_idx = 0

            # Outer loop: restart/resume for each segment
            while query_idx < len(queries):
                target_time = queries[query_idx][0]
                first_keyframe_seen = False
                query_idx_before_segment = query_idx

                # Inner loop: read frames until keyframe detected or all targets found
                for frame in self.read_frames(start_pts=target_time):
                    # Track keyframes
                    if frame.key_frame:
                        if first_keyframe_seen:
                            # Hit second keyframe - stop segment
                            break
                        first_keyframe_seen = True

                    # Match frames to queries in this segment
                    while query_idx < len(queries) and frame.time >= queries[query_idx][0]:
                        frames[queries[query_idx][1]] = frame
                        query_idx += 1

                    # Stop condition for inner loop
                    if query_idx >= len(queries):
                        # Found all remaining frames
                        break

                # If no progress made in inner loop, raise error. It's to prevent infinite loops.
                if query_idx_before_segment == query_idx:
                    raise ValueError(
                        f"No matching frames found for query starting at {target_time:.3f}s. This may indicate a corrupted video file or a decoding issue."
                    )

        if any(f is None for f in frames):
            missing_seconds = [s for i, s in enumerate(seconds) if frames[i] is None]
            raise ValueError(f"Could not find frames for the following timestamps: {missing_seconds}")

        return frames

    def _yield_frame_range(self, start_pts, end_pts):
        """Yield all frames in time range."""
        # Seek to start position
        timestamp_ts = int(av.time_base * start_pts)
        # NOTE: seek with anyframe=False must present before decoding to ensure flawless decoding
        self.container.seek(timestamp_ts, any_frame=False)

        # Yield frames in interval
        for frame in self.container.decode(video=0):
            if frame.time is None:
                raise ValueError("Frame time is None")
            if frame.time < start_pts:
                continue
            if frame.time > end_pts:
                break
            yield frame

    def _yield_frame_rated(self, start_pts, end_pts, fps):
        """Yield frames sampled at specified fps with proper VFR gap handling."""
        interval = 1.0 / fps
        next_time = start_pts

        for frame in self._yield_frame_range(start_pts, end_pts):
            if frame.time < next_time:
                continue
            yield frame
            next_time += interval

    def read_frame(self, pts: SECOND_TYPE = 0.0) -> av.VideoFrame:
        """Read single frame at or after given timestamp."""
        for frame in self.read_frames(start_pts=pts, end_pts=None):
            return frame
        raise ValueError(f"Frame not found at {float(pts):.2f}s in {self.video_path}")

    def close(self) -> None:
        """Release container reference."""
        self.container.close()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
