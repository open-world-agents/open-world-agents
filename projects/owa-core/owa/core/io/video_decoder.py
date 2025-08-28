"""Clean and minimal VideoDecoder that mocks torchcodec's behavior."""

import dataclasses
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Union

import av
import numpy as np
import numpy.typing as npt

from owa.core.utils.typing import PathLike

from .video import VideoReader


@dataclass
class VideoStreamMetadata:
    """Video metadata."""

    num_frames: int
    duration_seconds: Fraction
    average_rate: Fraction
    width: int
    height: int


# Copied from https://github.com/pytorch/torchcodec/blob/main/src/torchcodec/_frame.py#L14-L27
def _frame_repr(self):
    # Utility to replace __repr__ method of dataclasses below. This prints the
    # shape of the .data tensor rather than printing the (potentially very long)
    # data tensor itself.
    s = self.__class__.__name__ + ":\n"
    spaces = "  "
    for field in dataclasses.fields(self):
        field_name = field.name
        field_val = getattr(self, field_name)
        if field_name == "data":
            field_name = "data (shape)"
            field_val = field_val.shape
        s += f"{spaces}{field_name}: {field_val}\n"
    return s


@dataclass
class FrameBatch:
    """Frame batch with timing info."""

    data: npt.NDArray[np.uint8]  # [N, C, H, W]
    pts_seconds: npt.NDArray[np.float64]  # [N]
    duration_seconds: npt.NDArray[np.float64]  # [N]

    __repr__ = _frame_repr


class VideoDecoder:
    """Minimal VideoDecoder using existing VideoReader."""

    def __init__(self, video_path: PathLike):
        self.video_path = video_path
        self._reader = VideoReader(video_path, keep_av_open=True)
        self._metadata = self._extract_metadata()

    def _extract_metadata(self) -> VideoStreamMetadata:
        """Extract basic metadata."""
        container = self._reader.container
        stream = container.streams.video[0]

        # Get basic properties
        if stream.duration and stream.time_base:
            duration_seconds = stream.duration * stream.time_base
        elif container.duration:
            duration_seconds = container.duration * Fraction(1, av.time_base)
        else:
            raise ValueError("Failed to determine duration")

        if stream.average_rate:
            average_rate = stream.average_rate
        else:
            raise ValueError("Failed to determine average rate")

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
        return self._metadata

    def __getitem__(self, key: Union[int, slice]) -> npt.NDArray[np.uint8]:
        """Simple indexing: decoder[0] or decoder[0:10:2]."""
        if isinstance(key, int):
            # Single frame
            if key < 0:
                key = self.metadata.num_frames + key
            pts = key / self.metadata.average_rate
            frame = self._reader.read_frame(pts=pts)
            frame_rgb = frame.to_ndarray(format="rgb24")  # [H, W, C]
            return np.transpose(frame_rgb, (2, 0, 1)).astype(np.uint8)  # [C, H, W]

        elif isinstance(key, slice):
            # Multiple frames
            start, stop, step = key.indices(self.metadata.num_frames)
            indices = list(range(start, stop, step))
            return self.get_frames_at(indices).data

        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def get_frames_at(self, indices: List[int], *, strategy: str = "sequential_per_keyframe_block") -> FrameBatch:
        """Get frames at specific indices."""

        # TODO: much more efficient implementation
        pts = [idx / self.metadata.average_rate for idx in indices]
        return self.get_frames_played_at(seconds=pts, strategy=strategy)

    def get_frames_played_at(
        self, seconds: List[float], *, strategy: str = "sequential_per_keyframe_block"
    ) -> FrameBatch:
        """Get frames at specific time points."""
        frames = []
        pts_list = []
        duration = 1.0 / self.metadata.average_rate

        av_frames = self._reader.get_frames_played_at(seconds, strategy=strategy)
        frames = [frame.to_ndarray(format="rgb24") for frame in av_frames]
        frames = [np.transpose(frame, (2, 0, 1)).astype(np.uint8) for frame in frames]
        pts_list = [frame.time for frame in av_frames]

        return FrameBatch(
            data=np.stack(frames, axis=0),  # [N, C, H, W]
            pts_seconds=np.array(pts_list, dtype=np.float64),
            duration_seconds=np.full(len(seconds), duration, dtype=np.float64),
        )

    def close(self):
        """Clean up resources."""
        if hasattr(self, "_reader"):
            self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
