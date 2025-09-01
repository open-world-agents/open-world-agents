"""PyAV-based video decoder with TorchCodec-compatible interface."""

import dataclasses
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import numpy.typing as npt

from ...utils.typing import PathLike
from ..video.reader import BatchDecodingStrategy, VideoReader, VideoStreamMetadata


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
    """Batch of video frames with timing information in NCHW format."""

    data: npt.NDArray[np.uint8]  # [N, C, H, W]
    pts_seconds: npt.NDArray[np.float64]  # [N]
    duration_seconds: npt.NDArray[np.float64]  # [N]

    __repr__ = _frame_repr


class PyAVVideoDecoder:
    """TorchCodec-compatible video decoder built on PyAV."""

    def __init__(self, video_path: PathLike):
        self.video_path = video_path
        self._reader = VideoReader(video_path, keep_av_open=True)

    @property
    def metadata(self) -> VideoStreamMetadata:
        """Access video stream metadata."""
        return self._reader.metadata

    def __getitem__(self, key: Union[int, slice]) -> npt.NDArray[np.uint8]:
        """Enable array-like indexing for frame access."""
        if isinstance(key, int):
            return self.get_frames_at([key]).data

        elif isinstance(key, slice):
            start, stop, step = key.indices(self.metadata.num_frames)
            indices = list(range(start, stop, step))
            return self.get_frames_at(indices).data

        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def get_frames_at(
        self,
        indices: List[int],
        *,
        strategy: BatchDecodingStrategy = BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK,
    ) -> FrameBatch:
        """Retrieve frames at specific frame indices."""
        indices = [index % self.metadata.num_frames for index in indices]
        pts = [idx / self.metadata.average_rate for idx in indices]
        return self.get_frames_played_at(seconds=pts, strategy=strategy)

    def get_frames_played_at(
        self,
        seconds: List[float],
        *,
        strategy: BatchDecodingStrategy = BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK,
    ) -> FrameBatch:
        """Retrieve frames at specific timestamps."""
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
        """Release video decoder resources."""
        if hasattr(self, "_reader"):
            self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
