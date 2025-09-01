import gc
from fractions import Fraction
from typing import Any, Dict, Optional, Union

import av
import numpy as np

from ...utils.typing import PathLike
from .typing import SECOND_TYPE, PTSUnit

# Frame duplicate detection tolerance
DUPLICATE_TOLERANCE_SECOND: Fraction = Fraction(1, 120)


# Garbage collection counters for PyAV reference cycles
# Reference: https://github.com/pytorch/vision/blob/428a54c96e82226c0d2d8522e9cbfdca64283da0/torchvision/io/video.py#L53-L55
_CALLED_TIMES = 0
GC_COLLECTION_INTERVAL = 10


class VideoWriter:
    """VideoWriter uses PyAV to write video frames with VFR/CFR support.

    References:
        - https://stackoverflow.com/questions/65213302/how-to-write-variable-frame-rate-videos-in-python
        - https://github.com/PyAV-Org/PyAV/blob/main/examples/numpy/generate_video_with_pts.py
        - Design Reference: https://pytorch.org/vision/stable/generated/torchvision.io.read_video.html
    """

    def __init__(self, video_path: PathLike, fps: Optional[float] = None, vfr: bool = False, **kwargs):
        """
        Initialize video writer.

        Args:
            video_path: Output video file path
            fps: Frames per second (required for CFR or when pts not provided)
            vfr: Use Variable Frame Rate
            **kwargs: Additional codec parameters
        """
        self.video_path = video_path
        self.fps = fps
        self.vfr = vfr
        self._closed = False
        self.past_pts = None

        # Setup codec parameters
        self.codec_params = {"gop_size": kwargs.get("gop_size", 30)}

        # Initialize container and stream
        self.container = av.open(str(video_path), mode="w")
        self._setup_stream()

    def _setup_stream(self):
        """Configure video stream for VFR or CFR."""
        if self.vfr:
            self.stream = self.container.add_stream("h264", rate=-1)
            self._time_base = Fraction(1, 60000)  # Fine-grained for VFR
        else:
            if not self.fps or self.fps <= 0:
                raise ValueError("fps must be positive for CFR (vfr=False)")
            self.stream = self.container.add_stream("h264", rate=int(self.fps))
            self._time_base = Fraction(1, int(self.fps))

        # Apply settings
        self.stream.pix_fmt = "yuv420p"
        self.stream.time_base = self._time_base
        self.stream.codec_context.time_base = self._time_base
        for key, value in self.codec_params.items():
            setattr(self.stream.codec_context, key, value)

    def write_frame(
        self,
        frame: Union[av.VideoFrame, np.ndarray],
        pts: Optional[Union[int, SECOND_TYPE]] = None,
        pts_unit: PTSUnit = "pts",
    ) -> Dict[str, Any]:
        """Write frame to video with optional timestamp."""
        global _CALLED_TIMES
        _CALLED_TIMES += 1
        if _CALLED_TIMES % GC_COLLECTION_INTERVAL == 0:
            gc.collect()

        # Convert numpy to VideoFrame
        if isinstance(frame, np.ndarray):
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        elif not isinstance(frame, av.VideoFrame):
            raise TypeError("frame must be av.VideoFrame or np.ndarray")

        # Calculate PTS
        pts_as_pts = self._calculate_pts(pts, pts_unit)
        pts_as_sec = self.pts_to_sec(pts_as_pts)

        # Skip duplicate frames
        if self._is_duplicate(pts_as_pts):
            return {"source": str(self.video_path), "timestamp": float(pts_as_sec)}

        # Write frame
        frame.pts = pts_as_pts
        self.stream.width = frame.width
        self.stream.height = frame.height

        for packet in self.stream.encode(frame):
            self.container.mux(packet)

        self.past_pts = pts_as_pts
        return {"source": str(self.video_path), "timestamp": float(pts_as_sec)}

    def _calculate_pts(self, pts, pts_unit):
        """Calculate PTS value in pts units."""
        if pts is None:
            if not self.fps:
                raise ValueError("fps required when pts not provided")
            if self.past_pts is None:
                return 0
            return self.past_pts + self.sec_to_pts(Fraction(1, int(self.fps)))

        if pts_unit == "pts":
            if not isinstance(pts, int):
                raise TypeError("pts must be int when pts_unit is 'pts'")
            return pts
        elif pts_unit == "sec":
            if not isinstance(pts, (float, Fraction)):
                raise TypeError("pts must be float/Fraction when pts_unit is 'sec'")
            return self.sec_to_pts(pts)
        else:
            raise ValueError(f"Invalid pts_unit: {pts_unit}")

    def _is_duplicate(self, pts_as_pts):
        """Check if frame is duplicate within tolerance."""
        return self.past_pts is not None and pts_as_pts - self.past_pts < self.sec_to_pts(DUPLICATE_TOLERANCE_SECOND)

    def pts_to_sec(self, pts: int) -> Fraction:
        return pts * self.stream.codec_context.time_base

    def sec_to_pts(self, sec: SECOND_TYPE) -> int:
        if not isinstance(sec, (float, Fraction)):
            raise TypeError("sec must be numeric")
        return int(sec / self.stream.codec_context.time_base)

    def close(self) -> None:
        """Finalize and close the container."""
        if self._closed:
            return

        # Flush encoder
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self._closed = True

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
