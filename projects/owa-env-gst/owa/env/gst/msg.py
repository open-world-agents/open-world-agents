from fractions import Fraction
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from owa.core.io.video import VideoReader
from owa.core.message import OWAMessage
from owa.core.time import TimeUnits


class ScreenEmitted(OWAMessage):
    _type = "owa.env.gst.msg.ScreenEmitted"

    model_config = {"arbitrary_types_allowed": True}

    # Time since epoch as nanoseconds.
    utc_ns: int | None = None
    # The frame as a numpy array (optional, can be lazy-loaded)
    frame_arr: SkipJsonSchema[Optional[np.ndarray]] = Field(None, exclude=True)
    # Original shape of the frame before rescale, e.g. (width, height)
    original_shape: Optional[Tuple[int, int]] = None
    # Rescaled shape of the frame, e.g. (width, height)
    shape: Optional[Tuple[int, int]] = None

    # Path to the stream, e.g. output.mkv (optional)
    path: str | None = None
    # Time since stream start as nanoseconds.
    pts: int | None = None

    def model_post_init(self, __context):
        # At least one of frame_arr or (path and pts) must be provided
        if self.frame_arr is None:
            if self.path is None or self.pts is None:
                raise ValueError("ScreenEmitted requires either 'frame_arr' or both 'path' and 'pts' to be provided.")
        if self.frame_arr is not None:
            # If frame_arr is provided, set shape and original_shape based on its dimensions
            h, w = self.frame_arr.shape[:2]
            self.shape = (w, h)

    def lazy_load(self, *, force_close: bool = False) -> np.ndarray:
        """
        Lazy load the frame data if not already set.
        This is called when the object is created and frame_arr is None.
        Returns:
            np.ndarray: the frame as a BGRA array.
        """
        if self.frame_arr is None and self.path is not None and self.pts is not None:
            # Convert PTS from nanoseconds to seconds for VideoReader
            pts_seconds = Fraction(self.pts, TimeUnits.SECOND)

            with VideoReader(self.path, force_close=force_close) as reader:
                frame = reader.read_frame(pts=pts_seconds)

                # Convert to RGB first, then to BGRA for consumers
                rgb_array = frame.to_ndarray(format="rgb24")
                self.frame_arr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)

                # Set shape based on the loaded frame (width, height)
                h, w = self.frame_arr.shape[:2]
                shape_tuple = (w, h)
                self.shape = shape_tuple
                self.original_shape = shape_tuple

        return self.frame_arr

    def to_pil_image(self) -> Image.Image:
        """
        Convert the frame at the specified PTS to a PIL Image in RGB format.

        Returns:
            PIL.Image.Image: The frame as a PIL Image.
        """
        rgb_array = self.to_rgb_array()
        return Image.fromarray(rgb_array, mode="RGB")

    def to_rgb_array(self) -> np.ndarray:
        """
        Return the frame as a RGB numpy array.
        If frame_arr is not set, try to load from path and pts.
        """
        # If self.frame_arr is BGRA, convert to RGB.
        bgra_array = self.lazy_load()
        rgb_array = cv2.cvtColor(bgra_array, cv2.COLOR_BGRA2RGB)
        return rgb_array

    def __repr__(self):
        # give concise summary
        ks = ["utc_ns", "shape", "original_shape", "path", "pts"]
        return f"{self.__class__.__name__}({', '.join(f'{k}={getattr(self, k)!r}' for k in ks)})"


def main():
    d = {"path": "output.mkv", "pts": 2683333333, "utc_ns": 1741608540328534500}
    d = {"path": "output.mkv", "pts": int(10**9 * (0.99)), "utc_ns": 1741608540328534500}
    frame = ScreenEmitted(**d)

    print(frame)
    print(frame.to_pil_image())
    print(frame.shape)


if __name__ == "__main__":
    main()
