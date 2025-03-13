from typing import Optional

import av
import numpy as np
from PIL import Image

from owa.core.message import OWAMessage


class ScreenEmitted(OWAMessage):
    _type = "owa.env.gst.msg.ScreenEmitted"

    # Path to the stream, e.g. output.mkv
    path: str
    # Time since stream start as nanoseconds.
    pts: int
    # Time since epoch as nanoseconds.
    utc_ns: int

    _frame_arr: Optional[np.ndarray] = None

    def to_pil_image(self) -> Image.Image:
        """
        Convert the frame at the specified PTS to a PIL Image in RGBA format.

        Returns:
            PIL.Image.Image: The frame as a PIL Image.
        """
        bgra_array = self.to_bgra_array()
        # Convert BGRA to RGB by rearranging the channels
        rgb_array = bgra_array[..., [2, 1, 0]]
        return Image.fromarray(rgb_array, mode="RGB")

    def to_bgra_array(self) -> np.ndarray:
        """
        Extract the frame at the specified PTS and return it as a BGRA NumPy array.

        Returns:
            np.ndarray: The frame as a BGRA array with shape (H, W, 4).

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If a frame at the specified PTS cannot be found.
        """
        if self._frame_arr is not None:
            return self._frame_arr

        try:
            container = av.open(self.path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Video file not found: {self.path}")
        except av.AVError as e:
            raise ValueError(f"Error opening video file: {e}")

        try:
            # Select the first video stream
            stream = next(s for s in container.streams if s.type == "video")
        except StopIteration:
            container.close()
            raise ValueError("No video stream found in the file.")

        # Convert PTS from nanoseconds to seconds
        target_time = self.pts / 1e9

        # Seek to the target timestamp
        # Calculate the seek position in terms of stream time base
        seek_timestamp = int(target_time / stream.time_base)

        # Seek to the nearest keyframe before the target time
        container.seek(seek_timestamp, any_frame=False, backward=True, stream=stream)

        frame_found = False

        for frame in container.decode(stream):
            frame_time = frame.pts * stream.time_base
            if frame_time >= target_time:
                # Convert frame to BGRA format
                bgra_frame = frame.to_ndarray(format="bgra")
                self._frame_arr = bgra_frame
                frame_found = True
                break

        container.close()

        if not frame_found:
            raise ValueError(f"No frame found at PTS: {self.pts} ns")

        return self._frame_arr


class FrameStamped(OWAMessage):
    _type = "owa.env.gst.msg.FrameStamped"

    model_config = {"arbitrary_types_allowed": True}

    timestamp_ns: int
    frame_arr: np.ndarray  # [W, H, BGRA]


def main():
    d = {"path": "output.mkv", "pts": 2683333333, "utc_ns": 1741608540328534500}
    d = {"path": "output.mkv", "pts": 10**9 * (0.99), "utc_ns": 1741608540328534500}
    frame = ScreenEmitted(**d)

    print(frame)
    print(frame.to_pil_image())


if __name__ == "__main__":
    main()
