#!/usr/bin/env python3
"""
Simple client for video frame extraction using Triton Inference Server.
"""

import argparse
import sys
from typing import Tuple

import cv2
import numpy as np
import tritonclient.http as httpclient


def extract_frame(requests: Tuple[str, float], server_url: str = "127.0.0.1:8000") -> np.ndarray:
    """
    Extract a frame from video at specified time.

    Args:
        requests: Single (video_path, time_sec) tuple
        server_url: Triton server URL

    Returns:
        Frame as numpy array (H, W, 3)
    """
    client = httpclient.InferenceServerClient(url=server_url)

    video_path, time_sec = requests

    inputs = [
        httpclient.InferInput("video_path", [1, 1], "BYTES"),
        httpclient.InferInput("time_sec", [1, 1], "FP32"),
    ]

    # Convert to numpy arrays
    video_path_data = np.array([[video_path.encode()]], dtype=np.object_)
    time_sec_data = np.array([[time_sec]], dtype=np.float32)

    inputs[0].set_data_from_numpy(video_path_data)
    inputs[1].set_data_from_numpy(time_sec_data)

    outputs = [httpclient.InferRequestedOutput("frame")]
    response = client.infer("video_decoder", inputs=inputs, outputs=outputs)

    frames = response.as_numpy("frame")
    if frames is None:
        raise RuntimeError("Failed to extract frames from server response")

    return frames[0]


def main():
    parser = argparse.ArgumentParser(description="Extract frame(s) from video(s)")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("time", type=float, help="Time in seconds")
    parser.add_argument("--server-url", default="127.0.0.1:8000", help="Triton server URL")
    parser.add_argument("--output", "-o", help="Output image path (optional)")

    args = parser.parse_args()

    try:
        frame = extract_frame((args.video, args.time), args.server_url)

        print(f"Extracted frame shape: {frame.shape}")
        print(f"Frame dtype: {frame.dtype}")
        if args.output:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(args.output, frame_bgr)
            print(f"Frame saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
