#!/usr/bin/env python3
"""
Simple client for video frame extraction using Triton Inference Server.
"""

import argparse
import sys

import cv2
import numpy as np
import tritonclient.http as httpclient


def extract_frame(video_path: str, time_sec: float, server_url: str = "127.0.0.1:8000") -> np.ndarray:
    """
    Extract a frame from video at specified time.

    Args:
        video_path: Path to video file
        time_sec: Time in seconds
        server_url: Triton server URL

    Returns:
        Frame as numpy array (H, W, 3)
    """
    client = httpclient.InferenceServerClient(url=server_url)

    inputs = [httpclient.InferInput("video_path", [1], "BYTES"), httpclient.InferInput("time_sec", [1], "FP32")]
    inputs[0].set_data_from_numpy(np.array([str(video_path).encode()], dtype=np.object_))
    inputs[1].set_data_from_numpy(np.array([time_sec], dtype=np.float32))

    outputs = [httpclient.InferRequestedOutput("frame")]
    response = client.infer("video_decoder", inputs=inputs, outputs=outputs)

    frame = response.as_numpy("frame")
    if frame is None:
        raise RuntimeError("Failed to extract frame from server response")

    return frame


def main():
    parser = argparse.ArgumentParser(description="Extract frame from video")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("time", type=float, help="Time in seconds")
    parser.add_argument("--server-url", default="127.0.0.1:8000", help="Triton server URL")
    parser.add_argument("--output", "-o", help="Output image path (optional)")

    args = parser.parse_args()

    try:
        frame = extract_frame(args.video, args.time, args.server_url)

        if args.output:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(args.output, frame_bgr)
            print(f"Frame saved to: {args.output}")
        else:
            print(f"Extracted frame shape: {frame.shape}")
            print(f"Frame dtype: {frame.dtype}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
