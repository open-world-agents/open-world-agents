"""
Legacy media utilities - DEPRECATED. Code here moved to `mediaref` package.
"""

import os

import numpy as np

VIDEO_DECODING_SERVER_URL = os.environ.get("VIDEO_DECODING_SERVER_URL")  # e.g. 127.0.0.1:8000


# ============================================================================
# Triton Decoding Server Support (kept for backward compatibility)
# ============================================================================


def extract_frame(video_path: str, time_sec: float, server_url: str = "127.0.0.1:8000") -> np.ndarray:
    """
    Extract a frame from video at specified time using Triton server.

    DEPRECATED: This function is kept only for backward compatibility.
    Consider using mediaref package for new code.

    Args:
        video_path: Path to video file
        time_sec: Time in seconds
        server_url: Triton server URL

    Returns:
        Frame as numpy array (H, W, 3)
    """
    import tritonclient.http as httpclient

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
