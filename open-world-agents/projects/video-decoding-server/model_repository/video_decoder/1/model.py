"""
TODO: hardware-accelerated video frame extraction
TODO: GPU memory response which prevents memory copy across host-device
TODO: batch process for more efficient processing
"""

import gc
import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils


class Logger:
    """Wrapper around Triton's logger since `pb_utils.Logger` is not accesible before `initialize`."""

    def __init__(self):
        self._configured = False

    def configure(self):
        self.logger = pb_utils.Logger
        self._configured = True

    def info(self, message):
        assert self._configured, "Logger is not configured"
        self.logger.log_info(message)

    def warning(self, message):
        assert self._configured, "Logger is not configured"
        self.logger.log_warn(message)

    def error(self, message):
        assert self._configured, "Logger is not configured"
        self.logger.log_error(message)


logger = Logger()


def get_frame_cv2(video_path, time_sec):
    """Extract frame using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise Exception(f"Failed to capture frame at time: {time_sec}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def get_frame_pyav(video_path, time_sec):
    """Extract frame using PyAV."""
    import av

    with av.open(video_path) as container:
        container.seek(int(time_sec * av.time_base), any_frame=False)
        for frame in container.decode(video=0):
            if frame.pts * frame.time_base >= time_sec:
                return np.asarray(frame.to_rgb().to_image())
    raise Exception(f"Failed to capture frame at time: {time_sec}")


def get_frame_torchcodec(video_path, time_sec):
    """Extract frame using TorchCodec."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(video_path, num_ffmpeg_threads=1, device="cuda")
    frame = decoder.get_frame_played_at(time_sec)
    return frame.data.permute(1, 2, 0).cpu().numpy()


class VideoReader:
    """Class responsible for reading video files and extracting frames at specified timestamps."""

    _GC_COLLECT_COUNT = 0
    _GC_COLLECTION_INTERVAL = 10  # Adjust based on memory usage

    def __init__(self, backend="pyav"):
        """Initialize VideoReader with configurable backend."""
        self.backend = backend
        self.backend_func = {
            "cv2": get_frame_cv2,
            "pyav": get_frame_pyav,
            "torchcodec": get_frame_torchcodec,
        }[backend]

    def get_frame_at_time(self, video_path, time_sec):
        """Extract a frame from a video at a specified time."""
        # Only apply GC handling for PyAV backend
        if self.backend == "pyav":
            # Increment GC counter and occasionally run garbage collection
            self._GC_COLLECT_COUNT += 1
            if self._GC_COLLECT_COUNT % self._GC_COLLECTION_INTERVAL == 0:
                # mandatory to prevent thread explosion. if not called, thread is created over 500k for multi-gpu training and the program will crash
                # same logic is implemented in torchvision. https://github.com/pytorch/vision/blob/124dfa404f395db90280e6dd84a51c50c742d5fd/torchvision/io/video.py#L52
                gc.collect()

        return self.backend_func(video_path, time_sec)

    def clear_cache(self):
        """Clear any cached resources."""
        if self.backend == "pyav":
            gc.collect()


class TritonPythonModel:
    """Python model for video frame extraction that efficiently manages GPU memory."""

    def initialize(self, args):
        """
        Initialize the model.
        """
        logger.configure()
        self.model_config = json.loads(args["model_config"])
        self.output_dtype = pb_utils.triton_string_to_numpy(self.model_config["output"][0]["data_type"])

        # Set backend from environment variable
        backend = os.environ.get("VIDEO_BACKEND", "pyav")

        # Initialize the video reader
        self.video_reader = VideoReader(backend)

    def execute(self, requests):
        """
        Process inference requests.
        """
        responses = []

        for request in requests:
            # Get input tensors
            video_path_tensor = pb_utils.get_input_tensor_by_name(request, "video_path")
            time_sec_tensor = pb_utils.get_input_tensor_by_name(request, "time_sec")

            # Convert input tensors to Python types
            video_path = video_path_tensor.as_numpy()[0].decode("utf-8")
            time_sec = float(time_sec_tensor.as_numpy()[0])

            try:
                # Extract frame from video
                frame_array = self.video_reader.get_frame_at_time(video_path, time_sec)

                # Create output tensor
                output_tensor = pb_utils.Tensor("frame", frame_array.astype(self.output_dtype))

                # Create and append response
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

            except Exception as e:
                error = pb_utils.TritonError(str(e))
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=error))
                logger.error(f"Failed to process the request(s), message: {str(e)}")

        return responses

    def finalize(self):
        """
        Clean up resources when the model is unloaded.
        """
        self.video_reader.clear_cache()
