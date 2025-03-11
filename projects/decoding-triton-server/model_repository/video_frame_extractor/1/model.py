"""
TODO: hardware-accelerated video frame extraction
TODO: GPU memory response which prevents memory copy across host-device
TODO: batch process for more efficient processing
"""

import gc
import json
import os
import threading

import av
import numpy as np
import triton_python_backend_utils as pb_utils


class VideoReader:
    """Class responsible for reading video files and extracting frames at specified timestamps."""

    _GC_COLLECT_COUNT = 0
    _GC_COLLECTION_INTERVAL = 10  # Adjust based on memory usage

    _video_container_cache = {}
    _cache_lock = threading.Lock()
    _max_cache_size = 4  # Default maximum number of cached containers

    def __init__(self, max_cache_size=None):
        """Initialize VideoReader with an optional cache size."""
        if max_cache_size is not None:
            self._max_cache_size = max_cache_size

    def get_frame_at_time(self, video_path, time_sec):
        """
        Extract a frame from a video at a specified time.
        """
        # Increment GC counter and occasionally run garbage collection
        self._GC_COLLECT_COUNT += 1
        if self._GC_COLLECT_COUNT % self._GC_COLLECTION_INTERVAL == 0:
            # mandatory to prevent thread explosion. if not called, thread is created over 500k for multi-gpu training and the program will crash
            # same logic is implemented in torchvision. https://github.com/pytorch/vision/blob/124dfa404f395db90280e6dd84a51c50c742d5fd/torchvision/io/video.py#L52
            gc.collect()

        # Get the video container from cache or open a new one
        container = self._get_video_container(video_path)

        # Seek to the specified time
        container.seek(int(time_sec * av.time_base), any_frame=False)

        # Decode and find the frame
        for frame in container.decode(video=0):
            if frame.pts * frame.time_base >= time_sec:
                return frame.to_rgb().to_image()

        raise Exception(f"Failed to capture frame at time: {time_sec}")

    def _get_video_container(self, video_path):
        """
        Get a video container from cache or create a new one.
        Thread-safe implementation with size limiting.
        """
        with self._cache_lock:
            # Check if it's already cached
            if video_path in self._video_container_cache:
                return self._video_container_cache[video_path]

            # If cache is full, remove the oldest entry
            if len(self._video_container_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._video_container_cache))
                del self._video_container_cache[oldest_key]

            # Open a new container and add it to the cache
            container = av.open(video_path)
            self._video_container_cache[video_path] = container
            return container

    def clear_cache(self):
        """Close and clear all cached video containers."""
        with self._cache_lock:
            for container in self._video_container_cache.values():
                container.close()
            self._video_container_cache.clear()
        gc.collect()


class TritonPythonModel:
    """Python model for video frame extraction that efficiently manages GPU memory."""

    def initialize(self, args):
        """
        Initialize the model.
        """
        self.model_config = json.loads(args["model_config"])
        self.output_dtype = pb_utils.triton_string_to_numpy(self.model_config["output"][0]["data_type"])

        # Set cache size from environment variable if present
        max_cache_size = int(os.environ.get("MAX_CACHE_SIZE", "50"))

        # Initialize the video reader
        self.video_reader = VideoReader(max_cache_size)

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
                frame = self.video_reader.get_frame_at_time(video_path, time_sec)
                frame_array = np.asarray(frame)

                # Create output tensor
                output_tensor = pb_utils.Tensor("frame", frame_array.astype(self.output_dtype))

                # Create and append response
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

            except Exception as e:
                error = pb_utils.TritonError(str(e))
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=error))

        return responses

    def finalize(self):
        """
        Clean up resources when the model is unloaded.
        """
        self.video_reader.clear_cache()
