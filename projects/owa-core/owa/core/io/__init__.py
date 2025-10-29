"""
I/O utilities for Open World Agents.

Note: Media loading functions (load_image_as_bgra, load_video_frame_as_bgra, etc.)
have been removed. Use the `mediaref` package instead:
    pip install mediaref[video]
    from mediaref import MediaRef, DataURI
"""

from .image import load_image
from .video import VideoReader, VideoWriter

__all__ = [
    "load_image",
    "VideoReader",
    "VideoWriter",
]
