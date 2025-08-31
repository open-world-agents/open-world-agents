"""
Video I/O utilities using PyAV.

Provides VideoReader and VideoWriter classes for local files and remote URLs
with support for variable and constant frame rate encoding/decoding.
"""

from .reader import BatchDecodingStrategy, VideoReader
from .typing import SECOND_TYPE, PTSUnit
from .writer import VideoWriter

__all__ = ["BatchDecodingStrategy", "VideoReader", "VideoWriter", "SECOND_TYPE", "PTSUnit"]
