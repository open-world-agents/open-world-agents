import atexit
import gc
import os
import sys
import threading
import time
from typing import Literal, Optional, overload

import av
import av.container
from loguru import logger

from ...utils.resource_cache import ResourceCache
from ...utils.typing import PathLike
from .input_container_mixin import InputContainerMixin

DEFAULT_CACHE_SIZE = int(os.environ.get("AV_CACHE_SIZE", 10))

# Global cache instance
_container_cache = ResourceCache(max_size=DEFAULT_CACHE_SIZE)


@overload
def open(
    file: PathLike, mode: Literal["r"], *, keep_av_open: bool = False, **kwargs
) -> av.container.InputContainer: ...


@overload
def open(file: PathLike, mode: Literal["w"], **kwargs) -> av.container.OutputContainer: ...


def open(file: PathLike, mode: Literal["r", "w"], *, keep_av_open: bool = False, **kwargs):
    """Open video file with caching for read mode, direct av.open for write mode.

    Args:
        file: Path to video file
        mode: Open mode ('r' for read, 'w' for write)
        keep_av_open: If True, keep container in cache when closed. If False, force cleanup.
        **kwargs: Additional arguments passed to av.open
    """
    if mode == "r":
        if not keep_av_open:
            # Don't cache if keep_av_open=False - just return direct av.open
            return av.open(file, "r", **kwargs)

        # Cache only when keep_av_open=True
        cache_key = str(file)
        if cache_key not in _container_cache:
            return MockedInputContainer(file, **kwargs)
        return _container_cache[cache_key].obj
    else:
        return av.open(file, mode, **kwargs)


def cleanup_cache():
    """Manually cleanup all cached containers."""
    _container_cache.clear()


class MockedInputContainer(InputContainerMixin):
    """Wrapper for av.InputContainer that tracks references and usage for caching."""

    def __init__(self, file: PathLike, **kwargs):
        self._cache_key = str(file)
        self._container: av.container.InputContainer = av.open(file, "r", **kwargs)
        _container_cache.add_entry(self._cache_key, self)

    def __enter__(self) -> "MockedInputContainer":
        return self

    def close(self):
        """Decrement reference count and cleanup if no longer referenced."""
        _container_cache.release_entry(self._cache_key)


__all__ = ["open", "cleanup_cache"]
