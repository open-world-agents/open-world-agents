import os
from typing import Literal, overload

import av
import av.container

from ...utils.resource_cache import ResourceCache
from ...utils.typing import PathLike
from .input_container_mixin import InputContainerMixin

DEFAULT_CACHE_SIZE = int(os.environ.get("AV_CACHE_SIZE", 10))

# Global container cache for efficient video file access
_container_cache = ResourceCache(max_size=DEFAULT_CACHE_SIZE)


@overload
def open(
    file: PathLike, mode: Literal["r"], *, keep_av_open: bool = False, **kwargs
) -> av.container.InputContainer: ...


@overload
def open(file: PathLike, mode: Literal["w"], **kwargs) -> av.container.OutputContainer: ...


def open(file: PathLike, mode: Literal["r", "w"], *, keep_av_open: bool = False, **kwargs):
    """
    Open video container with optional caching for read operations.

    Args:
        file: Video file path or URL
        mode: Access mode ('r' for read, 'w' for write)
        keep_av_open: Enable caching for read containers
        **kwargs: Additional arguments passed to av.open
    """
    if mode == "r":
        if not keep_av_open:
            # Direct access without caching
            return av.open(file, "r", **kwargs)

        # Use cached container when keep_av_open=True
        cache_key = str(file)
        if cache_key not in _container_cache:
            return MockedInputContainer(file, **kwargs)
        return _container_cache[cache_key].obj
    else:
        return av.open(file, mode, **kwargs)


def cleanup_cache():
    """Clear all cached video containers from memory."""
    _container_cache.clear()


class MockedInputContainer(InputContainerMixin):
    """Cached wrapper for PyAV InputContainer with reference counting."""

    def __init__(self, file: PathLike, **kwargs):
        self._cache_key = str(file)
        self._container: av.container.InputContainer = av.open(file, "r", **kwargs)
        _container_cache.add_entry(self._cache_key, self)

    def __enter__(self) -> "MockedInputContainer":
        return self

    def close(self):
        """Release container reference and cleanup when no longer needed."""
        _container_cache.release_entry(self._cache_key)


__all__ = ["open", "cleanup_cache"]
