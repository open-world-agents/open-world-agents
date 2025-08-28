import os
from typing import Literal, overload

import av
import av.container

from ..utils.resource_cache import ResourceCache
from ..utils.typing import PathLike

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
            container = av.open(file, "r", **kwargs)
            original_exit = container.__exit__
            container.__exit__ = lambda *args: _container_cache.release_entry(cache_key)
            _container_cache.add_entry(cache_key, container, original_exit)
        return _container_cache[cache_key].obj
    else:
        return av.open(file, mode, **kwargs)


def cleanup_cache():
    """Manually cleanup all cached containers."""
    _container_cache.clear()


__all__ = ["open", "cleanup_cache"]
