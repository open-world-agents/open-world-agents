import atexit
import gc
import os
import sys
import threading
import time
from pathlib import Path
from typing import Literal, Optional, Union, overload

import av
import av.container
from loguru import logger

from .input_container_mixin import InputContainerMixin

DEFAULT_CACHE_SIZE = int(os.environ.get("AV_CACHE_SIZE", 10))

VideoPathType = Union[str, os.PathLike, Path]


class _CacheContext:
    """Context manager for thread-safe access to video container cache."""

    def __init__(self):
        self._cache: dict[VideoPathType, "MockedInputContainer"] = {}
        self._lock = threading.RLock()

    def __enter__(self) -> dict[VideoPathType, "MockedInputContainer"]:
        """Enter context manager and return locked cache."""
        self._lock.acquire()
        return self._cache

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and release lock."""
        self._lock.release()
        return False  # Don't suppress exceptions


# Global cache context instance
_cache_context = _CacheContext()


def get_cache_context():
    """Get cache context for atomic operations. Use with 'with' statement."""
    return _cache_context


@overload
def open(file: VideoPathType, mode: Literal["r"], **kwargs) -> "MockedInputContainer": ...


@overload
def open(file: VideoPathType, mode: Literal["w"], **kwargs) -> av.container.OutputContainer: ...


def open(file: VideoPathType, mode: Literal["r", "w"], **kwargs):
    """Open video file with caching for read mode, direct av.open for write mode."""
    if mode == "r":
        _implicit_cleanup()
        return _retrieve_cache(file)
    else:
        return av.open(file, mode, **kwargs)


def cleanup_cache(container: Optional["MockedInputContainer" | VideoPathType] = None):
    """Manually cleanup cached containers."""
    _explicit_cleanup(container=container)


def _retrieve_cache(file: VideoPathType):
    """Get or create cached container and update usage tracking."""
    with get_cache_context() as cache:
        if file not in cache:
            logger.info(f"Caching video container for {file}")
            cache[file] = MockedInputContainer(file)
        else:
            logger.info(f"Using cached video container for {file}")
        container = cache[file]
        container.refs += 1
        container.last_used = time.time()
        return container


def _explicit_cleanup(container: Optional["MockedInputContainer" | VideoPathType] = None):
    """Force cleanup of specific container or all containers."""
    if container is None:
        # Get a snapshot of containers to avoid modification during iteration
        with get_cache_context() as cache:
            containers = list(cache.values())

        # Clean up each container individually
        for cont in containers:
            _explicit_cleanup(cont)
    else:
        with get_cache_context() as cache:
            if isinstance(container, VideoPathType):
                container = cache.get(container)
                if container is None:
                    return
            logger.info(f"Cleaning up cached video container for {container.file_path}")
            container._container.close()
            cache.pop(container.file_path, None)


# Ensure no forked processes share the same container object.
# PyAV's FFmpeg objects are not fork-safe, must not be forked.
if sys.platform != "win32":
    os.register_at_fork(before=lambda: (_explicit_cleanup(), gc.collect()))

# Ensure all containers are closed on program exit
atexit.register(_explicit_cleanup)


def _implicit_cleanup():
    """Cleanup unreferenced containers first and then cleanup the oldest containers."""
    with get_cache_context() as cache:
        if len(cache) <= DEFAULT_CACHE_SIZE:
            return
        # Remove unreferenced containers first
        to_remove = [path for path, container in cache.items() if container.refs == 0]
        for path in to_remove:
            logger.info(f"Cleaning up unreferenced cached video container for {path}")
            _explicit_cleanup(path)

        if len(cache) <= DEFAULT_CACHE_SIZE:
            return
        # Remove oldest containers until we reach the cache size limit
        containers_sorted_by_last_used = sorted(cache.values(), key=lambda x: x.last_used)
        to_remove = containers_sorted_by_last_used[: len(containers_sorted_by_last_used) - DEFAULT_CACHE_SIZE]
        for container in to_remove:
            logger.info(f"Cleaning up oldest cached video container for {container.file_path}")
            _explicit_cleanup(container)


class MockedInputContainer(InputContainerMixin):
    """Wrapper for av.InputContainer that tracks references and usage for caching."""

    def __init__(self, file: VideoPathType):
        self.file_path = file
        self._container: av.container.InputContainer = av.open(file, "r")
        self.refs = 0  # Reference count for tracking usage
        self.last_used = time.time()

    def __enter__(self) -> "MockedInputContainer":
        return self

    def close(self):
        """Decrement reference count and cleanup if no longer referenced."""
        self.refs = max(0, self.refs - 1)
        if self.refs == 0:
            logger.info(f"Ref count reached 0 for cached video container for {self.file_path}")


__all__ = ["open", "cleanup_cache"]
