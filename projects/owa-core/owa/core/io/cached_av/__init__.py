import atexit
import os
import threading
import time
from pathlib import Path
from typing import Literal, Optional, Union, overload

import av
import av.container

from .input_container_mixin import InputContainerMixin

DEFAULT_CACHE_SIZE = 10

VideoPathType = Union[str, os.PathLike, Path]


class _CacheContext:
    """Context manager for thread-safe access to video container cache."""

    def __init__(self):
        self._cache: dict[VideoPathType, "MockedInputContainer"] = {}
        self._lock = threading.RLock()
        self._pid = os.getpid()

    def __enter__(self) -> dict[VideoPathType, "MockedInputContainer"]:
        """Enter context manager and return locked cache."""
        self._lock.acquire()
        current_pid = os.getpid()
        if self._pid != current_pid:
            # Process was forked, reset cache
            self._cache.clear()
            self._pid = current_pid
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
            cache[file] = MockedInputContainer(file)
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
            container._container.close()
            cache.pop(container.file_path, None)


# Ensure all containers are closed on program exit
atexit.register(_explicit_cleanup)


def _implicit_cleanup():
    """Cleanup refs == 0 first and then cleanup the oldest containers."""
    with get_cache_context() as cache:
        # Remove unreferenced containers first
        to_remove = [path for path, container in cache.items() if container.refs == 0]
        for path in to_remove:
            _explicit_cleanup(path)

        # Remove oldest containers if cache exceeds size limit
        if len(cache) <= DEFAULT_CACHE_SIZE:
            return
        containers_sorted_by_last_used = sorted(cache.values(), key=lambda x: x.last_used)
        to_remove = containers_sorted_by_last_used[: len(containers_sorted_by_last_used) - DEFAULT_CACHE_SIZE]
        for container in to_remove:
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
        with get_cache_context() as cache:
            self.refs = max(0, self.refs - 1)
            if self.refs == 0:
                # Cleanup immediately while holding the lock to prevent race conditions
                self._container.close()
                cache.pop(self.file_path, None)


__all__ = ["open", "cleanup_cache"]
