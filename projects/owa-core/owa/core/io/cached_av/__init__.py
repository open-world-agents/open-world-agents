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

from ...utils.typing import PathLike
from .input_container_mixin import InputContainerMixin

DEFAULT_CACHE_SIZE = int(os.environ.get("AV_CACHE_SIZE", 10))


class _CacheContext:
    """Context manager for thread-safe access to video container cache."""

    def __init__(self):
        self._cache: dict[PathLike, "MockedInputContainer"] = {}
        self._lock = threading.RLock()
        self._thread_id = threading.get_ident()  # Track the thread that created this cache

    def __enter__(self) -> dict[PathLike, "MockedInputContainer"]:
        """Enter context manager and return locked cache."""
        self._lock.acquire()
        self._check_thread_safety()
        return self._cache

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and release lock."""
        self._lock.release()
        return False  # Don't suppress exceptions

    def _check_thread_safety(self):
        """Check if we're in a different thread and clear cache if needed."""
        current_thread_id = threading.get_ident()
        if current_thread_id != self._thread_id:
            # Huge performance degradation observed when accessing cache from different threads.
            # Prefer to crash rather than introduce hard-to-debug issues.
            if len(self._cache) > 0:
                raise RuntimeError("Thread change detected, cannot access cache")

            logger.info(f"Thread change detected (from {self._thread_id} to {current_thread_id}), clearing cache")
            # Clear the cache since PyAV/FFmpeg objects are not thread-safe
            # Don't close containers as they may still be in use - just clear the cache
            self._cache.clear()
            self._thread_id = current_thread_id
            gc.collect()


# Global cache context instance
_cache_context = _CacheContext()


def get_cache_context():
    """Get cache context for atomic operations. Use with 'with' statement."""
    return _cache_context


@overload
def open(file: PathLike, mode: Literal["r"], *, keep_av_open: bool = False, **kwargs) -> "MockedInputContainer": ...


@overload
def open(file: PathLike, mode: Literal["w"], **kwargs) -> av.container.OutputContainer: ...


def open(file: PathLike, mode: Literal["r", "w"], *, keep_av_open: bool = False, **kwargs):
    """Open video file with caching for read mode, direct av.open for write mode.

    Args:
        file: Path to video file
        mode: Open mode ('r' for read, 'w' for write)
        keep_av_open: If True, keep container in cache when closed. If False, force cleanup.
        **kwargs: Additional arguments passed to av.open for write mode
    """
    if mode == "r":
        _implicit_cleanup()
        return _retrieve_cache(file, keep_av_open=keep_av_open)
    else:
        return av.open(file, mode, **kwargs)


def cleanup_cache(container: Optional["MockedInputContainer" | PathLike] = None):
    """Manually cleanup cached containers."""
    _explicit_cleanup(container=container)


def _retrieve_cache(file: PathLike, *, keep_av_open: bool = False):
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
        container.keep_av_open = keep_av_open  # Set the cache preference
        return container


def _explicit_cleanup(container: Optional["MockedInputContainer" | PathLike] = None):
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
            if isinstance(container, PathLike):
                container = cache.get(container)
                if container is None:
                    return
            # At this point, container must be MockedInputContainer
            assert isinstance(container, MockedInputContainer), f"Expected MockedInputContainer, got {type(container)}"
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

    def __init__(self, file: PathLike):
        self.file_path = file
        self._container: av.container.InputContainer = av.open(file, "r")
        self.refs = 0  # Reference count for tracking usage
        self.last_used = time.time()
        self.keep_av_open = False  # Default to not keeping open

    def __enter__(self) -> "MockedInputContainer":
        return self

    def close(self):
        """Decrement reference count and cleanup if no longer referenced or if keep_av_open=False."""
        self.refs = max(0, self.refs - 1)
        if self.refs == 0:
            logger.info(f"Ref count reached 0 for cached video container for {self.file_path}")
            # If keep_av_open is False, force cleanup immediately
            if not self.keep_av_open:
                logger.info(f"Force cleaning up cached video container for {self.file_path} (keep_av_open=False)")
                _explicit_cleanup(self)


__all__ = ["open", "cleanup_cache"]
