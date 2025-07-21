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


class _ProcessCache:
    """Thread-safe cache for video containers within a single process."""

    def __init__(self):
        self._cache: dict[VideoPathType, "MockedInputContainer"] = {}
        self._lock = threading.RLock()
        self._pid = os.getpid()

    def get_cache(self) -> dict[VideoPathType, "MockedInputContainer"]:
        """Get the cache dictionary, ensuring it's valid for current process."""
        current_pid = os.getpid()
        with self._lock:
            if self._pid != current_pid:
                # Process was forked, reset cache
                self._cache.clear()
                self._pid = current_pid
            return self._cache


# Process-level cache
_process_cache = _ProcessCache()


def get_cache() -> dict[VideoPathType, "MockedInputContainer"]:
    """Get cache for current process."""
    return _process_cache.get_cache()


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
    container = get_cache().get(file, MockedInputContainer(file))
    container.refs += 1
    container.last_used = time.time()
    return container


def _explicit_cleanup(container: Optional["MockedInputContainer" | VideoPathType] = None):
    """Force cleanup of specific container or all containers."""
    if container is None:
        for container in get_cache().values():
            _explicit_cleanup(container)
    else:
        if isinstance(container, VideoPathType):
            container = get_cache().get(container)
            if container is None:
                return
        container._container.close()
        get_cache().pop(container.file_path, None)


# Ensure all containers are closed on program exit
atexit.register(_explicit_cleanup)


def _implicit_cleanup():
    """Cleanup refs == 0 first and then cleanup the oldest containers."""
    # Remove unreferenced containers first
    to_remove = [path for path, container in get_cache().items() if container.refs == 0]
    for path in to_remove:
        _explicit_cleanup(path)

    # Remove oldest containers if cache exceeds size limit
    if len(get_cache()) <= DEFAULT_CACHE_SIZE:
        return
    containers_sorted_by_last_used = sorted(get_cache().values(), key=lambda x: x.last_used)
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
        self.refs = max(0, self.refs - 1)
        if self.refs == 0:
            _explicit_cleanup(self)
