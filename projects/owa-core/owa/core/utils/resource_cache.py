import atexit
import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, ContextManager

from loguru import logger


@dataclass
class CacheEntry:
    """Container for cached resources with metadata."""

    obj: Any
    cleanup_callback: Callable
    last_used: float = 0.0
    refs: int = 0


class ResourceCache(dict[str, CacheEntry]):
    """Reference-counted resource cache with LRU eviction and automatic cleanup."""

    def __init__(self, *args, max_size: int = 0, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)
        self._register_cleanup_handlers()

    def _register_cleanup_handlers(self):
        """Register cleanup callbacks for process lifecycle events."""
        if sys.platform != "win32":
            os.register_at_fork(before=lambda: (self.clear(), gc.collect()))
        atexit.register(self.clear)

    def add_entry(self, key: str, obj: Any, cleanup_callback: Callable | None = None):
        """Add or update a cached resource with reference counting."""
        if cleanup_callback is None:
            # Default to context manager cleanup
            if not isinstance(obj, ContextManager):
                raise ValueError(f"Object {obj} does not implement context manager protocol")
            cleanup_callback = lambda: obj.__exit__(None, None, None)  # noqa: E731

        if key not in self:
            self[key] = CacheEntry(obj=obj, cleanup_callback=cleanup_callback)
            logger.info(f"Added entry to cache: {key=}, total {len(self)}")

        self[key].refs += 1
        self[key].last_used = time.time()
        logger.debug(f"Cache entry for {key=} has {self[key].refs=}")

    def release_entry(self, key: str):
        """Decrease reference count and trigger cleanup if needed."""
        self[key].refs -= 1
        logger.debug(f"Released entry: {key=}, {self[key].refs=}")
        self._cleanup_if_needed()

    def pop(self, key: str, default: Any | None = None) -> CacheEntry | Any | None:
        """Remove and return cache entry with cleanup."""
        if key in self:
            self[key].cleanup_callback()
        logger.info(f"Popped entry from cache: {key=}, total {len(self)}")
        return super().pop(key, default)

    def clear(self):
        """Clear all cache entries and execute cleanup callbacks."""
        for entry in self.values():
            entry.cleanup_callback()
        logger.info(f"Cache cleared, total {len(self)}")
        super().clear()

    def _cleanup_if_needed(self):
        """Evict unreferenced entries using LRU policy when cache exceeds max_size."""
        if self.max_size == 0 or len(self) <= self.max_size:
            return

        # Find unreferenced entries and sort by last access time
        unreferenced = [k for k, v in self.items() if v.refs == 0]
        oldest_first = sorted(unreferenced, key=lambda k: self[k].last_used)

        # Remove excess entries starting with oldest
        excess_count = len(self) - self.max_size
        for key in oldest_first[:excess_count]:
            self.pop(key)
