import atexit
import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, ContextManager, Dict, Generic, Optional, TypeVar

from loguru import logger

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Container for cached resources with metadata."""

    obj: T
    cleanup_callback: Callable
    last_used: float = 0.0
    refs: int = 0


# TODO: thread-safe implementation of ResourceCache
class ResourceCache(Generic[T], Dict[str, CacheEntry[T]]):
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

    def add_entry(self, key: str, obj: T, cleanup_callback: Optional[Callable] = None):
        """Add new entry with refs=1. Raises ValueError if key already exists."""
        if key in self:
            raise ValueError(f"Entry {key} already exists. Use acquire_entry() to increment refs.")

        if cleanup_callback is None:
            # Default to context manager cleanup
            if not isinstance(obj, ContextManager):
                raise ValueError(f"Object {obj} does not implement context manager protocol")
            cleanup_callback = lambda: obj.__exit__(None, None, None)  # noqa: E731

        self[key] = CacheEntry(obj=obj, cleanup_callback=cleanup_callback, refs=1, last_used=time.time())
        logger.info(f"Added entry to cache: {key=}, total {len(self)}")
        logger.debug(f"Cache entry for {key=} has {self[key].refs=}")

    def acquire_entry(self, key: str) -> T:
        """Increment refs and return cached object. Raises KeyError if not found."""
        if key not in self:
            raise KeyError(f"Entry {key} not found in cache")

        self[key].refs += 1
        self[key].last_used = time.time()
        logger.debug(f"Acquired entry: {key=}, {self[key].refs=}")
        return self[key].obj

    def release_entry(self, key: str):
        """Decrement refs and trigger LRU cleanup if needed."""
        self[key].refs -= 1
        logger.debug(f"Released entry: {key=}, {self[key].refs=}")
        self._cleanup_if_needed()

    def pop(self, key: str, default: Optional[CacheEntry[T]] = None) -> Optional[CacheEntry[T]]:  # type: ignore[override]
        """Remove entry and execute cleanup callback."""
        if key in self:
            self[key].cleanup_callback()
        logger.info(f"Popped entry from cache: {key=}, total {len(self)}")
        return super().pop(key, default)

    def clear(self):
        """Clear all entries and execute cleanup callbacks."""
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
