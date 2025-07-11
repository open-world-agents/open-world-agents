import hashlib
import time
from pathlib import Path
from typing import Generic, Optional, TypeVar

import diskcache

from ..config import settings

T = TypeVar("T")


class CacheRepository(Generic[T]):
    """Repository for managing cached data with TTL and size limits"""

    def __init__(self, cache_dir: str, namespace: str):
        """
        Initialize cache repository

        Args:
            cache_dir: Directory where cache is stored
            namespace: Namespace for this cache to avoid collisions
        """
        self.cache = diskcache.Cache(cache_dir)
        self.namespace = namespace

    def _get_key(self, key: str) -> str:
        """Create namespaced key"""
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> Optional[T]:
        """Get item from cache"""
        return self.cache.get(self._get_key(key))

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """
        Set item in cache

        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds (None for no expiration)
        """
        ttl = ttl or settings.DEFAULT_CACHE_TTL
        self.cache.set(self._get_key(key), value, expire=ttl)

    def delete(self, key: str) -> None:
        """Delete item from cache"""
        self.cache.delete(self._get_key(key))

    def clear(self) -> None:
        """Clear all items in this namespace"""
        # Only clear items in our namespace
        namespace_prefix = f"{self.namespace}:"
        for key in list(self.cache):
            if str(key).startswith(namespace_prefix):
                self.cache.delete(key)

    def __del__(self):
        """Close cache when repository is deleted"""
        self.cache.close()


class FileCacheRepository:
    """Repository for caching files on disk with metadata"""

    def __init__(self, cache_dir: str = settings.CACHE_DIR):
        """
        Initialize file cache repository

        Args:
            cache_dir: Directory where cached files are stored
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_cache = CacheRepository(str(self.cache_dir), "file_metadata")

    def get_file_path(self, url: str) -> Optional[Path]:
        """
        Get cached file path if it exists and is valid

        Args:
            url: URL of the file

        Returns:
            Path to cached file if exists and valid, None otherwise
        """
        file_id = self._get_file_id(url)
        metadata = self.metadata_cache.get(file_id)

        if not metadata:
            return None

        # Check if file exists and metadata is valid
        file_path = self.cache_dir / file_id
        if not file_path.exists():
            self.metadata_cache.delete(file_id)
            return None

        # Check if expired
        if metadata.get("expires_at") and metadata["expires_at"] < time.time():
            self.delete_file(url)
            return None

        return file_path

    def store_file(self, url: str, file_path: Path, ttl: Optional[int] = None) -> Path:
        """
        Store file in cache

        Args:
            url: URL of the file
            file_path: Path to the file to store
            ttl: Time-to-live in seconds

        Returns:
            Path to cached file
        """
        file_id = self._get_file_id(url)
        target_path = self.cache_dir / file_id

        # Copy file to cache
        with open(file_path, "rb") as src, open(target_path, "wb") as dst:
            dst.write(src.read())

        # Store metadata
        metadata = {
            "url": url,
            "created_at": time.time(),
            "expires_at": time.time() + (ttl or settings.FILE_CACHE_TTL),
            "size": target_path.stat().st_size,
        }
        self.metadata_cache.set(file_id, metadata)

        return target_path

    def delete_file(self, url: str) -> None:
        """
        Delete file from cache

        Args:
            url: URL of the file
        """
        file_id = self._get_file_id(url)
        file_path = self.cache_dir / file_id

        if file_path.exists():
            file_path.unlink()

        self.metadata_cache.delete(file_id)

    def _get_file_id(self, url: str) -> str:
        """
        Create a unique ID for a file based on its URL

        Args:
            url: URL of the file

        Returns:
            Unique file ID
        """
        return hashlib.md5(url.encode()).hexdigest()

    def cleanup_expired(self) -> int:
        """
        Clean up expired files

        Returns:
            Number of files deleted
        """
        count = 0
        current_time = time.time()

        for file_id in self.metadata_cache.cache.iterkeys():
            if not isinstance(file_id, str) or not file_id.startswith("file_metadata:"):
                continue

            key = file_id[len("file_metadata:") :]
            metadata = self.metadata_cache.get(key)

            if not metadata:
                continue

            if metadata.get("expires_at") and metadata["expires_at"] < current_time:
                file_path = self.cache_dir / key
                if file_path.exists():
                    file_path.unlink()
                self.metadata_cache.delete(key)
                count += 1

        return count
