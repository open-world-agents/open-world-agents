import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import settings
from ..repositories.cache_repository import CacheRepository, FileCacheRepository

logger = logging.getLogger(__name__)

CLEANUP_PERIOD = min(settings.FILE_CACHE_TTL or 10**9, settings.DEFAULT_CACHE_TTL or 10**9)


class CacheService:
    """Service for managing different types of caches"""

    def __init__(self):
        """Initialize cache service with different cache repositories"""
        self.metadata_cache = CacheRepository(settings.CACHE_DIR, "metadata")
        self.file_list_cache = CacheRepository(settings.CACHE_DIR, "file_lists")
        self.file_cache = FileCacheRepository(settings.CACHE_DIR)
        self._cleanup_task = None
        self._stop_cleanup = False

    def get_cached_file(self, url: str) -> Optional[Path]:
        """
        Get cached file if available

        Args:
            url: URL of the file

        Returns:
            Path to cached file if exists, None otherwise
        """
        return self.file_cache.get_file_path(url)

    def cache_file(self, url: str, file_path: Path) -> Path:
        """
        Cache a file

        Args:
            url: URL or identifier for the file
            file_path: Path to the file to cache

        Returns:
            Path to the cached file
        """
        return self.file_cache.store_file(url, file_path)

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached metadata"""
        return self.metadata_cache.get(key)

    def set_metadata(self, key: str, metadata: Dict[str, Any]) -> None:
        """Cache metadata"""
        self.metadata_cache.set(key, metadata)

    def get_file_list(self, repo_id: str) -> Optional[list]:
        """Get cached file list"""
        return self.file_list_cache.get(repo_id)

    def set_file_list(self, repo_id: str, file_list: list) -> None:
        """Cache file list"""
        self.file_list_cache.set(repo_id, file_list)

    def clear(self) -> None:
        self.file_list_cache.clear()

    def cleanup_expired(self) -> None:
        """Run a single cache cleanup operation"""
        try:
            deleted_count = self.file_cache.cleanup_expired()
            logger.info(f"Cleaned up {deleted_count} expired cached files")
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}", exc_info=True)

    async def start_periodic_cleanup(self, interval_seconds: int = CLEANUP_PERIOD) -> None:
        """
        Start periodic cache cleanup in background

        Args:
            interval_seconds: Time between cleanup operations in seconds
        """
        self._stop_cleanup = False

        async def cleanup_loop():
            while not self._stop_cleanup:
                try:
                    self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Error in periodic cache cleanup: {e}", exc_info=True)
                await asyncio.sleep(interval_seconds)

        # Store the task so it can be cancelled later
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info(f"Started periodic cache cleanup (every {interval_seconds} seconds)")

    async def stop_periodic_cleanup(self) -> None:
        """Stop the periodic cache cleanup task"""
        if self._cleanup_task:
            self._stop_cleanup = True
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped periodic cache cleanup")


# Create singleton instance
cache_service = CacheService()
