import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import settings
from ..repositories.cache_repository import CacheRepository, FileCacheRepository

logger = logging.getLogger(__name__)


class CacheService:
    """Service for managing different types of caches"""

    def __init__(self):
        """Initialize cache service with different cache repositories"""
        self.metadata_cache = CacheRepository(settings.MCAP_CACHE_DIR, "metadata")
        self.file_list_cache = CacheRepository(settings.MCAP_CACHE_DIR, "file_lists")
        self.file_cache = FileCacheRepository(settings.MCAP_CACHE_DIR)

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

    def cleanup(self) -> None:
        """Clean up expired cache items"""
        deleted_count = self.file_cache.cleanup_expired()
        logger.info(f"Cleaned up {deleted_count} expired cached files")


# Create singleton instance
cache_service = CacheService()
