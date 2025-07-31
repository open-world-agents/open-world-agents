import logging
from pathlib import Path
from typing import List, Tuple

from ..models import OWAFile
from ..repository import FileRepository
from .cache_service import CacheService

logger = logging.getLogger(__name__)


class FileService:
    """Service for handling file operations"""

    def __init__(self, cache_service: CacheService):
        """Initialize file service with dependencies"""
        self.file_repository = FileRepository()
        self.cache_service = cache_service

    def list_files(self, repo_id: str) -> List[OWAFile]:
        """List all MCAP+MKV file pairs in a repository"""
        # Check cache first
        cached_files = self.cache_service.get_file_list(repo_id)
        if cached_files is not None:
            logger.info(f"Using cached file list for {repo_id}")
            return cached_files

        # Get fresh list and cache it
        files = self.file_repository.list_files(repo_id)
        self.cache_service.set_file_list(repo_id, files)
        logger.info(f"Cache miss for file list in {repo_id}, fetched {len(files)} files")
        return files

    def get_file_path(self, file_url: str, is_local: bool) -> Tuple[Path, bool]:
        """Get path to a file, downloading if necessary"""
        # For local files, just validate the path
        if is_local:
            return self.file_repository.get_local_file_path(file_url), False

        # For remote files, check cache first
        cached_path = self.cache_service.get_cached_file(file_url)
        if cached_path:
            logger.info(f"Using cached file for {file_url}")
            return cached_path, False

        # Download and cache the file
        temp_path, is_temp = self.file_repository.download_file(file_url)
        if is_temp:
            # Cache the file for future use
            cached_path = self.cache_service.cache_file(file_url, temp_path)
            logger.info(f"Cached downloaded file {file_url} at {cached_path}")
            # We can now use the cached version and remove the temp file
            temp_path.unlink(missing_ok=True)
            return cached_path, False

        return temp_path, is_temp

    def cleanup_temp_file(self, file_path: Path) -> None:
        """Clean up a temporary file if it exists"""
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {e}")

    def get_local_file_path(self, file_path: str) -> Path:
        """Get path to a local file (delegate to repository)"""
        return self.file_repository.get_local_file_path(file_path)
