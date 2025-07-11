import logging
from pathlib import Path
from typing import List, Tuple

from ..models.file import OWAFile
from ..repositories.file_repository import FileRepository
from ..services.cache_service import cache_service

logger = logging.getLogger(__name__)


class FileService:
    """Service for handling file operations"""

    def __init__(self, file_repository: FileRepository = None):
        """
        Initialize file service

        Args:
            file_repository: Repository for file operations
        """
        self.file_repository = file_repository or FileRepository()

    def list_files(self, repo_id: str) -> List[OWAFile]:
        """
        List all MCAP+MKV file pairs in a repository

        Args:
            repo_id: Repository ID ('local' or Hugging Face dataset ID)

        Returns:
            List of OWAFile objects
        """
        # Check cache first
        cached_files = cache_service.get_file_list(repo_id)
        if cached_files is not None:
            logger.info(f"Using cached file list for {repo_id}")
            return cached_files

        # Get fresh list and cache it
        files = self.file_repository.list_files(repo_id)
        cache_service.set_file_list(repo_id, files)
        logger.info(f"Cache miss for file list in {repo_id}, fetched {len(files)} files")
        return files

    def get_file_path(self, file_url: str, is_local: bool) -> Tuple[Path, bool]:
        """
        Get path to a file, downloading if necessary

        Args:
            file_url: URL or path to the file
            is_local: Whether the file is local

        Returns:
            Tuple of (file path, is_temporary)
        """
        # For local files, just validate the path
        if is_local:
            return self.file_repository.get_local_file_path(file_url), False

        # For remote files, check cache first
        cached_path = cache_service.get_cached_file(file_url)
        if cached_path:
            logger.info(f"Using cached file for {file_url}")
            return cached_path, False

        # Download and cache the file
        temp_path, is_temp = self.file_repository.download_file(file_url)
        if is_temp:
            # Cache the file for future use
            cached_path = cache_service.cache_file(file_url, temp_path)
            logger.info(f"Cached downloaded file {file_url} at {cached_path}")
            # We can now use the cached version and remove the temp file
            temp_path.unlink(missing_ok=True)
            return cached_path, False

        return temp_path, is_temp

    def cleanup_temp_file(self, file_path: Path) -> None:
        """
        Clean up a temporary file if it exists

        Args:
            file_path: Path to the file
        """
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {e}")


# Create singleton instance
file_service = FileService()
