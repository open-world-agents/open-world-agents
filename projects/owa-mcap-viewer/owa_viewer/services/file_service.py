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

    def list_files(self, repo_id: str, limit: int = 100, offset: int = 0, use_cache: bool = True) -> List[OWAFile]:
        """
        List MCAP files in a repository with pagination

        Args:
            repo_id: Repository ID ('local' or Hugging Face dataset ID)
            limit: Maximum number of files to return (default: 100)
            offset: Number of files to skip (default: 0)
            use_cache: Whether to use cache (default: True, disabled for pagination)

        Returns:
            List of OWAFile objects
        """
        # For pagination, skip cache to ensure consistent results
        if use_cache and offset == 0 and limit == 100:
            # Only use cache for default first page
            cached_files = cache_service.get_file_list(repo_id)
            if cached_files is not None:
                logger.info(f"Using cached file list for {repo_id}")
                return cached_files

        # Get fresh list with pagination (no expensive MediaRef analysis)
        files = self.file_repository.list_files(repo_id, limit=limit, offset=offset, analyze_media=False)

        # Only cache the first page with default settings
        if offset == 0 and limit == 100:
            cache_service.set_file_list(repo_id, files)

        logger.info(f"Fetched {len(files)} files for {repo_id} (offset={offset}, limit={limit})")
        return files

    def get_file_media_details(self, repo_id: str, filename: str) -> OWAFile:
        """
        Get detailed media information for a specific file (on-demand analysis)

        Args:
            repo_id: Repository ID ('local' or Hugging Face dataset ID)
            filename: Specific filename to analyze

        Returns:
            OWAFile object with detailed media analysis
        """
        # This performs expensive MediaRef analysis only for the requested file
        files = self.file_repository.list_files(repo_id, limit=1, offset=0, analyze_media=True)

        # Find the specific file (this is a simplified implementation)
        # In a real implementation, you'd want to search more efficiently
        for file in files:
            if file.basename == filename or file.url_mcap.endswith(f"{filename}.mcap"):
                return file

        raise FileNotFoundError(f"File {filename} not found in repository {repo_id}")

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
