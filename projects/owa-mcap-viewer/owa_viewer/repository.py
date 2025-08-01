import hashlib
import logging
import tempfile
import time
from pathlib import Path
from typing import Generic, List, Optional, TypeVar

import diskcache
import fsspec
import requests
from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import HfFileSystem

from .config import settings
from .models import OWAFile
from .utils import FileDownloadError, FileNotFoundError, extract_original_filename, safe_join

logger = logging.getLogger(__name__)
T = TypeVar("T")


class CacheRepository(Generic[T]):
    """Repository for managing cached data with TTL and size limits"""

    def __init__(self, cache_dir: str, namespace: str):
        """Initialize cache repository"""
        self.cache = diskcache.Cache(cache_dir)
        self.namespace = namespace

    def _get_key(self, key: str) -> str:
        """Create namespaced key"""
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> Optional[T]:
        """Get item from cache"""
        return self.cache.get(self._get_key(key))

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set item in cache"""
        ttl = ttl or settings.DEFAULT_CACHE_TTL
        self.cache.set(self._get_key(key), value, expire=ttl)

    def delete(self, key: str) -> None:
        """Delete item from cache"""
        self.cache.delete(self._get_key(key))

    def clear(self) -> None:
        """Clear all items in this namespace"""
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
        """Initialize file cache repository"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_cache = CacheRepository(str(self.cache_dir), "file_metadata")

    def get_file_path(self, url: str) -> Optional[Path]:
        """Get cached file path if it exists and is valid"""
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
        """Store file in cache"""
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
        """Delete file from cache"""
        file_id = self._get_file_id(url)
        file_path = self.cache_dir / file_id

        if file_path.exists():
            file_path.unlink()

        self.metadata_cache.delete(file_id)

    def _get_file_id(self, url: str) -> str:
        """Create a unique ID for a file based on its URL"""
        return hashlib.md5(url.encode()).hexdigest()

    def cleanup_expired(self) -> int:
        """Clean up expired files"""
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


class FileRepository:
    """Repository for file operations (both local and remote)"""

    def __init__(self, export_path: str = settings.EXPORT_PATH):
        """Initialize file repository"""
        self.export_path = Path(export_path).as_posix()

    def list_files(self, repo_id: str) -> List[OWAFile]:
        """List all MCAP+MKV file pairs in a repository"""
        # Select filesystem and path based on repo_id
        if repo_id == "local":
            if settings.PUBLIC_HOSTING_MODE:
                raise FileNotFoundError("Local repository not available in public hosting mode")
            protocol = "file"
            fs: LocalFileSystem = fsspec.filesystem(protocol=protocol)
            path = self.export_path
        else:
            protocol = "hf"
            fs: HfFileSystem = fsspec.filesystem(protocol=protocol)
            path = f"datasets/{repo_id}"

        # Find all MCAP files with corresponding MKV files
        files = []
        for mcap_file in fs.glob(f"{path}/**/*.mcap"):
            mcap_file = Path(mcap_file)

            # Only include if both MCAP and MKV files exist
            if fs.exists(mcap_file.with_suffix(".mkv").as_posix()) and fs.exists(mcap_file.as_posix()):
                basename = (mcap_file.parent / mcap_file.stem).as_posix()

                # Extract original basename for local files
                original_basename = None
                if repo_id == "local":
                    original_basename = extract_original_filename(mcap_file.stem)

                # Format URLs and paths based on repo type
                if repo_id == "local":
                    try:
                        # Convert both paths to consistent format before comparison
                        export_path_posix = Path(self.export_path).resolve()
                        basename_posix = Path(basename).resolve()

                        # Get the relative part by removing export_path prefix
                        rel_path = basename_posix.relative_to(export_path_posix).as_posix()
                        url = rel_path
                    except ValueError:
                        # Fallback to just the filename if path manipulation fails
                        url = mcap_file.stem

                    local = True
                else:
                    # For remote repositories, remove the datasets/repo_id/ prefix
                    prefix = f"datasets/{repo_id}/"
                    if basename.startswith(prefix):
                        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{basename[len(prefix) :]}"
                    else:
                        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{basename}"
                    local = False

                files.append(
                    OWAFile(
                        basename=mcap_file.stem,
                        original_basename=original_basename,
                        url=url,
                        size=fs.info(mcap_file.with_suffix(".mkv").as_posix()).get("size", 0)
                        + fs.info(mcap_file.as_posix()).get("size", 0),
                        local=local,
                        url_mcap=f"{url}.mcap" if url else f"{mcap_file.stem}.mcap",
                        url_mkv=f"{url}.mkv" if url else f"{mcap_file.stem}.mkv",
                    )
                )
        return files

    def get_local_file_path(self, file_path: str) -> Path:
        """Get path to a local file, ensuring it's within the export path"""
        full_path = safe_join(self.export_path, file_path)
        if not full_path or not full_path.exists():
            raise FileNotFoundError(file_path)
        return full_path

    def download_file(self, url: str) -> Path:
        """Download a file from a URL and return the path (caller should cache it)"""
        temp_file = tempfile.NamedTemporaryFile(suffix=Path(url).suffix, delete=False)
        temp_path = Path(temp_file.name)

        try:
            logger.info(f"Downloading file from {url} to {temp_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return temp_path

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise FileDownloadError(url, str(e))
