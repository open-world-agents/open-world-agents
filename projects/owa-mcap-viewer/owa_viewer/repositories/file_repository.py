import logging
import tempfile
from pathlib import Path
from typing import List, Tuple

import fsspec
import requests
from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import HfFileSystem

from ..config import settings
from ..models.file import OWAFile
from ..utils.exceptions import FileDownloadError, FileNotFoundError
from ..utils.path_utils import extract_original_filename, safe_join

logger = logging.getLogger(__name__)


class FileRepository:
    """Repository for file operations (both local and remote)"""

    def __init__(self, export_path: str = settings.EXPORT_PATH):
        """
        Initialize file repository

        Args:
            export_path: Path to local file storage
        """
        self.export_path = Path(export_path).as_posix()

    def list_files(self, repo_id: str) -> List[OWAFile]:
        """
        List all MCAP+MKV file pairs in a repository

        Args:
            repo_id: Repository ID ('local' or Hugging Face dataset ID)

        Returns:
            List of OWAFile objects
        """
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
        # NOTE: local glob skip symlinked directory, which is weird.
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
                    # Fix the relative path handling
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
        """
        Get path to a local file, ensuring it's within the export path

        Args:
            file_path: Relative path to the file

        Returns:
            Absolute path to the file
        """
        full_path = safe_join(self.export_path, file_path)
        if not full_path or not full_path.exists():
            raise FileNotFoundError(file_path)
        return full_path

    def download_file(self, url: str) -> Tuple[Path, bool]:
        """
        Download a file from a URL to a temporary location

        Args:
            url: URL to download from

        Returns:
            Tuple of (file path, is_temporary)
        """
        temp_file = tempfile.NamedTemporaryFile(suffix=Path(url).suffix, delete=False)
        temp_path = Path(temp_file.name)

        try:
            logger.info(f"Downloading file from {url} to {temp_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return temp_path, True

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise FileDownloadError(url, str(e))
