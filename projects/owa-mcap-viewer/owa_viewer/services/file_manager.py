import logging
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import fsspec
import requests
from dotenv import load_dotenv
from fastapi import HTTPException
from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import HfFileSystem

from mcap_owa.highlevel import OWAMcapReader

from ..schema import McapMetadata, OWAFile

# Load environment variables
load_dotenv()

# Configure export path
export_path_env = os.environ.get("EXPORT_PATH", None)
if export_path_env:
    PUBLIC_HOSTING_MODE = False
    EXPORT_PATH = Path(export_path_env).as_posix()
else:
    PUBLIC_HOSTING_MODE = True  # if EXPORT_PATH is not set, we are in public hosting mode
    EXPORT_PATH = Path("./data").as_posix()  # default path for public hosting

logger = logging.getLogger(__name__)
logger.info(f"{PUBLIC_HOSTING_MODE=} {EXPORT_PATH=}")

# Cache structures
MCAP_METADATA_CACHE: Dict[str, McapMetadata] = {}  # key: mcap_filename, value: McapMetadata object
OWAFILE_CACHE: Dict[str, List[OWAFile]] = {}  # key: repo_id, value: list of OWAFile objects


class FileManager:
    """
    A static class to manage file operations for both local and remote files.
    Abstracts away the difference between local and remote file operations.
    """

    @staticmethod
    def safe_join(base_dir: str, *paths: str) -> Optional[Path]:
        """
        Join paths and ensure the result is within the base directory.

        Args:
            base_dir: The base directory to constrain paths within
            paths: Path components to join

        Returns:
            Path object if safe, None if path would escape base directory
        """
        base = Path(base_dir).resolve()
        target = (base / Path(*paths)).resolve()

        if not str(target).startswith(str(base)):
            logger.error(f"Unsafe path: {target} is outside of base directory {base}")
            return None

        return target

    @staticmethod
    def list_files(repo_id: str) -> List[OWAFile]:
        """
        For a given repository, list all available data files.

        Args:
            repo_id: Repository ID ('local' or Hugging Face dataset ID)

        Returns:
            List of OWAFile objects
        """
        # Choose filesystem based on repo_id
        if repo_id == "local":
            if PUBLIC_HOSTING_MODE:
                raise HTTPException(status_code=400, detail="repo_id=`local` requires EXPORT_PATH to be set")
            protocol = "file"
            fs: LocalFileSystem = fsspec.filesystem(protocol=protocol)
            path = EXPORT_PATH
        else:
            protocol = "hf"
            fs: HfFileSystem = fsspec.filesystem(protocol=protocol)
            path = f"datasets/{repo_id}"

        # Find all MCAP files with corresponding MKV files
        files = []
        for mcap_file in fs.glob(f"{path}/**/*.mcap"):
            mcap_file = PurePosixPath(mcap_file)
            if fs.exists(mcap_file.with_suffix(".mkv")) and fs.exists(mcap_file.with_suffix(".mcap")):
                basename = (mcap_file.parent / mcap_file.stem).as_posix()

                # Extract original basename for randomly generated files
                # Pattern matches: original_name_uuid where uuid is a standard UUID format
                original_basename = None
                if repo_id == "local":
                    uuid_pattern = r"(.+)_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
                    match = re.search(uuid_pattern, mcap_file.stem)
                    if match:
                        original_basename = match.group(1)

                # Prepare URL and path information
                if repo_id == "local":
                    basename = PurePosixPath(basename).relative_to(EXPORT_PATH).as_posix()
                    url = f"{basename}"
                    local = True
                else:
                    basename = basename[len(f"datasets/{repo_id}/") :]
                    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{basename}"
                    local = False

                files.append(
                    OWAFile(
                        basename=basename,
                        original_basename=original_basename,
                        url=url,
                        size=fs.info(mcap_file).get("size", 0),
                        local=local,
                        url_mcap=url + ".mcap",
                        url_mkv=url + ".mkv",
                    )
                )
        return files

    @staticmethod
    def get_mcap_path(mcap_filename: str, is_local: bool) -> Tuple[Path, bool]:
        """
        Returns the path to a MCAP file. If the file is remote, it is downloaded first.

        Args:
            mcap_filename: The name/path of the MCAP file
            is_local: Whether the file is local or remote

        Returns:
            Tuple containing:
                - Path to the MCAP file
                - Whether the path is a temporary file that should be deleted after use
        """
        if is_local:
            return FileManager._get_local_mcap_path(mcap_filename)
        else:
            return FileManager._get_remote_mcap_path(mcap_filename)

    @staticmethod
    def _get_local_mcap_path(mcap_filename: str) -> Tuple[Path, bool]:
        """
        Get the path to a local MCAP file.

        Args:
            mcap_filename: Local path to MCAP file

        Returns:
            Tuple of (file path, is_temporary)
        """
        mcap_path = FileManager.safe_join(EXPORT_PATH, mcap_filename)

        if mcap_path is None or not mcap_path.exists():
            raise HTTPException(status_code=404, detail="MCAP file not found")

        return mcap_path, False

    @staticmethod
    def _get_remote_mcap_path(mcap_filename: str) -> Tuple[Path, bool]:
        """
        Download a remote MCAP file and return its local path.

        Args:
            mcap_filename: URL to the MCAP file

        Returns:
            Tuple of (file path, is_temporary)
        """
        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as temp_mcap:
            logger.info(f"Downloading MCAP file to: {temp_mcap.name}")

            try:
                resp = requests.get(mcap_filename)
                resp.raise_for_status()  # Raise exception for HTTP errors
                temp_mcap.write(resp.content)
            except Exception as e:
                # Clean up the temp file if download fails
                Path(temp_mcap.name).unlink(missing_ok=True)
                logger.error(f"Error downloading MCAP file: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error downloading MCAP file: {str(e)}")

        return Path(temp_mcap.name), True

    @staticmethod
    def cleanup_temp_file(file_path: Path) -> None:
        """
        Clean up a temporary file if it exists.

        Args:
            file_path: Path to the temporary file
        """
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file {file_path}: {e}", exc_info=True)

    @staticmethod
    def build_mcap_metadata(mcap_path: Path, mcap_filename: str) -> None:
        """
        Build metadata about an MCAP file (time range, topics, etc.)
        and store it in the cache.

        Args:
            mcap_path: Path to the MCAP file
            mcap_filename: Identifier for the MCAP file (for caching)
        """
        if not Path(mcap_path).exists():
            raise HTTPException(status_code=404, detail="MCAP file not found")

        logger.info(f"Building metadata for MCAP file: {mcap_path}")

        try:
            with OWAMcapReader(mcap_path) as reader:
                metadata = McapMetadata(
                    start_time=reader.start_time,
                    end_time=reader.end_time,
                    topics=reader.topics,
                )

                logger.info(
                    f"Metadata built for {mcap_path}: {len(metadata.topics)} topics, "
                    f"time range {metadata.start_time} to {metadata.end_time}"
                )

                # Store in the cache
                MCAP_METADATA_CACHE[mcap_filename] = metadata
                logger.info(f"Metadata cached for {mcap_filename}")

        except Exception as e:
            logger.error(f"Error building MCAP metadata: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error building MCAP metadata: {str(e)}")
