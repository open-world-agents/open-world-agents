import logging
import os
import tempfile
from pathlib import Path, PurePosixPath

import fsspec
import requests
from dotenv import load_dotenv
from fastapi import HTTPException
from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import HfFileSystem

from mcap_owa.highlevel import OWAMcapReader

from ..schema import McapMetadata, OWAFile

load_dotenv()

# Configure export path
EXPORT_PATH = os.environ.get("EXPORT_PATH", "./data")
EXPORT_PATH = Path(EXPORT_PATH).as_posix()

logger = logging.getLogger(__name__)
logger.info(f"{EXPORT_PATH=}")


MCAP_METADATA_CACHE: dict[str, McapMetadata] = dict()  # key: mcap_filename, value: McapMetadata object

OWAFILE_CACHE: dict[str, list[OWAFile]] = dict()  # key: repo_id, value: list of OWAFile objects


class FileManager:
    """
    A static class to manage file operations for both local and remote files.
    Abstracts away the difference between local and remote file operations.
    """

    @staticmethod
    def safe_join(base_dir: str, *paths: str) -> Path | None:
        """Join paths and ensure the result is within the base directory."""
        base = Path(base_dir).resolve()
        target = (base / Path(*paths)).resolve()

        if not str(target).startswith(str(base)):
            logger.error(f"Unsafe path: {target} is outside of base directory {base}")
            return None

        return target

    @staticmethod
    def list_files(repo_id: str) -> list[OWAFile]:
        """For a given repository, list all available data files."""

        if repo_id == "local":
            protocol = "file"
            fs: LocalFileSystem = fsspec.filesystem(protocol=protocol)
            path = EXPORT_PATH
        else:
            protocol = "hf"
            fs: HfFileSystem = fsspec.filesystem(protocol=protocol)
            path = f"datasets/{repo_id}"

        files = []
        for mcap_file in fs.glob(f"{path}/**/*.mcap"):
            mcap_file = PurePosixPath(mcap_file)
            if fs.exists(mcap_file.with_suffix(".mkv")) and fs.exists(mcap_file.with_suffix(".mcap")):
                basename = (mcap_file.parent / mcap_file.stem).as_posix()
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
                        url=url,
                        size=fs.info(mcap_file).get("size", 0),
                        local=local,
                        url_mcap=url + ".mcap",
                        url_mkv=url + ".mkv",
                    )
                )
        return files

    @staticmethod
    def get_mcap_path(mcap_filename: str, is_local: bool) -> tuple[Path, bool]:
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
        is_temp = False

        if is_local:
            # Check if local file exists
            mcap_path = FileManager.safe_join(EXPORT_PATH, mcap_filename)

            if mcap_path is None or not mcap_path.exists():
                raise HTTPException(status_code=404, detail="MCAP file not found")

        else:
            # Download remote file
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

            mcap_path = Path(temp_mcap.name)
            is_temp = True

        return mcap_path, is_temp

    @staticmethod
    def cleanup_temp_file(file_path: Path) -> None:
        """Clean up a temporary file if it exists"""
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file {file_path}: {e}", exc_info=True)

    @staticmethod
    def build_mcap_metadata(mcap_path: Path, mcap_filename: str):
        """Build metadata about an MCAP file (time range, topics, etc.)"""

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
