import logging
import tempfile
from pathlib import Path
from typing import List, Set, Tuple

import fsspec
import requests
from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import HfFileSystem

from mcap_owa.highlevel import OWAMcapReader

from ..config import settings
from ..models.file import MediaReference, OWAFile
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

    def _analyze_mcap_media_references(
        self, mcap_path: Path, fs: fsspec.AbstractFileSystem
    ) -> Tuple[List[MediaReference], bool, bool]:
        """
        Analyze MCAP file to extract media references from ScreenCaptured messages.

        Args:
            mcap_path: Path to the MCAP file
            fs: Filesystem to use for file operations

        Returns:
            Tuple of (media_references, has_external_media, has_embedded_media)
        """
        media_references = []
        has_external_media = False
        has_embedded_media = False
        unique_uris = set()

        try:
            # For remote files, we need to download temporarily to analyze
            if not isinstance(fs, LocalFileSystem):
                # Skip analysis for remote files for now - too expensive
                # We'll rely on legacy mkv detection
                return media_references, has_external_media, has_embedded_media

            with OWAMcapReader(str(mcap_path)) as reader:
                # Sample only first few screen messages to understand media pattern
                # This is just for metadata analysis - we never load actual media content
                screen_messages_checked = 0
                max_messages_to_check = 3  # Very small sample to avoid performance issues

                for mcap_msg in reader.iter_messages(topics=["screen"]):
                    if screen_messages_checked >= max_messages_to_check:
                        break

                    screen_messages_checked += 1
                    # Only decode the message structure, not the actual media content
                    screen_data = mcap_msg.decoded

                    # Handle both new MediaRef format and legacy format
                    media_ref = None
                    if hasattr(screen_data, "media_ref") and screen_data.media_ref:
                        media_ref = screen_data.media_ref
                    elif hasattr(screen_data, "path") and screen_data.path:
                        # Legacy format - convert to MediaRef-like structure
                        pts_ns = getattr(screen_data, "pts", 0)
                        media_ref = type(
                            "MediaRef",
                            (),
                            {
                                "uri": screen_data.path,
                                "pts_ns": pts_ns,
                                "is_embedded": False,
                                "is_video": pts_ns is not None,
                                "is_remote": screen_data.path.startswith(("http://", "https://")),
                                "is_local": not screen_data.path.startswith(("http://", "https://")),
                            },
                        )()

                    if media_ref:
                        uri = getattr(media_ref, "uri", "")
                        if uri and uri not in unique_uris:
                            unique_uris.add(uri)

                            # Determine media type
                            is_embedded = getattr(media_ref, "is_embedded", False) or uri.startswith("data:")
                            is_video = (
                                getattr(media_ref, "is_video", False) or getattr(media_ref, "pts_ns", None) is not None
                            )
                            is_remote = getattr(media_ref, "is_remote", False) or uri.startswith(
                                ("http://", "https://")
                            )

                            if is_embedded:
                                media_type = "embedded"
                                has_embedded_media = True
                            elif is_remote:
                                media_type = "remote"
                                has_external_media = True
                            elif is_video:
                                media_type = "video"
                                has_external_media = True
                            else:
                                media_type = "image"
                                has_external_media = True

                            file_extension = None
                            if not is_embedded:
                                file_extension = Path(uri).suffix.lower()

                            media_references.append(
                                MediaReference(
                                    uri=uri,
                                    media_type=media_type,
                                    is_video=is_video,
                                    is_embedded=is_embedded,
                                    is_remote=is_remote,
                                    file_extension=file_extension,
                                )
                            )

        except Exception as e:
            logger.warning(f"Failed to analyze media references in {mcap_path}: {e}")

        return media_references, has_external_media, has_embedded_media

    def list_files(self, repo_id: str) -> List[OWAFile]:
        """
        List all MCAP files in a repository, analyzing their media references

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
            filesystem: LocalFileSystem = fsspec.filesystem(protocol=protocol)
            path = self.export_path
        else:
            protocol = "hf"
            filesystem: HfFileSystem = fsspec.filesystem(protocol=protocol)
            path = f"datasets/{repo_id}"

        # Find all MCAP files
        files = []
        # NOTE: local glob skip symlinked directory, which is weird.
        for mcap_file in filesystem.glob(f"{path}/**/*.mcap"):
            mcap_file = Path(mcap_file)

            # Check if MCAP file exists
            if not filesystem.exists(mcap_file.as_posix()):
                continue

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

            # Analyze media references in the MCAP file
            media_references, has_external_media, has_embedded_media = self._analyze_mcap_media_references(
                mcap_file, filesystem
            )

            # Check for legacy MKV file
            mkv_file_path = mcap_file.with_suffix(".mkv")
            url_mkv = None
            mcap_size = filesystem.info(mcap_file.as_posix()).get("size", 0)
            total_size = mcap_size

            if filesystem.exists(mkv_file_path.as_posix()):
                url_mkv = f"{url}.mkv" if url else f"{mcap_file.stem}.mkv"
                mkv_size = filesystem.info(mkv_file_path.as_posix()).get("size", 0)
                total_size += mkv_size

            files.append(
                OWAFile(
                    basename=mcap_file.stem,
                    original_basename=original_basename,
                    url=url,
                    size=total_size,
                    local=local,
                    url_mcap=f"{url}.mcap" if url else f"{mcap_file.stem}.mcap",
                    url_mkv=url_mkv,
                    media_references=media_references,
                    has_external_media=has_external_media,
                    has_embedded_media=has_embedded_media,
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
