import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlparse

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

from mcap_owa.highlevel import OWAMcapReader

from ..models.file import MediaReference
from ..services.file_service import file_service
from ..utils.exceptions import FileNotFoundError, MediaResolutionError

logger = logging.getLogger(__name__)


class MediaService:
    """Service for resolving and serving media references from MCAP files"""

    def __init__(self):
        self.file_service = file_service

    def resolve_media_reference(
        self, mcap_filename: str, media_uri: str, local: bool = True
    ) -> Tuple[Path, bool, str]:
        """
        Resolve a media reference to an actual file path.

        Args:
            mcap_filename: Name of the MCAP file containing the reference
            media_uri: URI from MediaRef
            local: Whether the MCAP file is local

        Returns:
            Tuple of (resolved_path, is_temporary, media_type)

        Raises:
            MediaResolutionError: If media cannot be resolved
        """
        try:
            # Handle embedded data URIs
            if media_uri.startswith("data:"):
                return self._handle_data_uri(media_uri)

            # Handle remote URLs
            if media_uri.startswith(("http://", "https://")):
                return self._handle_remote_url(media_uri)

            # Handle local file paths (relative or absolute)
            return self._handle_local_path(mcap_filename, media_uri, local)

        except Exception as e:
            logger.error(f"Failed to resolve media reference {media_uri}: {e}")
            raise MediaResolutionError(f"Failed to resolve media reference: {str(e)}")

    def _handle_data_uri(self, data_uri: str) -> Tuple[Path, bool, str]:
        """Handle embedded data URI by extracting to temporary file"""
        try:
            # Parse data URI: data:image/png;base64,<data>
            parsed = urlparse(data_uri)
            if parsed.scheme != "data":
                raise ValueError("Invalid data URI scheme")

            # Extract media type and data
            header, data = parsed.path.split(",", 1)
            media_type_part = header.split(";")[0]

            # Decode base64 data
            decoded_data = base64.b64decode(data)

            # Determine file extension from media type
            extension = ".png"  # default
            if "jpeg" in media_type_part or "jpg" in media_type_part:
                extension = ".jpg"
            elif "gif" in media_type_part:
                extension = ".gif"
            elif "webp" in media_type_part:
                extension = ".webp"

            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=extension, delete=False)
            temp_path = Path(temp_file.name)

            with open(temp_path, "wb") as f:
                f.write(decoded_data)

            # Determine media type for serving
            if "image" in media_type_part:
                serve_media_type = media_type_part
            else:
                serve_media_type = "application/octet-stream"

            return temp_path, True, serve_media_type

        except Exception as e:
            raise MediaResolutionError(f"Failed to decode data URI: {str(e)}")

    def _handle_remote_url(self, url: str) -> Tuple[Path, bool, str]:
        """Handle remote URL by downloading to temporary file"""
        try:
            # Use file service to download and cache
            temp_path, is_temp = self.file_service.get_file_path(url, is_local=False)

            # Determine media type from file extension
            extension = Path(url).suffix.lower()
            if extension in [".mkv", ".webm", ".mp4"]:
                media_type = "video/x-matroska" if extension == ".mkv" else f"video/{extension[1:]}"
            elif extension in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
                media_type = f"image/{extension[1:].replace('jpg', 'jpeg')}"
            else:
                media_type = "application/octet-stream"

            return temp_path, is_temp, media_type

        except Exception as e:
            raise MediaResolutionError(f"Failed to download remote URL {url}: {str(e)}")

    def _handle_local_path(self, mcap_filename: str, media_path: str, local: bool) -> Tuple[Path, bool, str]:
        """Handle local file path, resolving relative paths against MCAP location"""
        try:
            # Get MCAP file path for relative resolution
            mcap_path, mcap_is_temp = self.file_service.get_file_path(mcap_filename, local)

            # Resolve relative path against MCAP directory
            if not Path(media_path).is_absolute():
                mcap_dir = mcap_path.parent
                resolved_path = mcap_dir / media_path
            else:
                resolved_path = Path(media_path)

            # Check if file exists
            if not resolved_path.exists():
                raise FileNotFoundError(f"Media file not found: {resolved_path}")

            # Determine media type from file extension
            extension = resolved_path.suffix.lower()
            if extension in [".mkv", ".webm", ".mp4"]:
                media_type = "video/x-matroska" if extension == ".mkv" else f"video/{extension[1:]}"
            elif extension in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
                media_type = f"image/{extension[1:].replace('jpg', 'jpeg')}"
            else:
                media_type = "application/octet-stream"

            # Clean up temporary MCAP file if needed
            if mcap_is_temp:
                self.file_service.cleanup_temp_file(mcap_path)

            return resolved_path, False, media_type

        except Exception as e:
            raise MediaResolutionError(f"Failed to resolve local path {media_path}: {str(e)}")

    def serve_media_from_reference(
        self, mcap_filename: str, media_uri: str, local: bool = True
    ) -> Union[FileResponse, Response]:
        """
        Serve media content from a MediaRef URI.

        Args:
            mcap_filename: Name of the MCAP file containing the reference
            media_uri: URI from MediaRef
            local: Whether the MCAP file is local

        Returns:
            FastAPI response with media content
        """
        try:
            resolved_path, is_temp, media_type = self.resolve_media_reference(mcap_filename, media_uri, local)

            logger.info(f"Serving media from {resolved_path} (type: {media_type}, temp: {is_temp})")

            # Always use FileResponse for efficient streaming, even for temporary files
            # This avoids loading large files (potentially 10GB+) into memory
            if is_temp:
                # For temporary files, we'll let the OS handle cleanup after serving
                # This is more memory-efficient than loading the entire file
                logger.warning(f"Serving temporary file {resolved_path} - cleanup may be delayed")
                return FileResponse(str(resolved_path), media_type=media_type)
            else:
                # For permanent files, use FileResponse for efficient streaming
                return FileResponse(str(resolved_path), media_type=media_type)

        except MediaResolutionError as e:
            logger.error(f"Media resolution error: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error serving media: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error serving media: {str(e)}")

    def get_primary_video_reference(self, mcap_filename: str, local: bool = True) -> Optional[str]:
        """
        Get the primary video reference from an MCAP file for the video player.

        Args:
            mcap_filename: Name of the MCAP file
            local: Whether the MCAP file is local

        Returns:
            URI of the primary video reference, or None if no video found
        """
        try:
            mcap_path, is_temp = self.file_service.get_file_path(mcap_filename, local)

            with OWAMcapReader(str(mcap_path)) as reader:
                # Look for the first video reference in screen messages
                # This only reads message metadata, not actual media content
                messages_checked = 0
                max_messages_to_check = 10  # Check more messages to find video reference

                for mcap_msg in reader.iter_messages(topics=["screen"]):
                    if messages_checked >= max_messages_to_check:
                        break
                    messages_checked += 1

                    screen_data = mcap_msg.decoded

                    # Handle both new MediaRef format and legacy format
                    if hasattr(screen_data, "media_ref") and screen_data.media_ref:
                        media_ref = screen_data.media_ref
                        if getattr(media_ref, "is_video", False) or getattr(media_ref, "pts_ns", None) is not None:
                            uri = getattr(media_ref, "uri", "")
                            if uri:
                                logger.info(f"Found video reference in MediaRef: {uri}")
                                return uri
                    elif hasattr(screen_data, "path") and screen_data.path:
                        # Legacy format
                        pts = getattr(screen_data, "pts", None)
                        if pts is not None:
                            logger.info(f"Found video reference in legacy format: {screen_data.path}")
                            return screen_data.path

                logger.warning(f"No video reference found in first {messages_checked} screen messages")

            # Clean up temporary MCAP file if needed
            if is_temp:
                self.file_service.cleanup_temp_file(mcap_path)

            return None

        except Exception as e:
            logger.warning(f"Failed to get primary video reference from {mcap_filename}: {e}")
            return None

    def validate_media_references(self, mcap_filename: str, local: bool = True) -> dict:
        """
        Validate all media references in an MCAP file.

        Args:
            mcap_filename: Name of the MCAP file
            local: Whether the MCAP file is local

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "total_references": 0,
            "valid_references": 0,
            "invalid_references": 0,
            "embedded_references": 0,
            "external_references": 0,
            "errors": [],
        }

        try:
            mcap_path, is_temp = self.file_service.get_file_path(mcap_filename, local)
            unique_uris = set()
            messages_checked = 0
            max_messages_to_check = 10  # Limit validation to avoid performance issues

            with OWAMcapReader(str(mcap_path)) as reader:
                for mcap_msg in reader.iter_messages(topics=["screen"]):
                    if messages_checked >= max_messages_to_check:
                        break
                    messages_checked += 1
                    screen_data = mcap_msg.decoded

                    # Extract media reference
                    media_ref = None
                    if hasattr(screen_data, "media_ref") and screen_data.media_ref:
                        media_ref = screen_data.media_ref
                    elif hasattr(screen_data, "path") and screen_data.path:
                        # Legacy format
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
                            validation_results["total_references"] += 1

                            # Check if embedded
                            if uri.startswith("data:"):
                                validation_results["embedded_references"] += 1
                                validation_results["valid_references"] += 1
                            else:
                                validation_results["external_references"] += 1

                                # For performance, only do basic validation without actually resolving
                                # Full resolution could be expensive for large files
                                try:
                                    if uri.startswith(("http://", "https://")):
                                        # Remote URL - assume valid for now
                                        validation_results["valid_references"] += 1
                                    elif Path(uri).is_absolute() or not uri.startswith("/"):
                                        # Local path - do basic existence check only for absolute paths
                                        if Path(uri).is_absolute() and not Path(uri).exists():
                                            validation_results["invalid_references"] += 1
                                            validation_results["errors"].append(f"File not found: {uri}")
                                        else:
                                            validation_results["valid_references"] += 1
                                    else:
                                        # Relative path - assume valid (would need MCAP context to resolve)
                                        validation_results["valid_references"] += 1
                                except Exception as e:
                                    validation_results["invalid_references"] += 1
                                    validation_results["errors"].append(f"Error validating {uri}: {str(e)}")

            # Clean up temporary MCAP file if needed
            if is_temp:
                self.file_service.cleanup_temp_file(mcap_path)

        except Exception as e:
            validation_results["errors"].append(f"Failed to validate MCAP file: {str(e)}")

        return validation_results


# Global instance
media_service = MediaService()
