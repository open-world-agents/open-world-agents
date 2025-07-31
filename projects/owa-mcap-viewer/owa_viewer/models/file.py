from typing import List, Optional, Set

from pydantic import BaseModel


class MediaReference(BaseModel):
    """Represents a media reference found in MCAP screen messages"""

    uri: str
    media_type: str  # "video", "image", "embedded", "remote"
    is_video: bool = False
    is_embedded: bool = False
    is_remote: bool = False
    file_extension: Optional[str] = None


class OWAFile(BaseModel):
    """Represents an OWA MCAP file with flexible media references"""

    basename: str
    original_basename: Optional[str] = None
    size: int
    local: bool
    url: str
    url_mcap: str

    # Media references found in the MCAP file
    media_references: List[MediaReference] = []
    has_external_media: bool = False
    has_embedded_media: bool = False

    # Legacy fields for backward compatibility
    url_mkv: Optional[str] = None  # Only set if traditional mkv file exists


class DatasetInfo(BaseModel):
    """Dataset information for the viewer"""

    repo_id: str
    files: int
    size: str
