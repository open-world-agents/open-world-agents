from typing import Optional

from pydantic import BaseModel


class OWAFile(BaseModel):
    """Represents an OWA file (MCAP+MKV pair)"""

    basename: str
    original_basename: Optional[str] = None
    size: int
    local: bool
    url: str
    url_mcap: str
    url_mkv: str


class DatasetInfo(BaseModel):
    """Dataset information for the viewer"""

    repo_id: str
    files: int
    size: str
