from typing import Optional
from pydantic import BaseModel


class McapMetadata(BaseModel):
    start_time: int = None
    end_time: int = None
    topics: set = set()


class OWAFile(BaseModel):
    basename: str
    original_basename: Optional[str] = None  # To store the original filename
    size: int
    local: bool
    url: str
    url_mcap: str
    url_mkv: str
