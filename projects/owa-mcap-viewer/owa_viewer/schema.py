from pydantic import BaseModel


class OWAFile(BaseModel):
    basename: str
    size: int
    local: bool
    url: str
    url_mcap: str
    url_mkv: str
