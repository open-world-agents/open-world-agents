from pydantic import BaseModel


class File(BaseModel):
    name: str
    url: str
    size: int


class FilePair(BaseModel):
    mcap_file: str
    mkv_file: str
    basename: str
