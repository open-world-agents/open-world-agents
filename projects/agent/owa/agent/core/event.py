from typing import Any

from pydantic import BaseModel


class Event(BaseModel):
    timestamp: int
    topic: str
    msg: Any
