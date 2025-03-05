import io
from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseMessage(ABC):
    _type: str

    @abstractmethod
    def serialize(self, buffer: io.BytesIO): ...

    @classmethod
    @abstractmethod
    def deserialize(cls, buffer): ...

    @classmethod
    @abstractmethod
    def get_schema(cls): ...


class OWAMessage(BaseModel, BaseMessage):
    _type: str

    def serialize(self, buffer):
        buffer.write(self.model_dump_json().encode())

    @classmethod
    def deserialize(cls, buffer):
        return cls.model_validate_json(buffer.read())

    @classmethod
    def get_schema(cls):
        return cls.model_json_schema()
