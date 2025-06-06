"""
Message system for the OWA framework.

This module provides the base classes and utilities for creating and handling
messages in the Open World Agents framework. All messages must implement the
BaseMessage interface to ensure consistent serialization and schema handling.
"""

import io
from abc import ABC, abstractmethod
from typing import Any, Dict, Self

from pydantic import BaseModel


class BaseMessage(ABC):
    """
    Abstract base class for all OWA messages.

    This class defines the interface that all messages must implement to ensure
    consistent serialization, deserialization, and schema handling across the
    OWA framework.
    """

    _type: str

    @abstractmethod
    def serialize(self, buffer: io.BytesIO) -> None:
        """
        Serialize the message to a binary buffer.

        Args:
            buffer: Binary buffer to write the serialized message to
        """
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, buffer: io.BytesIO) -> Self:
        """
        Deserialize a message from a binary buffer.

        Args:
            buffer: Binary buffer containing the serialized message

        Returns:
            Deserialized message instance
        """
        pass

    @classmethod
    @abstractmethod
    def get_schema(cls) -> Dict[str, Any]:
        """
        Get the JSON schema for this message type.

        Returns:
            JSON schema dictionary
        """
        pass


class OWAMessage(BaseModel, BaseMessage):
    """
    Standard OWA message implementation using Pydantic.

    This class provides a convenient base for creating messages that use
    Pydantic for validation and JSON serialization. Most OWA messages
    should inherit from this class.
    """

    model_config = {"extra": "forbid", "validate_assignment": True}

    # _type is defined as a class attribute, not a Pydantic field
    # Subclasses should override this
    _type: str

    def serialize(self, buffer: io.BytesIO) -> None:
        """
        Serialize the message to JSON format.

        Args:
            buffer: Binary buffer to write the serialized message to
        """
        buffer.write(self.model_dump_json(exclude_none=True).encode("utf-8"))

    @classmethod
    def deserialize(cls, buffer: io.BytesIO) -> Self:
        """
        Deserialize a message from JSON format.

        Args:
            buffer: Binary buffer containing the serialized message

        Returns:
            Deserialized message instance
        """
        return cls.model_validate_json(buffer.read())

    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        """
        Get the JSON schema for this message type.

        Returns:
            JSON schema dictionary
        """
        return cls.model_json_schema()
