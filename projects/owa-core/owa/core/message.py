"""
Message system for the OWA framework.

This module provides the base classes and utilities for creating and handling
messages in the Open World Agents framework. All messages must implement the
BaseMessage interface to ensure consistent serialization and schema handling.
"""

import io
import warnings
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Self

from pydantic import BaseModel, model_validator
from pydantic.fields import ModelPrivateAttr


class BaseMessage(ABC):
    """
    Abstract base class for all OWA messages.

    This class defines the interface that all messages must implement to ensure
    consistent serialization, deserialization, and schema handling across the
    OWA framework.
    """

    _type: ClassVar[str]

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

    model_config = {"extra": "forbid"}

    # _type is defined as a class attribute, not a Pydantic field
    # Subclasses should override this
    _type: ClassVar[str]

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

    @classmethod
    def verify_type(cls) -> bool:
        """
        Verify that the _type field is valid and uses the correct format.

        This method validates that the _type field uses the domain-based format
        introduced in OEP-0006. Domain-based messages are registered via entry points
        and handled by the message registry system.

        Returns:
            True if verification passes

        Raises:
            ValueError: If the _type format is invalid or missing
        """
        if not hasattr(cls, "_type"):
            raise ValueError(f"Class {cls.__name__} must define a _type attribute")

        # Handle Pydantic ModelPrivateAttr - extract the actual string value
        type_attr = cls._type
        if isinstance(type_attr, ModelPrivateAttr):
            type_str = type_attr.default
        else:
            type_str = type_attr

        if not type_str or not isinstance(type_str, str):
            raise ValueError(f"Class {cls.__name__} must define a non-empty _type attribute as a string")

        # Only support domain-based format (domain/MessageType) introduced in OEP-0006
        if "/" not in type_str:
            raise ValueError(f"Invalid _type format '{type_str}'. Expected format: 'domain/MessageType'")

        return True

    @model_validator(mode="after")
    def _validate_type_on_creation(self) -> "OWAMessage":
        """
        Automatically verify _type when creating message instances.

        This validator runs after model creation to ensure the _type field
        is valid. It only issues warnings for verification failures to avoid
        breaking existing code.
        """
        try:
            self.__class__.verify_type()
        except (ImportError, AttributeError, ValueError) as e:
            warnings.warn(
                f"Message type verification failed for {self.__class__.__name__}: {e}. "
                f"This may cause issues with message deserialization.",
                UserWarning,
            )
        return self
