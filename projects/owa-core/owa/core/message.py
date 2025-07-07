"""
Message system for the OWA framework.

This module provides the base classes and utilities for creating and handling
messages in the Open World Agents framework. All messages must implement the
BaseMessage interface to ensure consistent serialization and schema handling.
"""

import importlib.util
import io
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Self

from pydantic import BaseModel, model_validator
from pydantic.fields import ModelPrivateAttr


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

    model_config = {"extra": "forbid"}

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

    @classmethod
    def verify_type(cls) -> bool:
        """
        Verify that the _type field is valid and the import logic can operate correctly.

        This method checks whether the module path specified in _type can be
        successfully imported and the class can be accessed. This ensures that
        the message type can be properly deserialized by the decoder.

        Returns:
            True if verification passes, False otherwise

        Raises:
            ImportError: If the module cannot be imported
            AttributeError: If the class cannot be found in the module
            ValueError: If the _type format is invalid
        """
        if not hasattr(cls, "_type"):
            raise ValueError(f"Class {cls.__name__} must define a _type attribute")

        # Handle Pydantic ModelPrivateAttr - extract the actual string value
        type_attr = cls._type
        if isinstance(type_attr, ModelPrivateAttr):
            type_str = type_attr.default
        else:
            type_str = type_attr

        if not type_str:
            raise ValueError(f"Class {cls.__name__} must define a non-empty _type attribute")

        # Support both old module-based format (module.path.ClassName) and new domain-based format (domain/MessageType)
        # OEP-0006 introduces domain-based naming for better organization
        if "/" in type_str:
            # New domain-based format (OEP-0006): domain/MessageType
            # For domain-based messages, we skip module verification since they're registered via entry points
            # The message registry system handles discovery and validation
            return True
        elif "." in type_str:
            # Old module-based format: module.path.ClassName
            # Split into module path and class name (same logic as decoder)
            try:
                module_path, class_name = type_str.rsplit(".", 1)
            except ValueError:
                raise ValueError(
                    f"Invalid _type format '{type_str}'. Expected format: 'module.path.ClassName' or 'domain/MessageType'"
                )

            # Check if module can be imported
            try:
                spec = importlib.util.find_spec(module_path)
                if spec is None:
                    raise ImportError(f"Module '{module_path}' specified in _type '{type_str}' cannot be found")
            except (ImportError, ModuleNotFoundError, ValueError) as e:
                raise ImportError(f"Module '{module_path}' specified in _type '{type_str}' cannot be found: {e}")

            # Try to import the module and get the class
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                raise ImportError(f"Failed to import module '{module_path}' specified in _type '{type_str}': {e}")

            # Check if class exists in the module
            if not hasattr(module, class_name):
                raise AttributeError(
                    f"Class '{class_name}' not found in module '{module_path}' (from _type '{type_str}')"
                )

            # Get the class and verify it's the same as the current class
            target_class = getattr(module, class_name)
            if target_class is not cls:
                warnings.warn(
                    f"Class mismatch: _type '{type_str}' points to {target_class} but verification was called on {cls}. "
                    f"This may indicate an inconsistent _type definition.",
                    UserWarning,
                )
        else:
            raise ValueError(
                f"Invalid _type format '{type_str}'. Expected format: 'module.path.ClassName' or 'domain/MessageType'"
            )

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
