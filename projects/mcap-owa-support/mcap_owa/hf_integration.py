"""
HuggingFace datasets integration for McapMessage and ScreenCaptured.

This module provides feature classes that implement the datasets
feature interface for serializing/deserializing McapMessage and ScreenCaptured objects.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Union

import pyarrow as pa

from .highlevel import McapMessage

# Import for HuggingFace datasets registration
try:
    from datasets.features.features import register_feature

    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False


@dataclass
class McapMessageFeature:
    """
    HuggingFace datasets feature for McapMessage.

    This is a separate class that implements the datasets feature interface
    for serializing/deserializing McapMessage objects.
    """

    decode: bool = True
    pa_type: ClassVar[Any] = pa.struct(
        {"topic": pa.string(), "timestamp": pa.int64(), "message": pa.binary(), "message_type": pa.string()}
    )
    _type: str = field(default="McapMessage", init=False, repr=False)

    def __call__(self):
        """Required for HuggingFace datasets feature interface."""
        return self.pa_type

    def encode_example(self, value: Union["McapMessage", dict]) -> dict:
        """
        Encode example into a format for Arrow.

        Args:
            value: Either a McapMessage instance or a dict with the required fields

        Returns:
            dict with "topic", "timestamp", "message", and "message_type" fields
        """
        # Import here to avoid circular imports
        from .highlevel.reader import McapMessage

        if isinstance(value, McapMessage):
            return value.model_dump()
        elif isinstance(value, dict):
            # Validate required fields
            required_fields = {"topic", "timestamp", "message", "message_type"}
            if not required_fields.issubset(value.keys()):
                missing = required_fields - value.keys()
                raise ValueError(f"Missing required fields: {missing}")
            return {
                "topic": value["topic"],
                "timestamp": value["timestamp"],
                "message": value["message"],
                "message_type": value["message_type"],
            }
        else:
            raise ValueError(f"Expected McapMessage or dict, got {type(value)}")

    def decode_example(self, value: dict, token_per_repo_id=None) -> Union["McapMessage", dict]:
        """
        Decode example from Arrow format.

        Args:
            value: dict with "topic", "timestamp", "message", and "message_type" fields
            token_per_repo_id: Optional parameter for HuggingFace datasets compatibility

        Returns:
            If decode=True: McapMessage object
            If decode=False: dict with the raw fields
        """
        if not self.decode:
            # Return raw dict when decode=False
            return {
                "topic": value["topic"],
                "timestamp": value["timestamp"],
                "message": value["message"],
                "message_type": value["message_type"],
            }
        else:
            # Import here to avoid circular imports
            from .highlevel.reader import McapMessage

            # Return McapMessage object when decode=True
            return McapMessage(**value)

    def flatten(self):
        """If in the decodable state, return the feature itself, otherwise flatten the feature into a dictionary."""
        if self.decode:
            return self
        else:
            # When decode=False, flatten to individual Value features
            from datasets.features.features import Value

            return {
                "topic": Value("string"),
                "timestamp": Value("int64"),
                "message": Value("binary"),
                "message_type": Value("string"),
            }

    def items(self):
        """Return empty items since this is a leaf feature, not a nested structure."""
        return []

    def __repr__(self) -> str:
        return f"McapMessageFeature(decode={self.decode})"


# Register features as HuggingFace datasets features
if _DATASETS_AVAILABLE:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "'register_feature' is experimental and might be subject to breaking changes in the future.",
            category=UserWarning,
        )
        # Register McapMessageFeature for HuggingFace datasets
        register_feature(McapMessageFeature, "McapMessage")
