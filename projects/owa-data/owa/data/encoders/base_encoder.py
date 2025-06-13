"""
Base EventEncoder interface for OWA data pipeline.

This module defines the common interface that all event encoders should implement,
ensuring consistency across different encoding strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from owa.env.gst.msg import ScreenEmitted


class BaseEventEncoder(ABC):
    """
    Abstract base class for all event encoders.

    This interface ensures that all encoders provide consistent encode/decode
    functionality while allowing for different internal representations and
    optimization strategies.
    """

    @abstractmethod
    def encode(self, raw_event: Dict[str, Any]) -> Tuple[str, List[Union[ScreenEmitted, Dict]]]:
        """
        Encode a single raw event to the encoder's format.

        Args:
            raw_event: Raw event dictionary with keys:
                - topic: Event topic (e.g., 'keyboard', 'screen')
                - timestamp_ns: Timestamp in nanoseconds
                - message_type: Full message type identifier
                - msg: Serialized message content (bytes or string)
                - file_path: Source MCAP file path (optional)

        Returns:
            Tuple containing:
                - str: Encoded representation as string
                - List[Union[ScreenEmitted, Dict]]: Image data for screen events

        Raises:
            ValueError: If the raw_event format is invalid
        """
        pass

    @abstractmethod
    def decode(self, encoded_data: str, images: Optional[List[Union[ScreenEmitted, Dict]]] = None) -> Dict[str, Any]:
        """
        Decode encoded data back to original raw event format.

        Args:
            encoded_data: Encoded representation as string
            images: Optional list of image data for screen events

        Returns:
            Dict: Reconstructed raw event in original format

        Raises:
            ValueError: If encoded_data format is invalid
        """
        pass

    @abstractmethod
    def encode_batch(
        self, raw_events: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[List[Union[ScreenEmitted, Dict]]]]:
        """
        Encode a batch of raw events.

        Args:
            raw_events: List of raw event dictionaries

        Returns:
            Tuple containing:
                - List[str]: Batch of encoded representations as strings
                - List[List[Union[ScreenEmitted, Dict]]]: Image data for each event
        """
        pass

    @abstractmethod
    def decode_batch(
        self, encoded_batch: List[str], all_images: Optional[List[List[Union[ScreenEmitted, Dict]]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Decode a batch of encoded data.

        Args:
            encoded_batch: Batch of encoded representations as strings
            all_images: Optional list of image data lists for each event

        Returns:
            List[Dict]: Reconstructed raw events
        """
        pass

    def get_vocab_size(self) -> Optional[int]:
        """
        Get the vocabulary size for token-based encoders.

        Returns:
            Optional[int]: Vocabulary size if applicable, None for string-based encoders
        """
        return None

    def get_encoder_info(self) -> Dict[str, Any]:
        """
        Get information about this encoder.

        Returns:
            Dict containing encoder metadata like type, vocab size, etc.
        """
        return {
            "encoder_type": self.__class__.__name__,
            "vocab_size": self.get_vocab_size(),
        }
