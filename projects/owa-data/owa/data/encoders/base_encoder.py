"""
Base EventEncoder interface for OWA data pipeline.

This module defines the common interface that all event encoders should implement,
ensuring consistency across different encoding strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from mcap_owa.highlevel import McapMessage
from owa.msgs.desktop.screen import ScreenCaptured


@dataclass
class BaseEventEncoderConfig:
    image_token: str = "<image>"
    image_token_prefix: str = "<fake_token_around_image><global-img>"
    image_token_suffix: str = "<fake_token_around_image>"


class BaseEventEncoder(ABC):
    """
    Abstract base class for all event encoders.

    This interface ensures that all encoders provide consistent encode/decode
    functionality while allowing for different internal representations and
    optimization strategies.
    """

    @abstractmethod
    def encode(self, mcap_message: McapMessage) -> Tuple[str, List[ScreenCaptured]]:
        """
        Encode a single McapMessage object to the encoder's format.

        Args:
            mcap_message: McapMessage instance

        Returns:
            Tuple containing:
                - str: Encoded representation as string
                - List[ScreenCaptured]: Image data for screen events

        Raises:
            ValueError: If the mcap_message format is invalid
        """
        pass

    @abstractmethod
    def decode(self, encoded_data: str, images: Optional[List[ScreenCaptured]] = None) -> "McapMessage":
        """
        Decode encoded data back to McapMessage format.

        Args:
            encoded_data: Encoded representation as string
            images: Optional list of image data for screen events

        Returns:
            McapMessage: Reconstructed message in McapMessage format

        Raises:
            ValueError: If encoded_data format is invalid
        """
        pass

    @abstractmethod
    def encode_batch(self, mcap_messages: List[McapMessage]) -> Tuple[List[str], List[List[ScreenCaptured]]]:
        """
        Encode a batch of McapMessage-like objects.

        Args:
            mcap_messages: List of McapMessage instances or dictionaries

        Returns:
            Tuple containing:
                - List[str]: Batch of encoded representations as strings
                - List[List[ScreenCaptured]]: Image data for each event
        """
        pass

    @abstractmethod
    def decode_batch(
        self, encoded_batch: List[str], all_images: Optional[List[List[ScreenCaptured]]] = None
    ) -> List[McapMessage]:
        """
        Decode a batch of encoded data.

        Args:
            encoded_batch: Batch of encoded representations as strings
            all_images: Optional list of image data lists for each event

        Returns:
            List[McapMessage]: Reconstructed messages in McapMessage format
        """
        pass

    @abstractmethod
    def get_vocab(self) -> Set[str]:
        """Get all tokens in the vocabulary."""
        pass

    def extend_vocab_from_tokenizer(self, tokenizer, model=None):
        """
        Extend tokenizer and model vocabulary with encoder tokens.

        Args:
            tokenizer: HuggingFace tokenizer to extend
            model: Optional model to resize embeddings
        """
        tokenizer.add_tokens(self.get_vocab())
        if model is not None:
            model.resize_token_embeddings(len(tokenizer))

    def get_encoder_info(self) -> Dict[str, Any]:
        """
        Get information about this encoder.

        Returns:
            Dict containing encoder metadata like type, added tokens, etc.
        """
        return {
            "encoder_type": self.__class__.__name__,
            "added_tokens": self.get_vocab(),
        }
