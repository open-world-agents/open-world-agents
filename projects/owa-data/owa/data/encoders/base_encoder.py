"""
Base EventEncoder interface for OWA data pipeline.

This module defines the common interface that all event encoders should implement,
ensuring consistency across different encoding strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from mcap_owa.highlevel import McapMessage
from owa.msgs.desktop.screen import ScreenCaptured


@dataclass
class BaseEventEncoderConfig:
    image_token: str = "<image>"
    image_token_prefix: str = "<fake_token_around_image><global-img>"
    image_token_suffix: str = "<fake_token_around_image>"


class BaseEventEncoder(ABC):
    """Abstract base class for all event encoders."""

    @abstractmethod
    def encode(self, mcap_message: McapMessage) -> Tuple[str, List[ScreenCaptured]]:
        """
        Encode a single McapMessage object to the encoder's format.

        Args:
            mcap_message: McapMessage instance

        Returns:
            Tuple containing encoded string and list of images for screen events

        Raises:
            ValueError: If the mcap_message format is invalid
        """
        pass

    @abstractmethod
    def decode(self, encoded_data: str, images: Optional[List[ScreenCaptured]] = None) -> McapMessage:
        """
        Decode encoded data back to McapMessage format.

        Args:
            encoded_data: Encoded representation as string
            images: Optional list of image data for screen events

        Returns:
            McapMessage: Reconstructed message

        Raises:
            ValueError: If encoded_data format is invalid
        """
        pass

    def encode_batch(self, mcap_messages: List[McapMessage]) -> Tuple[List[str], List[List[ScreenCaptured]]]:
        """Encode a batch of McapMessage objects."""
        all_tokens, all_images = [], []
        for message in mcap_messages:
            tokens, images = self.encode(message)
            all_tokens.append(tokens)
            all_images.append(images)
        return all_tokens, all_images

    def decode_batch(
        self, encoded_batch: List[str], all_images: Optional[List[List[ScreenCaptured]]] = None
    ) -> List[McapMessage]:
        """Decode a batch of encoded data."""
        if all_images is None:
            all_images = [[] for _ in encoded_batch]
        if len(encoded_batch) != len(all_images):
            raise ValueError("Length mismatch between encoded data and images")
        return [self.decode(data, images) for data, images in zip(encoded_batch, all_images)]

    @abstractmethod
    def get_vocab(self) -> Set[str]:
        """Get all tokens in the vocabulary."""
        pass
