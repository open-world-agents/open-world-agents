"""
Base EventEncoder interface for OWA data pipeline.

This module defines the common interface that all event encoders should implement,
ensuring consistency across different encoding strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional, Set, Tuple, Union, overload

from mcap_owa.highlevel import McapMessage
from owa.msgs.desktop.screen import ScreenCaptured


@dataclass
class BaseEventEncoderConfig:
    # Placeholder token for screen events - actual image processing happens in EpisodeTokenizer
    fake_image_placeholder: str = "<fake_image_placeholder>"


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

    @overload
    def decode_batch(
        self,
        encoded_batch: List[str],
        all_images: Optional[List[List[ScreenCaptured]]] = None,
        *,
        suppress_errors: Literal[False] = False,
    ) -> List[McapMessage]: ...

    @overload
    def decode_batch(
        self,
        encoded_batch: List[str],
        all_images: Optional[List[List[ScreenCaptured]]] = None,
        *,
        suppress_errors: Literal[True],
    ) -> List[Optional[McapMessage]]: ...

    def decode_batch(
        self,
        encoded_batch: List[str],
        all_images: Optional[List[List[ScreenCaptured]]] = None,
        *,
        suppress_errors: bool = False,
    ) -> Union[List[McapMessage], List[Optional[McapMessage]]]:
        """
        Decode a batch of encoded data.

        Args:
            encoded_batch: List of encoded event strings
            all_images: Optional list of images for each event
            suppress_errors: If True, return None for invalid events instead of raising exceptions

        Returns:
            List of McapMessage objects, or List of Optional[McapMessage] if suppress_errors=True
        """
        if all_images is None:
            all_images = [None] * len(encoded_batch)
        if len(encoded_batch) != len(all_images):
            raise ValueError("Length mismatch between encoded data and images")

        if suppress_errors:
            results = []
            for data, images in zip(encoded_batch, all_images):
                try:
                    results.append(self.decode(data, images))
                except Exception:
                    results.append(None)
            return results
        else:
            return [self.decode(data, images) for data, images in zip(encoded_batch, all_images)]

    @abstractmethod
    def get_vocab(self) -> Set[str]:
        """Get all tokens in the vocabulary."""
        pass
