"""
Base EventEncoder interface for OWA data pipeline.

This module defines the common interface that all event encoders should implement,
ensuring consistency across different encoding strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, overload

from mcap_owa.highlevel import McapMessage
from owa.msgs.desktop.screen import ScreenCaptured


@dataclass
class BaseEventEncoderConfig:
    # Placeholder token for screen events - actual image processing happens in EpisodeTokenizer
    fake_image_placeholder: str = "<fake_image_placeholder>"


class BaseEventEncoder(ABC):
    """Abstract base class for all event encoders."""

    @abstractmethod
    def encode(self, mcap_message: McapMessage) -> tuple[str, list[ScreenCaptured]]:
        """
        Encode a single McapMessage object to the encoder's format.

        Args:
            mcap_message: McapMessage instance

        Returns:
            Tuple containing encoded string and list of images for screen events

        Raises:
            InvalidInputError: If the input is invalid
            UnsupportedInputError: If the input is valid but encoder does not support it
        """

    @abstractmethod
    def decode(self, encoded_data: str, images: list[ScreenCaptured] | None = None) -> McapMessage:
        """
        Decode encoded data back to McapMessage format.

        Args:
            encoded_data: Encoded representation as string
            images: Optional list of image data for screen events

        Returns:
            McapMessage: Reconstructed message

        Raises:
            InvalidTokenError: If the token is invalid
            UnsupportedTokenError: If the token is valid but decoder does not support it
        """

    def encode_batch(self, mcap_messages: list[McapMessage]) -> tuple[list[str], list[list[ScreenCaptured]]]:
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
        encoded_batch: list[str],
        all_images: list[list[ScreenCaptured]] | None = None,
        *,
        suppress_errors: Literal[False] = False,
    ) -> list[McapMessage]: ...

    @overload
    def decode_batch(
        self,
        encoded_batch: list[str],
        all_images: list[list[ScreenCaptured]] | None = None,
        *,
        suppress_errors: Literal[True],
    ) -> list[McapMessage | None]: ...

    def decode_batch(
        self,
        encoded_batch: list[str],
        all_images: list[list[ScreenCaptured]] | None = None,
        *,
        suppress_errors: bool = False,
    ) -> list[McapMessage] | list[McapMessage | None]:
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
                except Exception:  # noqa: BLE001
                    results.append(None)
            return results
        else:
            return [self.decode(data, images) for data, images in zip(encoded_batch, all_images)]

    @abstractmethod
    def get_vocab(self) -> set[str]:
        """Get all tokens in the vocabulary."""
