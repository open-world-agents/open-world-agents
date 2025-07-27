"""
JSONEventEncoder for converting raw events to MLLM-compatible JSON format.

This module implements the JSONEventEncoder class that converts raw event data
from the Event Dataset into JSON string format suitable for training Vision-Language-Action (VLA) models.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from mcap_owa.highlevel.reader import McapMessage
from owa.msgs.desktop.screen import ScreenCaptured

from .base_encoder import BaseEventEncoder, BaseEventEncoderConfig


@dataclass
class JSONEventEncoderConfig(BaseEventEncoderConfig):
    pass


class JSONEventEncoder(BaseEventEncoder):
    """JSON-based encoder for converting raw events to MLLM training format."""

    def __init__(self, config: JSONEventEncoderConfig = JSONEventEncoderConfig(), **kwargs):
        self.config = JSONEventEncoderConfig(**(config.__dict__ | kwargs))

    def encode(self, mcap_message: McapMessage) -> Tuple[str, List[ScreenCaptured]]:
        """Encode a single McapMessage object to MLLM training format."""
        mcap_message = mcap_message if isinstance(mcap_message, McapMessage) else McapMessage(**mcap_message)

        images = []
        if mcap_message.topic == "screen" and mcap_message.message_type == "desktop/ScreenCaptured":
            screen_event = mcap_message.decoded
            if not isinstance(screen_event, ScreenCaptured):
                raise ValueError(f"Expected ScreenCaptured object, got {type(screen_event)}")
            images.append(screen_event)
            image_token_with_prefix_suffix = (
                f"{self.config.image_token_prefix}{self.config.image_token}{self.config.image_token_suffix}"
            )
            mcap_message.message = image_token_with_prefix_suffix.encode("utf-8")

        return f"<EVENT_START>{mcap_message.model_dump_json()}<EVENT_END>", images

    def decode(self, encoded_data: str, images: Optional[List[ScreenCaptured]] = None) -> McapMessage:
        """Decode serialized event back to McapMessage format."""
        if not encoded_data.startswith("<EVENT_START>") or not encoded_data.endswith("<EVENT_END>"):
            raise ValueError("Invalid serialized format: missing <EVENT_START> or <EVENT_END> tokens")

        content = encoded_data[len("<EVENT_START>") : -len("<EVENT_END>")]

        try:
            event_dict = eval(content)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Failed to parse event content: {e}")

        if not isinstance(event_dict, dict):
            raise ValueError("Decoded content is not a dictionary")

        # Handle screen events with image data
        expected_image_token = (
            f"{self.config.image_token_prefix}{self.config.image_token}{self.config.image_token_suffix}"
        )
        if (
            event_dict.get("topic") == "screen"
            and event_dict.get("message_type") == "desktop/ScreenCaptured"
            and event_dict.get("message") == expected_image_token
        ):
            if not images:
                raise ValueError("Screen event requires image data but none provided")
            image_data = images[0]
            event_dict["message"] = image_data.model_dump_json(exclude={"frame_arr"})

        return McapMessage(
            topic=event_dict["topic"],
            timestamp=event_dict["timestamp_ns"],
            message_type=event_dict["message_type"],
            message=event_dict["message"]
            if isinstance(event_dict["message"], bytes)
            else event_dict["message"].encode("utf-8"),
        )

    def encode_batch(self, mcap_messages: List[McapMessage]) -> Tuple[List[str], List[List[ScreenCaptured]]]:
        """Encode a batch of McapMessages."""
        texts, all_images = [], []
        for mcap_message in mcap_messages:
            text, images = self.encode(mcap_message)
            texts.append(text)
            all_images.append(images)
        return texts, all_images

    def decode_batch(
        self, encoded_batch: List[str], all_images: Optional[List[List[ScreenCaptured]]] = None
    ) -> List[McapMessage]:
        """Decode a batch of serialized events."""
        if all_images is None:
            all_images = [[] for _ in encoded_batch]
        if len(encoded_batch) != len(all_images):
            raise ValueError("Length mismatch between texts and images")
        return [self.decode(text, images) for text, images in zip(encoded_batch, all_images)]

    def get_vocab(self) -> set[str]:
        """Get all tokens in the vocabulary."""
        # JSONEventEncoder doesn't add special tokens to the vocabulary
        # except for the image tokens which might be needed by the tokenizer
        return {self.config.image_token, self.config.image_token_prefix, self.config.image_token_suffix}
