"""
JSONEventEncoder for converting raw events to MLLM-compatible JSON format.
"""

import json
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from mcap_owa.highlevel.reader import McapMessage
from owa.msgs.desktop.screen import ScreenCaptured

from .base_encoder import BaseEventEncoder, BaseEventEncoderConfig


@dataclass
class JSONEventEncoderConfig(BaseEventEncoderConfig):
    pass


class JSONEventEncoder(BaseEventEncoder):
    """JSON-based encoder for converting raw events to MLLM training format."""

    def __init__(self, config: JSONEventEncoderConfig = None, **kwargs):
        if config is None:
            config = JSONEventEncoderConfig()
        self.config = JSONEventEncoderConfig(**(config.__dict__ | kwargs))

    def encode(self, mcap_message: McapMessage) -> Tuple[str, List[ScreenCaptured]]:
        """Encode a single McapMessage object to JSON format."""
        mcap_message = mcap_message if isinstance(mcap_message, McapMessage) else McapMessage(**mcap_message)
        images = []

        # Handle screen events specially
        if mcap_message.topic == "screen" and mcap_message.message_type == "desktop/ScreenCaptured":
            screen_event = mcap_message.decoded
            if not isinstance(screen_event, ScreenCaptured):
                raise ValueError(f"Expected ScreenCaptured object, got {type(screen_event)}")
            images.append(screen_event)

            # Replace message with simple image placeholder - EpisodeTokenizer handles prefix/suffix/repetition
            mcap_message.message = self.config.fake_image_placeholder.encode("utf-8")

        return f"<EVENT_START>{mcap_message.model_dump_json()}<EVENT_END>", images

    def decode(self, encoded_data: str, images: Optional[List[ScreenCaptured]] = None) -> McapMessage:
        """Decode JSON event back to McapMessage format."""
        if not encoded_data.startswith("<EVENT_START>") or not encoded_data.endswith("<EVENT_END>"):
            raise ValueError("Invalid format: missing <EVENT_START> or <EVENT_END> tokens")

        content = encoded_data[len("<EVENT_START>") : -len("<EVENT_END>")]

        try:
            event_dict = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON content: {e}")

        # Handle screen events with image data
        if (
            event_dict.get("topic") == "screen"
            and event_dict.get("message_type") == "desktop/ScreenCaptured"
            and event_dict.get("message") == self.config.fake_image_placeholder
        ):
            if not images:
                raise ValueError("Screen event requires image data but none provided")
            image_data = images[0]
            event_dict["message"] = image_data.model_dump_json(exclude={"frame_arr"})

        return McapMessage(
            topic=event_dict["topic"],
            timestamp=event_dict["timestamp"],
            message_type=event_dict["message_type"],
            message=event_dict["message"].encode("utf-8")
            if isinstance(event_dict["message"], str)
            else event_dict["message"],
        )

    def get_vocab(self) -> Set[str]:
        """Get all tokens in the vocabulary.

        Note: fake_image_placeholder is NOT included as it's not a real token,
        just an internal placeholder used during encoding.
        """
        return set()  # No real tokens in JSON encoder vocab
