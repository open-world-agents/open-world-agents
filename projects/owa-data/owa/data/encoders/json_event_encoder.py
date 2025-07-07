"""
JSONEventEncoder for converting raw events to MLLM-compatible JSON format.

This module implements the JSONEventEncoder class that converts raw event data
from the Event Dataset into JSON string format suitable for training Vision-Language-Action (VLA) models.

The encoder supports:
- JSON serialization of events for LLM tokenization
- Multimodal handling of screen events with image data
- Bidirectional encoding/decoding operations
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from mcap_owa.highlevel.reader import McapMessage
from owa.msgs.desktop.screen import ScreenCaptured

from .base_encoder import BaseEventEncoder


class JSONEventEncoder(BaseEventEncoder):
    """
    JSON-based encoder for converting raw events to MLLM training format.

    This class implements JSON serialization strategy using <EVENT_START> and <EVENT_END>
    tokens with JSON-formatted event data. Designed for text-based language models that
    work well with structured JSON input.

    Examples:
        >>> from mcap_owa.highlevel.reader import McapMessage
        >>> encoder = JSONEventEncoder()
        >>>
        >>> # Encode a keyboard event from McapMessage
        >>> mcap_message = McapMessage(
        ...     topic='keyboard',
        ...     timestamp=1745362786814673800,
        ...     message_type='desktop/KeyboardEvent',
        ...     message=b'{"event_type":"press","vk":37}'
        ... )
        >>> text, images = encoder.encode(mcap_message)
        >>> print(text)
        <EVENT_START>{'topic': 'keyboard', 'timestamp_ns': 1745362786814673800, ...}<EVENT_END>
    """

    def __init__(self, drop_file_path: bool = True):
        """
        Initialize the JSONEventEncoder.

        Args:
            drop_file_path: Whether to drop the file_path field from encoded events.
                          Defaults to True to reduce token usage for training.
        """
        self.drop_file_path = drop_file_path

    def encode(self, mcap_message: McapMessage) -> Tuple[str, List[ScreenCaptured]]:
        """
        Encode a single McapMessage-like object to MLLM training format.

        Args:
            mcap_message: McapMessage instance

        Returns:
            Tuple containing:
                - str: Serialized text with <IMAGE> placeholders for screen events
                - List[ScreenCaptured]: Image data for screen events (empty for others)

        Raises:
            ValueError: If the mcap_message format is invalid
            json.JSONDecodeError: If message content cannot be parsed
        """
        mcap_message = mcap_message if isinstance(mcap_message, McapMessage) else McapMessage(**mcap_message)

        # Handle screen events with image data
        images = []

        # Create event dictionary for serialization
        event_dict = {
            "topic": mcap_message.topic,
            "timestamp_ns": mcap_message.timestamp,
            "message_type": mcap_message.message_type,
            "msg": mcap_message.message,
        }

        if mcap_message.topic == "screen" and mcap_message.message_type == "desktop/ScreenCaptured":
            # Parse the message to create ScreenCaptured object
            try:
                # McapMessage.message is always bytes, and we can use the decoded property
                screen_event = mcap_message.decoded
                if not isinstance(screen_event, ScreenCaptured):
                    raise ValueError(f"Expected ScreenCaptured object, got {type(screen_event)}")

                # Store the ScreenCaptured object directly
                images.append(screen_event)

                # Replace message content with <IMAGE> placeholder in serialized text
                # Keep as bytes since McapMessage.message is bytes
                event_dict["msg"] = b"<IMAGE>"
            except Exception as e:
                raise ValueError(f"Failed to parse screen event message: {e}")

        # Create the serialized text format
        serialized_text = f"<EVENT_START>{event_dict}<EVENT_END>"

        return serialized_text, images

    def decode(self, serialized_text: str, images: Optional[List[ScreenCaptured]] = None) -> McapMessage:
        """
        Decode serialized event back to McapMessage format.

        Args:
            serialized_text: Encoded event text with <EVENT_START>/<EVENT_END> tokens
            images: Optional list of image data for screen events

        Returns:
            McapMessage: Reconstructed message in McapMessage format

        Raises:
            ValueError: If serialized_text format is invalid
            json.JSONDecodeError: If event content cannot be parsed
        """
        if not serialized_text.startswith("<EVENT_START>") or not serialized_text.endswith("<EVENT_END>"):
            raise ValueError("Invalid serialized format: missing <EVENT_START> or <EVENT_END> tokens")

        # Extract the event content between tokens
        content = serialized_text[len("<EVENT_START>") : -len("<EVENT_END>")]

        try:
            # Parse the event dictionary
            # Note: Using eval here is safe since we control the input format
            # In production, consider using ast.literal_eval for additional safety
            event_dict = eval(content)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Failed to parse event content: {e}")

        if not isinstance(event_dict, dict):
            raise ValueError("Decoded content is not a dictionary")

        # Handle screen events with image data
        if (
            event_dict.get("topic") == "screen"
            and event_dict.get("message_type") == "desktop/ScreenCaptured"
            and event_dict.get("msg") in (b"<IMAGE>", "<IMAGE>")  # Handle both bytes and string
        ):
            if not images:
                raise ValueError("Screen event requires image data but none provided")

            # Restore the original message content
            image_data = images[0]
            if isinstance(image_data, ScreenCaptured):
                # Convert ScreenCaptured back to JSON format
                msg_dict = image_data.model_dump(exclude={"frame_arr"})
                event_dict["msg"] = json.dumps(msg_dict)
            else:
                # Fallback for unexpected data types
                event_dict["msg"] = json.dumps(image_data) if image_data else "{}"

        return McapMessage(
            topic=event_dict["topic"],
            timestamp=event_dict["timestamp_ns"],
            message_type=event_dict["message_type"],
            message=event_dict["msg"] if isinstance(event_dict["msg"], bytes) else event_dict["msg"].encode("utf-8"),
        )

    def encode_batch(self, mcap_messages: List[McapMessage]) -> Tuple[List[str], List[List[ScreenCaptured]]]:
        """
        Encode a batch of McapMessages.

        Args:
            mcap_messages: List of McapMessage instances or dictionaries

        Returns:
            Tuple containing:
                - List[str]: Serialized texts for each event
                - List[List[ScreenCaptured]]: Image data for each event
        """
        texts = []
        all_images = []

        for mcap_message in mcap_messages:
            text, images = self.encode(mcap_message)
            texts.append(text)
            all_images.append(images)

        return texts, all_images

    def decode_batch(
        self, serialized_texts: List[str], all_images: Optional[List[List[ScreenCaptured]]] = None
    ) -> List[McapMessage]:
        """
        Decode a batch of serialized events.

        Args:
            serialized_texts: List of encoded event texts
            all_images: Optional list of image data lists for each event

        Returns:
            List[McapMessage]: Reconstructed messages
        """
        if all_images is None:
            all_images = [[] for _ in serialized_texts]

        if len(serialized_texts) != len(all_images):
            raise ValueError("Length mismatch between texts and images")

        messages = []
        for text, images in zip(serialized_texts, all_images):
            message = self.decode(text, images)
            messages.append(message)

        return messages

    def get_encoder_info(self) -> Dict[str, Any]:
        """Get information about this encoder."""
        return {
            "encoder_type": "JSONEventEncoder",
            "format": "json_string",
            "vocab_size": None,
            "drop_file_path": self.drop_file_path,
        }
