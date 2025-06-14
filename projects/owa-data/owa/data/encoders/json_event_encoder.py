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
from typing import Any, Dict, List, Optional, Tuple, Union

from owa.env.gst.msg import ScreenEmitted

from .base_encoder import BaseEventEncoder


class JSONEventEncoder(BaseEventEncoder):
    """
    JSON-based encoder for converting raw events to MLLM training format.

    This class implements JSON serialization strategy using <EVENT_START> and <EVENT_END>
    tokens with JSON-formatted event data. Designed for text-based language models that
    work well with structured JSON input.

    Examples:
        >>> # Default: drop file_path to reduce token usage
        >>> encoder = JSONEventEncoder()
        >>>
        >>> # Encode a keyboard event
        >>> raw_event = {
        ...     'file_path': '/path/to/file.mcap',
        ...     'topic': 'keyboard',
        ...     'timestamp_ns': 1745362786814673800,
        ...     'message_type': 'owa.env.desktop.msg.KeyboardEvent',
        ...     'msg': b'{"event_type":"press","vk":37}'
        ... }
        >>> text, images = encoder.encode(raw_event)
        >>> print(text)
        <EVENT_START>{'topic': 'keyboard', 'timestamp_ns': 1745362786814673800, ...}<EVENT_END>
        >>>
        >>> # Keep file_path if needed
        >>> encoder_with_path = JSONEventEncoder(drop_file_path=False)
        >>> text, images = encoder_with_path.encode(raw_event)
        >>> # Now file_path is preserved in the encoded text
    """

    def __init__(self, drop_file_path: bool = True):
        """
        Initialize the JSONEventEncoder.

        Args:
            drop_file_path: Whether to drop the file_path field from encoded events.
                          Defaults to True to reduce token usage for training.
        """
        self.drop_file_path = drop_file_path

    def encode(self, raw_event: Dict[str, Any]) -> Tuple[str, List[Union[ScreenEmitted, Dict]]]:
        """
        Encode a single raw event to MLLM training format.

        Args:
            raw_event: Raw event dictionary with keys:
                - file_path: Source MCAP file path
                - topic: Event topic (e.g., 'keyboard', 'screen')
                - timestamp_ns: Timestamp in nanoseconds
                - message_type: Full message type identifier
                - msg: Serialized message content (bytes or string)

        Returns:
            Tuple containing:
                - str: Serialized text with <IMAGE> placeholders for screen events
                - List[Union[ScreenEmitted, Dict]]: Image data for screen events (empty for others)

        Raises:
            ValueError: If the raw_event format is invalid
            json.JSONDecodeError: If message content cannot be parsed
        """
        if not isinstance(raw_event, dict):
            raise ValueError("raw_event must be a dictionary")

        required_keys = {"topic", "timestamp_ns", "message_type", "msg"}
        if not self.drop_file_path:
            required_keys.add("file_path")
        if not required_keys.issubset(raw_event.keys()):
            missing = required_keys - raw_event.keys()
            raise ValueError(f"raw_event missing required keys: {missing}")

        # Handle screen events with image data
        images = []
        event_copy = raw_event.copy()

        # Drop file_path if requested
        if self.drop_file_path and "file_path" in event_copy:
            del event_copy["file_path"]

        if raw_event["topic"] == "screen" and raw_event["message_type"] == "owa.env.gst.msg.ScreenEmitted":
            # Parse the message to create ScreenEmitted object
            try:
                # Handle both bytes and string msg formats
                if isinstance(raw_event["msg"], bytes):
                    msg_data = json.loads(raw_event["msg"].decode("utf-8"))
                else:
                    msg_data = json.loads(raw_event["msg"])
                screen_event = ScreenEmitted(**msg_data)

                # Store both the ScreenEmitted object and the original message data
                # This preserves the exact original format for round-trip consistency
                images.append({"screen_event": screen_event, "original_msg": raw_event["msg"]})

                # Replace message content with <IMAGE> placeholder in serialized text
                # Use same format as original (bytes or string)
                if isinstance(raw_event["msg"], bytes):
                    event_copy["msg"] = b"<IMAGE>"
                else:
                    event_copy["msg"] = "<IMAGE>"
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"Failed to parse screen event message: {e}")

        # Create the serialized text format
        serialized_text = f"<EVENT_START>{event_copy}<EVENT_END>"

        return serialized_text, images

    def decode(
        self, serialized_text: str, images: Optional[List[Union[ScreenEmitted, Dict]]] = None
    ) -> Dict[str, Any]:
        """
        Decode serialized event back to original raw event format.

        Args:
            serialized_text: Encoded event text with <EVENT_START>/<EVENT_END> tokens
            images: Optional list of image data for screen events

        Returns:
            Dict: Reconstructed raw event in original format

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
            and event_dict.get("message_type") == "owa.env.gst.msg.ScreenEmitted"
            and event_dict.get("msg") in (b"<IMAGE>", "<IMAGE>")  # Handle both bytes and string
        ):
            if not images:
                raise ValueError("Screen event requires image data but none provided")

            # Restore the original message content
            image_data = images[0]
            if isinstance(image_data, dict) and "original_msg" in image_data:
                # Use the preserved original message for exact round-trip consistency
                event_dict["msg"] = image_data["original_msg"]
            elif isinstance(image_data, ScreenEmitted):
                # Fallback: convert ScreenEmitted back to JSON (string format for new pipeline)
                msg_dict = image_data.model_dump(exclude={"frame_arr"})
                event_dict["msg"] = json.dumps(msg_dict)
            elif isinstance(image_data, dict):
                # Fallback: assume it's a message dict (string format for new pipeline)
                event_dict["msg"] = json.dumps(image_data)

        # Add back file_path if it was dropped during encoding
        if self.drop_file_path and "file_path" not in event_dict:
            event_dict["file_path"] = "<DROPPED>"

        return event_dict

    def encode_batch(
        self, raw_events: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[List[Union[ScreenEmitted, Dict]]]]:
        """
        Encode a batch of raw events.

        Args:
            raw_events: List of raw event dictionaries

        Returns:
            Tuple containing:
                - List[str]: Serialized texts for each event
                - List[List[Union[ScreenEmitted, Dict]]]: Image data for each event
        """
        texts = []
        all_images = []

        for event in raw_events:
            text, images = self.encode(event)
            texts.append(text)
            all_images.append(images)

        return texts, all_images

    def decode_batch(
        self, serialized_texts: List[str], all_images: Optional[List[List[Union[ScreenEmitted, Dict]]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Decode a batch of serialized events.

        Args:
            serialized_texts: List of encoded event texts
            all_images: Optional list of image data lists for each event

        Returns:
            List[Dict]: Reconstructed raw events
        """
        if all_images is None:
            all_images = [[] for _ in serialized_texts]

        if len(serialized_texts) != len(all_images):
            raise ValueError("Length mismatch between texts and images")

        events = []
        for text, images in zip(serialized_texts, all_images):
            event = self.decode(text, images)
            events.append(event)

        return events

    def get_encoder_info(self) -> Dict[str, Any]:
        """Get information about this encoder."""
        return {
            "encoder_type": "JSONEventEncoder",
            "format": "json_string",
            "vocab_size": None,
            "drop_file_path": self.drop_file_path,
        }
