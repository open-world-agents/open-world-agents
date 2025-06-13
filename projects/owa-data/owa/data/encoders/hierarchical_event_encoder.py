"""
HierarchicalEventEncoder for converting raw events to hierarchical token format.

This module implements the HierarchicalEventEncoder class that converts raw event data
from the Event Dataset into hierarchical token sequences optimized for VLA training.

The encoder uses a compositional token structure:
- <TIMESTAMP><index> instead of <TIMESTAMP_index>
- <KEYBOARD><vk><action> instead of <KEYBOARD_vk_action>
- <MOUSE><action><params...> instead of <MOUSE_action_params>

This approach reduces vocabulary size by ~95% while maintaining full expressiveness.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from owa.core.time import TimeUnits
from owa.env.desktop.msg import KeyboardEvent, MouseEvent
from owa.env.gst.msg import ScreenEmitted

from .base_encoder import BaseEventEncoder


class HierarchicalVocabulary:
    """Manages the hierarchical token vocabulary for efficient encoding/decoding."""

    def __init__(self):
        # Base event type tokens
        self.base_tokens = {
            "<TIMESTAMP>": 0,
            "<KEYBOARD>": 1,
            "<MOUSE>": 2,
            "<SCREEN>": 3,
            "<PAD>": 4,
            "<UNK>": 5,
        }

        # Parameter tokens (shared across event types)
        self.param_tokens = {}
        offset = len(self.base_tokens)

        # Numbers 0-255 for various parameters (vk codes, coordinates, etc.)
        for i in range(256):
            self.param_tokens[f"<{i}>"] = offset + i
        offset += 256

        # Action types
        action_tokens = ["<press>", "<release>", "<move>", "<click>", "<scroll>"]
        for i, token in enumerate(action_tokens):
            self.param_tokens[token] = offset + i
        offset += len(action_tokens)

        # Mouse buttons
        button_tokens = ["<left>", "<right>", "<middle>", "<unknown>"]
        for i, token in enumerate(button_tokens):
            self.param_tokens[token] = offset + i
        offset += len(button_tokens)

        # Special tokens for negative numbers (scroll deltas)
        for i in range(-10, 11):  # -10 to +10 for scroll deltas
            self.param_tokens[f"<{i}>"] = offset
            offset += 1

        self.vocab_size = offset

        # Create reverse mapping
        self.id_to_token = {}
        for token, token_id in self.base_tokens.items():
            self.id_to_token[token_id] = token
        for token, token_id in self.param_tokens.items():
            self.id_to_token[token_id] = token

    def encode_token(self, token: str) -> int:
        """Convert token string to token ID."""
        if token in self.base_tokens:
            return self.base_tokens[token]
        elif token in self.param_tokens:
            return self.param_tokens[token]
        else:
            return self.base_tokens["<UNK>"]

    def decode_token(self, token_id: int) -> str:
        """Convert token ID to token string."""
        return self.id_to_token.get(token_id, "<UNK>")

    def get_vocab_size(self) -> int:
        """Get total vocabulary size."""
        return self.vocab_size


class HierarchicalEventEncoderConfig:
    """Configuration for HierarchicalEventEncoder."""

    def __init__(
        self,
        timestamp_min_ns: int = -2 * TimeUnits.SECOND,
        timestamp_max_ns: int = 2 * TimeUnits.SECOND,
        timestamp_interval_ns: int = 20 * TimeUnits.MSECOND,  # 50fps
        mouse_move_bins: List[int] = None,
        screen_size: Tuple[int, int] = (1920, 1080),
        drop_file_path: bool = True,
    ):
        self.timestamp_min_ns = timestamp_min_ns
        self.timestamp_max_ns = timestamp_max_ns
        self.timestamp_interval_ns = timestamp_interval_ns
        self.mouse_move_bins = mouse_move_bins or [16, 16, 16]  # 3-level residual quantization
        self.screen_size = screen_size
        self.drop_file_path = drop_file_path

        # Initialize vocabulary
        self.vocabulary = HierarchicalVocabulary()

    @property
    def timestamp_count(self) -> int:
        """Number of timestamp bins."""
        return ((self.timestamp_max_ns - self.timestamp_min_ns) // self.timestamp_interval_ns) + 1


class HierarchicalMouseProcessor:
    """Processes mouse events with hierarchical residual quantization."""

    def __init__(self, config: HierarchicalEventEncoderConfig):
        self.config = config

    def _limit(self, x: float, low: float = 0.0, high: float = 1.0) -> float:
        """Limit x to the range [low, high]."""
        return max(low, min(x, high))

    def encode_move(self, event: MouseEvent, screen_size: Optional[Tuple[int, int]] = None) -> List[str]:
        """
        Encode mouse movement with hierarchical residual quantization.
        Returns: [<MOUSE>, <move>, <level0_x>, <level0_y>, <level1_x>, <level1_y>, ...]
        """
        if screen_size is None:
            screen_size = self.config.screen_size

        x, y = event.x, event.y
        fx = self._limit(x / screen_size[0])
        fy = self._limit(y / screen_size[1])

        tokens = ["<MOUSE>", "<move>"]

        # Jointly quantize the pair (x, y) repeatedly at each level
        vx, vy = fx, fy
        for i, nbins in enumerate(self.config.mouse_move_bins):
            # Using floor for better accuracy
            idx_x = int(vx * nbins)
            idx_y = int(vy * nbins)
            tokens.extend([f"<{idx_x}>", f"<{idx_y}>"])

            # Calculate residuals for next level
            vx = vx * nbins - idx_x
            vy = vy * nbins - idx_y

        return tokens

    def decode_move(self, tokens: List[str], screen_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Decode hierarchical mouse movement tokens back to coordinates."""
        if screen_size is None:
            screen_size = self.config.screen_size

        # Extract coordinate pairs from tokens (skip <MOUSE> and <move>)
        coord_tokens = tokens[2:]  # Skip base tokens

        if len(coord_tokens) != len(self.config.mouse_move_bins) * 2:
            raise ValueError(
                f"Expected {len(self.config.mouse_move_bins) * 2} coordinate tokens, got {len(coord_tokens)}"
            )

        # Parse coordinate pairs
        indices = []
        for i in range(0, len(coord_tokens), 2):
            x_token = coord_tokens[i]
            y_token = coord_tokens[i + 1]

            # Extract numbers from tokens like <15>
            x_match = re.match(r"<(\d+)>", x_token)
            y_match = re.match(r"<(\d+)>", y_token)

            if not x_match or not y_match:
                raise ValueError(f"Invalid coordinate tokens: {x_token}, {y_token}")

            idx_x = int(x_match.group(1))
            idx_y = int(y_match.group(1))
            indices.append((idx_x, idx_y))

        # Reconstruct coordinates using residual quantization
        fx = fy = 0.0
        for i in reversed(range(len(indices))):
            idx_x, idx_y = indices[i]
            nbins = self.config.mouse_move_bins[i]

            fx = (fx + idx_x) / nbins
            fy = (fy + idx_y) / nbins

        # Convert to pixel coordinates
        pix_x = int(round(fx * (screen_size[0] - 1)))
        pix_y = int(round(fy * (screen_size[1] - 1)))

        return pix_x, pix_y

    def encode_click(self, event: MouseEvent) -> List[str]:
        """Encode mouse click: [<MOUSE>, <click>, <button>, <action>]"""
        button = event.button or "unknown"
        action = "press" if bool(event.pressed) else "release"
        return ["<MOUSE>", "<click>", f"<{button}>", f"<{action}>"]

    def encode_scroll(self, event: MouseEvent) -> List[str]:
        """Encode mouse scroll: [<MOUSE>, <scroll>, <dx>, <dy>]"""
        dx = event.dx if event.dx is not None else 0
        dy = event.dy if event.dy is not None else 0
        return ["<MOUSE>", "<scroll>", f"<{dx}>", f"<{dy}>"]

    def encode_mouse_event(self, event: MouseEvent, screen_size: Optional[Tuple[int, int]] = None) -> List[str]:
        """Encode any mouse event with position + action."""
        # Always include position
        tokens = self.encode_move(event, screen_size)

        # Add specific action
        if event.event_type == "move":
            return tokens
        elif event.event_type == "click":
            return tokens + self.encode_click(event)[2:]  # Skip <MOUSE>, <click> prefix
        elif event.event_type == "scroll":
            return tokens + self.encode_scroll(event)[2:]  # Skip <MOUSE>, <scroll> prefix
        else:
            return tokens + ["<UNK>"]


class HierarchicalEventEncoder(BaseEventEncoder):
    """
    Hierarchical event encoder for VLA training with compositional token structure.

    This encoder converts raw events to hierarchical token sequences that are more
    efficient and learnable than traditional flat token approaches.

    Examples:
        >>> config = HierarchicalEventEncoderConfig()
        >>> encoder = HierarchicalEventEncoder(config)
        >>>
        >>> # Encode a keyboard event
        >>> raw_event = {
        ...     'topic': 'keyboard',
        ...     'timestamp_ns': 1745362786814673800,
        ...     'message_type': 'owa.env.desktop.msg.KeyboardEvent',
        ...     'msg': '{"event_type":"press","vk":65}'
        ... }
        >>> tokens, images = encoder.encode(raw_event)
        >>> print(tokens)
        ['<TIMESTAMP>', '<123>', '<KEYBOARD>', '<65>', '<press>']
    """

    def __init__(self, config: Optional[HierarchicalEventEncoderConfig] = None):
        """Initialize the hierarchical event encoder."""
        self.config = config or HierarchicalEventEncoderConfig()
        self.mouse_processor = HierarchicalMouseProcessor(self.config)

    def _encode_timestamp(self, timestamp_ns: int) -> List[str]:
        """Encode timestamp to hierarchical tokens: [<TIMESTAMP>, <index>]"""
        # Normalize timestamp to config range
        mod = timestamp_ns % (self.config.timestamp_max_ns - self.config.timestamp_min_ns)
        idx = mod // self.config.timestamp_interval_ns

        # Ensure index is within bounds
        max_idx = self.config.timestamp_count - 1
        idx = min(max_idx, max(0, idx))

        return ["<TIMESTAMP>", f"<{idx}>"]

    def _decode_timestamp(self, tokens: List[str]) -> int:
        """Decode timestamp tokens back to nanoseconds."""
        if len(tokens) != 2 or tokens[0] != "<TIMESTAMP>":
            raise ValueError(f"Invalid timestamp tokens: {tokens}")

        # Extract index from token like <123>
        idx_match = re.match(r"<(\d+)>", tokens[1])
        if not idx_match:
            raise ValueError(f"Invalid timestamp index token: {tokens[1]}")

        idx = int(idx_match.group(1))
        timestamp_ns = self.config.timestamp_min_ns + idx * self.config.timestamp_interval_ns
        return timestamp_ns

    def _encode_keyboard(self, event: KeyboardEvent) -> List[str]:
        """Encode keyboard event: [<KEYBOARD>, <vk>, <action>]"""
        return ["<KEYBOARD>", f"<{event.vk}>", f"<{event.event_type}>"]

    def _decode_keyboard(self, tokens: List[str]) -> KeyboardEvent:
        """Decode keyboard tokens back to KeyboardEvent."""
        if len(tokens) != 3 or tokens[0] != "<KEYBOARD>":
            raise ValueError(f"Invalid keyboard tokens: {tokens}")

        # Extract vk code
        vk_match = re.match(r"<(\d+)>", tokens[1])
        if not vk_match:
            raise ValueError(f"Invalid vk token: {tokens[1]}")
        vk = int(vk_match.group(1))

        # Extract action
        action_match = re.match(r"<(\w+)>", tokens[2])
        if not action_match:
            raise ValueError(f"Invalid action token: {tokens[2]}")
        event_type = action_match.group(1)

        if event_type not in ("press", "release"):
            raise ValueError(f"Invalid keyboard event type: {event_type}")

        return KeyboardEvent(event_type=event_type, vk=vk)

    def _decode_mouse(self, tokens: List[str], screen_size: Optional[Tuple[int, int]] = None) -> MouseEvent:
        """Decode mouse tokens back to MouseEvent."""
        if len(tokens) < 2 or tokens[0] != "<MOUSE>":
            raise ValueError(f"Invalid mouse tokens: {tokens}")

        # Find the action type
        action_token = tokens[1]
        if action_token == "<move>":
            # Pure movement
            move_tokens = tokens[: 2 + len(self.config.mouse_move_bins) * 2]
            x, y = self.mouse_processor.decode_move(move_tokens, screen_size)
            return MouseEvent(event_type="move", x=x, y=y)

        elif action_token == "<click>" or action_token == "<scroll>":
            # Movement + action
            move_end_idx = 2 + len(self.config.mouse_move_bins) * 2
            move_tokens = ["<MOUSE>", "<move>"] + tokens[2:move_end_idx]
            x, y = self.mouse_processor.decode_move(move_tokens, screen_size)

            if action_token == "<click>":
                # Extract button and press/release
                if len(tokens) < move_end_idx + 2:
                    raise ValueError("Insufficient tokens for mouse click")

                button_token = tokens[move_end_idx]
                action_token = tokens[move_end_idx + 1]

                button_match = re.match(r"<(\w+)>", button_token)
                action_match = re.match(r"<(\w+)>", action_token)

                if not button_match or not action_match:
                    raise ValueError(f"Invalid click tokens: {button_token}, {action_token}")

                button = button_match.group(1)
                pressed = action_match.group(1) == "press"

                return MouseEvent(event_type="click", x=x, y=y, button=button, pressed=pressed)

            elif action_token == "<scroll>":
                # Extract dx and dy
                if len(tokens) < move_end_idx + 2:
                    raise ValueError("Insufficient tokens for mouse scroll")

                dx_token = tokens[move_end_idx]
                dy_token = tokens[move_end_idx + 1]

                dx_match = re.match(r"<(-?\d+)>", dx_token)
                dy_match = re.match(r"<(-?\d+)>", dy_token)

                if not dx_match or not dy_match:
                    raise ValueError(f"Invalid scroll tokens: {dx_token}, {dy_token}")

                dx = int(dx_match.group(1))
                dy = int(dy_match.group(1))

                return MouseEvent(event_type="scroll", x=x, y=y, dx=dx, dy=dy)

        raise ValueError(f"Unknown mouse action: {action_token}")

    def encode(self, raw_event: Dict[str, Any]) -> Tuple[List[str], List[Union[ScreenEmitted, Dict]]]:
        """
        Encode a single raw event to hierarchical token format.

        Args:
            raw_event: Raw event dictionary with keys:
                - topic: Event topic (e.g., 'keyboard', 'screen')
                - timestamp_ns: Timestamp in nanoseconds
                - message_type: Full message type identifier
                - msg: Serialized message content (bytes or string)
                - file_path: Source MCAP file path (optional)

        Returns:
            Tuple containing:
                - List[str]: Hierarchical token sequence
                - List[Union[ScreenEmitted, Dict]]: Image data for screen events (empty for others)

        Raises:
            ValueError: If the raw_event format is invalid
            json.JSONDecodeError: If message content cannot be parsed
        """
        if not isinstance(raw_event, dict):
            raise ValueError("raw_event must be a dictionary")

        required_keys = {"topic", "timestamp_ns", "message_type", "msg"}
        if not self.config.drop_file_path:
            required_keys.add("file_path")
        if not required_keys.issubset(raw_event.keys()):
            missing = required_keys - raw_event.keys()
            raise ValueError(f"raw_event missing required keys: {missing}")

        # Start with timestamp
        tokens = self._encode_timestamp(raw_event["timestamp_ns"])
        images = []

        # Parse message content
        try:
            if isinstance(raw_event["msg"], bytes):
                msg_data = json.loads(raw_event["msg"].decode("utf-8"))
            else:
                msg_data = json.loads(raw_event["msg"])
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Failed to parse message content: {e}")

        # Encode based on event type
        if raw_event["topic"] == "keyboard" and raw_event["message_type"] == "owa.env.desktop.msg.KeyboardEvent":
            keyboard_event = KeyboardEvent(**msg_data)
            tokens.extend(self._encode_keyboard(keyboard_event))

        elif raw_event["topic"] == "mouse" and raw_event["message_type"] == "owa.env.desktop.msg.MouseEvent":
            mouse_event = MouseEvent(**msg_data)
            tokens.extend(self.mouse_processor.encode_mouse_event(mouse_event))

        elif raw_event["topic"] == "screen" and raw_event["message_type"] == "owa.env.gst.msg.ScreenEmitted":
            screen_event = ScreenEmitted(**msg_data)
            tokens.append("<SCREEN>")
            # Store image data
            images.append({"screen_event": screen_event, "original_msg": raw_event["msg"]})

        else:
            tokens.append("<UNK>")

        return tokens, images

    def decode(
        self,
        tokens: List[str],
        images: Optional[List[Union[ScreenEmitted, Dict]]] = None,
        screen_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Decode hierarchical tokens back to original raw event format.

        Args:
            tokens: Hierarchical token sequence
            images: Optional list of image data for screen events
            screen_size: Optional screen size for mouse coordinate decoding

        Returns:
            Dict: Reconstructed raw event in original format

        Raises:
            ValueError: If token sequence format is invalid
        """
        if not tokens or len(tokens) < 2:
            raise ValueError("Token sequence too short")

        # Decode timestamp (first 2 tokens)
        timestamp_ns = self._decode_timestamp(tokens[:2])

        # Determine event type and decode accordingly
        if len(tokens) < 3:
            # Only timestamp, create unknown event
            return {
                "topic": "unknown",
                "timestamp_ns": timestamp_ns,
                "message_type": "unknown",
                "msg": "{}",
            }

        event_type_token = tokens[2]

        if event_type_token == "<KEYBOARD>":
            # Decode keyboard event
            if len(tokens) < 5:
                raise ValueError("Insufficient tokens for keyboard event")

            keyboard_event = self._decode_keyboard(tokens[2:5])
            msg_data = {"event_type": keyboard_event.event_type, "vk": keyboard_event.vk}

            return {
                "topic": "keyboard",
                "timestamp_ns": timestamp_ns,
                "message_type": "owa.env.desktop.msg.KeyboardEvent",
                "msg": json.dumps(msg_data),
            }

        elif event_type_token == "<MOUSE>":
            # Decode mouse event
            mouse_tokens = tokens[2:]  # Skip timestamp
            mouse_event = self._decode_mouse(mouse_tokens, screen_size)

            msg_data = {
                "event_type": mouse_event.event_type,
                "x": mouse_event.x,
                "y": mouse_event.y,
            }

            if mouse_event.button is not None:
                msg_data["button"] = mouse_event.button
            if mouse_event.pressed is not None:
                msg_data["pressed"] = mouse_event.pressed
            if mouse_event.dx is not None:
                msg_data["dx"] = mouse_event.dx
            if mouse_event.dy is not None:
                msg_data["dy"] = mouse_event.dy

            return {
                "topic": "mouse",
                "timestamp_ns": timestamp_ns,
                "message_type": "owa.env.desktop.msg.MouseEvent",
                "msg": json.dumps(msg_data),
            }

        elif event_type_token == "<SCREEN>":
            # Decode screen event
            if not images:
                raise ValueError("Screen event requires image data but none provided")

            image_data = images[0]
            if isinstance(image_data, dict) and "original_msg" in image_data:
                # Use preserved original message for exact round-trip consistency
                msg = image_data["original_msg"]
            elif isinstance(image_data, ScreenEmitted):
                # Fallback: convert ScreenEmitted back to JSON
                msg_dict = image_data.model_dump(exclude={"frame_arr"})
                msg = json.dumps(msg_dict)
            elif isinstance(image_data, dict):
                # Fallback: assume it's a message dict
                msg = json.dumps(image_data)
            else:
                msg = "{}"

            return {
                "topic": "screen",
                "timestamp_ns": timestamp_ns,
                "message_type": "owa.env.gst.msg.ScreenEmitted",
                "msg": msg,
            }

        else:
            # Unknown event type
            return {
                "topic": "unknown",
                "timestamp_ns": timestamp_ns,
                "message_type": "unknown",
                "msg": "{}",
            }

    def encode_batch(
        self, raw_events: List[Dict[str, Any]]
    ) -> Tuple[List[List[str]], List[List[Union[ScreenEmitted, Dict]]]]:
        """
        Encode a batch of raw events.

        Args:
            raw_events: List of raw event dictionaries

        Returns:
            Tuple containing:
                - List[List[str]]: Hierarchical token sequences for each event
                - List[List[Union[ScreenEmitted, Dict]]]: Image data for each event
        """
        all_tokens = []
        all_images = []

        for event in raw_events:
            tokens, images = self.encode(event)
            all_tokens.append(tokens)
            all_images.append(images)

        return all_tokens, all_images

    def decode_batch(
        self,
        all_tokens: List[List[str]],
        all_images: Optional[List[List[Union[ScreenEmitted, Dict]]]] = None,
        screen_size: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Decode a batch of hierarchical token sequences.

        Args:
            all_tokens: List of hierarchical token sequences
            all_images: Optional list of image data lists for each event
            screen_size: Optional screen size for mouse coordinate decoding

        Returns:
            List[Dict]: Reconstructed raw events
        """
        if all_images is None:
            all_images = [[] for _ in all_tokens]

        if len(all_tokens) != len(all_images):
            raise ValueError("Length mismatch between tokens and images")

        events = []
        for tokens, images in zip(all_tokens, all_images):
            event = self.decode(tokens, images, screen_size)
            events.append(event)

        return events

    def get_vocab_size(self) -> int:
        """Get the total vocabulary size."""
        return self.config.vocabulary.get_vocab_size()

    def get_token_ids(self, tokens: List[str]) -> List[int]:
        """Convert token strings to token IDs."""
        return [self.config.vocabulary.encode_token(token) for token in tokens]

    def get_tokens_from_ids(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs to token strings."""
        return [self.config.vocabulary.decode_token(token_id) for token_id in token_ids]
