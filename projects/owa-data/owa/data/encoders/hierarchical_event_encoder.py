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
from typing import List, Optional, Tuple

from mcap_owa.highlevel.reader import McapMessage
from owa.core.time import TimeUnits
from owa.msgs.desktop.keyboard import KeyboardEvent
from owa.msgs.desktop.mouse import MouseEvent
from owa.msgs.desktop.screen import ScreenCaptured

from .base_encoder import BaseEventEncoder


class HierarchicalVocabulary:
    """Manages the hierarchical token vocabulary for efficient encoding/decoding."""

    def __init__(self):
        # Base event type tokens
        self.base_tokens = {
            "<EVENT_START>": 0,
            "<EVENT_END>": 1,
            "<TIMESTAMP>": 2,
            "<KEYBOARD>": 3,
            "<MOUSE>": 4,
            "<SCREEN>": 5,
            "<PAD>": 6,
            "<UNK>": 7,
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
        ...     'message_type': 'desktop/KeyboardEvent',
        ...     'msg': '{"event_type":"press","vk":65}'
        ... }
        >>> tokens, images = encoder.encode(raw_event)
        >>> print(tokens)
        '<EVENT_START><TIMESTAMP><123><KEYBOARD><65><press><EVENT_END>'
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

    def encode(self, mcap_message: McapMessage) -> Tuple[str, List[ScreenCaptured]]:
        """
        Encode a single McapMessage object to hierarchical token format.

        Args:
            mcap_message: McapMessage instance

        Returns:
            Tuple containing:
                - str: Hierarchical token sequence concatenated without spaces
                - List[ScreenCaptured]: Image data for screen events (empty for others)

        Raises:
            ValueError: If the mcap_message format is invalid
            json.JSONDecodeError: If message content cannot be parsed
        """
        mcap_message = mcap_message if isinstance(mcap_message, McapMessage) else McapMessage(**mcap_message)

        # Start with timestamp
        tokens = self._encode_timestamp(mcap_message.timestamp)
        images = []

        # Parse message content
        try:
            if isinstance(mcap_message.message, bytes):
                msg_data = json.loads(mcap_message.message.decode("utf-8"))
            else:
                msg_data = json.loads(mcap_message.message)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Failed to parse message content: {e}")

        # Encode based on event type
        if mcap_message.topic == "keyboard" and mcap_message.message_type == "desktop/KeyboardEvent":
            keyboard_event = KeyboardEvent(**msg_data)
            tokens.extend(self._encode_keyboard(keyboard_event))

        elif mcap_message.topic == "mouse" and mcap_message.message_type == "desktop/MouseEvent":
            mouse_event = MouseEvent(**msg_data)
            tokens.extend(self.mouse_processor.encode_mouse_event(mouse_event))

        elif mcap_message.topic == "screen" and mcap_message.message_type == "desktop/ScreenCaptured":
            screen_event = ScreenCaptured(**msg_data)
            tokens.append("<SCREEN>")
            images.append(screen_event)

        else:
            tokens.append("<UNK>")

        # Wrap with EVENT_START and EVENT_END tokens for consistent parsing
        tokens_str = "".join(tokens)
        return f"<EVENT_START>{tokens_str}<EVENT_END>", images

    def decode(
        self,
        encoded_data: str,
        images: Optional[List[ScreenCaptured]] = None,
        screen_size: Optional[Tuple[int, int]] = None,
    ) -> McapMessage:
        """
        Decode hierarchical tokens back to original raw event format.

        Args:
            encoded_data: Hierarchical token sequence as concatenated string
            images: Optional list of image data for screen events
            screen_size: Optional screen size for mouse coordinate decoding

        Returns:
            Dict: Reconstructed raw event in original format

        Raises:
            ValueError: If token sequence format is invalid
        """
        # Check for EVENT_START and EVENT_END tokens
        if not encoded_data.startswith("<EVENT_START>") or not encoded_data.endswith("<EVENT_END>"):
            raise ValueError("Invalid encoded format: missing <EVENT_START> or <EVENT_END> tokens")

        # Extract the token sequence between EVENT_START and EVENT_END
        token_content = encoded_data[len("<EVENT_START>") : -len("<EVENT_END>")].strip()
        # Parse tokens without spaces using regex to find all <...> patterns
        tokens = re.findall(r"<[^>]*>", token_content) if token_content else []

        if not tokens or len(tokens) < 2:
            raise ValueError("Token sequence too short")

        # Decode timestamp (first 2 tokens)
        timestamp_ns = self._decode_timestamp(tokens[:2])

        # Determine event type and decode accordingly
        if len(tokens) < 3:
            # Only timestamp, create unknown event
            return McapMessage(
                topic="unknown",
                timestamp=timestamp_ns,
                message_type="unknown",
                message="{}".encode("utf-8"),
            )

        event_type_token = tokens[2]

        if event_type_token == "<KEYBOARD>":
            # Decode keyboard event
            if len(tokens) < 5:
                raise ValueError("Insufficient tokens for keyboard event")

            keyboard_event = self._decode_keyboard(tokens[2:5])
            msg_data = {"event_type": keyboard_event.event_type, "vk": keyboard_event.vk}

            return McapMessage(
                topic="keyboard",
                timestamp=timestamp_ns,
                message_type="desktop/KeyboardEvent",
                message=json.dumps(msg_data).encode("utf-8"),
            )

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

            return McapMessage(
                topic="mouse",
                timestamp=timestamp_ns,
                message_type="desktop/MouseEvent",
                message=json.dumps(msg_data).encode("utf-8"),
            )

        elif event_type_token == "<SCREEN>":
            # Decode screen event
            if not images:
                raise ValueError("Screen event requires image data but none provided")

            image_data = images[0]
            if isinstance(image_data, ScreenCaptured):
                # Convert ScreenCaptured back to JSON
                msg_dict = image_data.model_dump(exclude={"frame_arr"})
                msg = json.dumps(msg_dict)
            else:
                # Fallback for unexpected data types
                msg = json.dumps(image_data) if image_data else "{}"

            return McapMessage(
                topic="screen",
                timestamp=timestamp_ns,
                message_type="desktop/ScreenCaptured",
                message=msg.encode("utf-8") if isinstance(msg, str) else msg,
            )

        else:
            raise ValueError(f"Unknown event type token: {event_type_token}")

    def encode_batch(self, mcap_messages: List[McapMessage]) -> Tuple[List[str], List[List[ScreenCaptured]]]:
        """Encode a batch of McapMessage objects."""
        all_tokens = []
        all_images = []
        for event in mcap_messages:
            tokens, images = self.encode(event)
            all_tokens.append(tokens)
            all_images.append(images)
        return all_tokens, all_images

    def decode_batch(
        self,
        encoded_batch: List[str],
        all_images: Optional[List[List[ScreenCaptured]]] = None,
        screen_size: Optional[Tuple[int, int]] = None,
    ) -> List[McapMessage]:
        """Decode a batch of hierarchical token sequences."""
        if all_images is None:
            all_images = [[] for _ in encoded_batch]
        if len(encoded_batch) != len(all_images):
            raise ValueError("Length mismatch between tokens and images")
        events = []
        for encoded_data, images in zip(encoded_batch, all_images):
            event = self.decode(encoded_data, images, screen_size)
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
