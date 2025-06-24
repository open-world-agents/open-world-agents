"""
FlatEventEncoder for converting raw events to flat token format.

This module implements the FlatEventEncoder class based on the example.py pattern,
using flat tokens like <TIMESTAMP_123>, <KEYBOARD_65_press>, <MOUSE_move_0_15_32>.

This approach provides a direct token-based representation where each unique
combination gets its own token, suitable for models that work well with
large vocabularies and direct token prediction.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from mcap_owa.highlevel.reader import McapMessage
from owa.core.time import TimeUnits
from owa.msgs.desktop.keyboard import KeyboardEvent
from owa.msgs.desktop.mouse import MouseEvent
from owa.msgs.desktop.screen import ScreenCaptured

from .base_encoder import BaseEventEncoder


class FlatEventEncoderConfig:
    """Configuration for FlatEventEncoder."""

    def __init__(
        self,
        timestamp_min_ns: int = -2 * TimeUnits.SECOND,
        timestamp_max_ns: int = 2 * TimeUnits.SECOND,
        timestamp_interval_ns: int = 20 * TimeUnits.MSECOND,  # 50fps
        keyboard_vk_count: int = 256,
        mouse_move_bins: List[int] = None,
        screen_size: Tuple[int, int] = (1920, 1080),
        drop_file_path: bool = True,
    ):
        self.timestamp_min_ns = timestamp_min_ns
        self.timestamp_max_ns = timestamp_max_ns
        self.timestamp_interval_ns = timestamp_interval_ns
        self.keyboard_vk_count = keyboard_vk_count
        self.mouse_move_bins = mouse_move_bins or [16, 16, 16]  # 3-level residual quantization
        self.screen_size = screen_size
        self.drop_file_path = drop_file_path

        # Token formats (flat style)
        self.timestamp_token_format = "<TIMESTAMP_{idx}>"
        self.keyboard_token_format = "<KEYBOARD_{vk}_{pressed}>"
        self.mouse_move_token_format = "<MOUSE_move_{level}_{idx_x}_{idx_y}>"
        self.mouse_click_token_format = "<MOUSE_click_{button}_{pressed}>"
        self.mouse_scroll_token_format = "<MOUSE_scroll_{dx}_{dy}>"
        self.screen_token = "<SCREEN>"

    @property
    def timestamp_count(self) -> int:
        """Number of timestamp bins."""
        return ((self.timestamp_max_ns - self.timestamp_min_ns) // self.timestamp_interval_ns) + 1

    @property
    def timestamp_tokens(self) -> List[str]:
        """All unique timestamp tokens."""
        return [self.timestamp_token_format.format(idx=idx) for idx in range(self.timestamp_count)]

    @property
    def keyboard_tokens(self) -> List[str]:
        """All unique keyboard event tokens."""
        tokens = []
        for vk in range(self.keyboard_vk_count):
            tokens.append(self.keyboard_token_format.format(vk=vk, pressed="press"))
            tokens.append(self.keyboard_token_format.format(vk=vk, pressed="release"))
        return tokens

    @property
    def mouse_move_tokens(self) -> List[str]:
        """All unique mouse move tokens for all quantization levels."""
        tokens = []
        for level, bins in enumerate(self.mouse_move_bins):
            for idx_x in range(bins):
                for idx_y in range(bins):
                    tokens.append(self.mouse_move_token_format.format(level=level, idx_x=idx_x, idx_y=idx_y))
        return tokens

    @property
    def mouse_click_tokens(self) -> List[str]:
        """All unique mouse click tokens."""
        tokens = []
        buttons = ["left", "right", "middle", "unknown"]
        for button in buttons:
            for pressed in ["press", "release"]:
                tokens.append(self.mouse_click_token_format.format(button=button, pressed=pressed))
        return tokens

    @property
    def mouse_scroll_tokens(self) -> List[str]:
        """All mouse scroll tokens."""
        tokens = []
        for dx in [-3, -2, -1, 0, 1, 2, 3]:
            for dy in [-3, -2, -1, 0, 1, 2, 3]:
                tokens.append(self.mouse_scroll_token_format.format(dx=dx, dy=dy))
        return tokens

    @property
    def screen_tokens(self) -> List[str]:
        """All unique screen marker tokens."""
        return [self.screen_token]

    @property
    def all_tokens(self) -> List[str]:
        """All unique tokens in the vocabulary."""
        return (
            ["<EVENT_START>", "<EVENT_END>"]
            + self.timestamp_tokens
            + self.keyboard_tokens
            + self.mouse_move_tokens
            + self.mouse_click_tokens
            + self.mouse_scroll_tokens
            + self.screen_tokens
            + ["<UNKNOWN>"]
        )

    def get_vocab_size(self) -> int:
        """Get total vocabulary size."""
        return len(self.all_tokens)


class FlatMouseProcessor:
    """Processes mouse events with flat token encoding."""

    def __init__(self, config: FlatEventEncoderConfig):
        self.config = config

    def _limit(self, x: float, low: float = 0.0, high: float = 1.0) -> float:
        """Limit x to the range [low, high]."""
        return max(low, min(x, high))

    def encode_move(self, event: MouseEvent, screen_size: Optional[Tuple[int, int]] = None) -> List[str]:
        """
        Encode mouse movement with residual quantization.
        Returns: [<MOUSE_move_0_x_y>, <MOUSE_move_1_x_y>, <MOUSE_move_2_x_y>]
        """
        if screen_size is None:
            screen_size = self.config.screen_size

        x, y = event.x, event.y
        fx = self._limit(x / screen_size[0])
        fy = self._limit(y / screen_size[1])

        tokens = []

        # Jointly quantize the pair (x, y) repeatedly at each level
        vx, vy = fx, fy
        for level, nbins in enumerate(self.config.mouse_move_bins):
            # Using floor for better accuracy
            idx_x = int(vx * nbins)
            idx_y = int(vy * nbins)
            tokens.append(self.config.mouse_move_token_format.format(level=level, idx_x=idx_x, idx_y=idx_y))

            # Calculate residuals for next level
            vx = vx * nbins - idx_x
            vy = vy * nbins - idx_y

        return tokens

    def encode_click(self, event: MouseEvent) -> str:
        """Encode mouse click: <MOUSE_click_button_action>"""
        button = event.button or "unknown"
        action = "press" if bool(event.pressed) else "release"
        return self.config.mouse_click_token_format.format(button=button, pressed=action)

    def encode_scroll(self, event: MouseEvent) -> str:
        """Encode mouse scroll: <MOUSE_scroll_dx_dy>"""
        dx = event.dx if event.dx is not None else 0
        dy = event.dy if event.dy is not None else 0
        # Clamp to supported range
        dx = max(-3, min(3, dx))
        dy = max(-3, min(3, dy))
        return self.config.mouse_scroll_token_format.format(dx=dx, dy=dy)

    def encode_mouse_event(self, event: MouseEvent, screen_size: Optional[Tuple[int, int]] = None) -> List[str]:
        """Encode any mouse event."""
        # Always include position
        tokens = self.encode_move(event, screen_size)

        # Add specific action
        if event.event_type == "move":
            return tokens
        elif event.event_type == "click":
            return tokens + [self.encode_click(event)]
        elif event.event_type == "scroll":
            return tokens + [self.encode_scroll(event)]
        else:
            return tokens + ["<MOUSE_unknown>"]


class FlatEventEncoder(BaseEventEncoder):
    """
    Flat event encoder using traditional flat token format.

    This encoder converts raw events to flat token sequences where each unique
    combination gets its own token (e.g., <KEYBOARD_65_press>, <MOUSE_move_0_15_32>).

    Examples:
        >>> config = FlatEventEncoderConfig()
        >>> encoder = FlatEventEncoder(config)
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
        '<EVENT_START><TIMESTAMP_123><KEYBOARD_65_press><EVENT_END>'
    """

    def __init__(self, config: Optional[FlatEventEncoderConfig] = None):
        """Initialize the flat event encoder."""
        self.config = config or FlatEventEncoderConfig()
        self.mouse_processor = FlatMouseProcessor(self.config)

        # Create token to ID mapping
        all_tokens = self.config.all_tokens
        self.token_to_id = {token: i for i, token in enumerate(all_tokens)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def _encode_timestamp(self, timestamp_ns: int) -> str:
        """Encode timestamp to flat token: <TIMESTAMP_index>"""
        # Normalize timestamp to config range
        mod = timestamp_ns % (self.config.timestamp_max_ns - self.config.timestamp_min_ns)
        idx = mod // self.config.timestamp_interval_ns

        # Ensure index is within bounds
        max_idx = self.config.timestamp_count - 1
        idx = min(max_idx, max(0, idx))

        return self.config.timestamp_token_format.format(idx=idx)

    def _encode_keyboard(self, event: KeyboardEvent) -> str:
        """Encode keyboard event: <KEYBOARD_vk_action>"""
        return self.config.keyboard_token_format.format(vk=event.vk, pressed=event.event_type)

    def encode(self, mcap_message: McapMessage) -> Tuple[str, List[ScreenCaptured]]:
        """
        Encode a single McapMessage object to flat token format.

        Args:
            mcap_message: McapMessage instance

        Returns:
            Tuple containing:
                - str: Flat token sequence concatenated without spaces
                - List[ScreenCaptured]: Image data for screen events (empty for others)

        Raises:
            ValueError: If the mcap_message format is invalid
            json.JSONDecodeError: If message content cannot be parsed
        """
        mcap_message = mcap_message if isinstance(mcap_message, McapMessage) else McapMessage(**mcap_message)

        # Start with timestamp
        tokens = [self._encode_timestamp(mcap_message.timestamp)]
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
            tokens.append(self._encode_keyboard(keyboard_event))

        elif mcap_message.topic == "mouse" and mcap_message.message_type == "desktop/MouseEvent":
            mouse_event = MouseEvent(**msg_data)
            tokens.extend(self.mouse_processor.encode_mouse_event(mouse_event))

        elif mcap_message.topic == "screen" and mcap_message.message_type == "desktop/ScreenCaptured":
            screen_event = ScreenCaptured(**msg_data)
            tokens.append(self.config.screen_token)
            # Store image data
            images.append(screen_event)

        else:
            tokens.append("<UNKNOWN>")

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
        Decode flat tokens back to original raw event format.

        Args:
            encoded_data: Flat token sequence as concatenated string
            images: Optional list of image data for screen events
            screen_size: Optional screen size for mouse coordinate decoding

        Returns:
            McapMessage: Reconstructed message in McapMessage format

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

        if not tokens or len(tokens) < 1:
            raise ValueError("Token sequence too short")

        # Decode timestamp (first token)
        timestamp_token = tokens[0]
        timestamp_match = re.match(r"<TIMESTAMP_(\d+)>", timestamp_token)
        if not timestamp_match:
            raise ValueError(f"Invalid timestamp token: {timestamp_token}")

        idx = int(timestamp_match.group(1))
        timestamp_ns = self.config.timestamp_min_ns + idx * self.config.timestamp_interval_ns

        # Determine event type and decode accordingly
        if len(tokens) < 2:
            # Only timestamp, create unknown event
            return McapMessage(
                topic="unknown",
                timestamp=timestamp_ns,
                message_type="unknown",
                message="{}".encode("utf-8"),
            )

        event_token = tokens[1]

        if event_token.startswith("<KEYBOARD_"):
            # Decode keyboard event
            keyboard_match = re.match(r"<KEYBOARD_(\d+)_(\w+)>", event_token)
            if not keyboard_match:
                raise ValueError(f"Invalid keyboard token: {event_token}")

            vk = int(keyboard_match.group(1))
            event_type = keyboard_match.group(2)

            msg_data = {"event_type": event_type, "vk": vk}

            return McapMessage(
                topic="keyboard",
                timestamp=timestamp_ns,
                message_type="desktop/KeyboardEvent",
                message=json.dumps(msg_data).encode("utf-8"),
            )

        elif event_token.startswith("<MOUSE_move_"):
            # Decode mouse event (complex due to multi-level encoding)
            return self._decode_mouse_event(tokens[1:], timestamp_ns, screen_size)

        elif event_token == "<SCREEN>":
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
            # Unknown event type
            return McapMessage(
                topic="unknown",
                timestamp=timestamp_ns,
                message_type="unknown",
                message="{}".encode("utf-8"),
            )

    def _decode_mouse_event(
        self, mouse_tokens: List[str], timestamp_ns: int, screen_size: Optional[Tuple[int, int]] = None
    ) -> McapMessage:
        """Decode mouse tokens back to MouseEvent."""
        if screen_size is None:
            screen_size = self.config.screen_size

        # Extract move tokens (first N tokens for N levels)
        move_tokens = []
        other_tokens = []

        for token in mouse_tokens:
            if token.startswith("<MOUSE_move_"):
                move_tokens.append(token)
            else:
                other_tokens.append(token)

        if len(move_tokens) != len(self.config.mouse_move_bins):
            raise ValueError(f"Expected {len(self.config.mouse_move_bins)} move tokens, got {len(move_tokens)}")

        # Decode coordinates using residual quantization
        indices = []
        for token in move_tokens:
            move_match = re.match(r"<MOUSE_move_(\d+)_(\d+)_(\d+)>", token)
            if not move_match:
                raise ValueError(f"Invalid mouse move token: {token}")

            level = int(move_match.group(1))
            idx_x = int(move_match.group(2))
            idx_y = int(move_match.group(3))
            indices.append((level, idx_x, idx_y))

        # Sort by level to ensure correct order
        indices.sort(key=lambda x: x[0])

        # Reconstruct coordinates
        fx = fy = 0.0
        for i in reversed(range(len(indices))):
            level, idx_x, idx_y = indices[i]
            nbins = self.config.mouse_move_bins[i]

            fx = (fx + idx_x) / nbins
            fy = (fy + idx_y) / nbins

        # Convert to pixel coordinates
        pix_x = int(round(fx * (screen_size[0] - 1)))
        pix_y = int(round(fy * (screen_size[1] - 1)))

        # Determine event type and decode additional parameters
        if not other_tokens:
            # Pure movement
            msg_data = {
                "event_type": "move",
                "x": pix_x,
                "y": pix_y,
            }
        elif other_tokens[0].startswith("<MOUSE_click_"):
            # Click event
            click_match = re.match(r"<MOUSE_click_(\w+)_(\w+)>", other_tokens[0])
            if not click_match:
                raise ValueError(f"Invalid mouse click token: {other_tokens[0]}")

            button = click_match.group(1)
            pressed = click_match.group(2) == "press"

            msg_data = {
                "event_type": "click",
                "x": pix_x,
                "y": pix_y,
                "button": button,
                "pressed": pressed,
            }
        elif other_tokens[0].startswith("<MOUSE_scroll_"):
            # Scroll event
            scroll_match = re.match(r"<MOUSE_scroll_(-?\d+)_(-?\d+)>", other_tokens[0])
            if not scroll_match:
                raise ValueError(f"Invalid mouse scroll token: {other_tokens[0]}")

            dx = int(scroll_match.group(1))
            dy = int(scroll_match.group(2))

            msg_data = {
                "event_type": "scroll",
                "x": pix_x,
                "y": pix_y,
                "dx": dx,
                "dy": dy,
            }
        else:
            # Unknown mouse event
            msg_data = {
                "event_type": "unknown",
                "x": pix_x,
                "y": pix_y,
            }

        return McapMessage(
            topic="mouse",
            timestamp=timestamp_ns,
            message_type="desktop/MouseEvent",
            message=json.dumps(msg_data).encode("utf-8"),
        )

    def encode_batch(self, mcap_messages: List[McapMessage]) -> Tuple[List[str], List[List[ScreenCaptured]]]:
        """
        Encode a batch of McapMessage-like objects.

        Args:
            mcap_messages: List of McapMessage instances or dictionaries

        Returns:
            Tuple containing:
                - List[str]: Flat token sequences for each event as strings
                - List[List[ScreenCaptured]]: Image data for each event
        """
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
        """
        Decode a batch of flat token sequences.

        Args:
            encoded_batch: List of flat token sequences as strings
            all_images: Optional list of image data lists for each event
            screen_size: Optional screen size for mouse coordinate decoding

        Returns:
            List[McapMessage]: Reconstructed messages
        """
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
        return self.config.get_vocab_size()

    def get_token_ids(self, tokens: List[str]) -> List[int]:
        """Convert token strings to token IDs."""
        return [self.token_to_id.get(token, self.token_to_id.get("<UNKNOWN>", 0)) for token in tokens]

    def get_tokens_from_ids(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs to token strings."""
        return [self.id_to_token.get(token_id, "<UNKNOWN>") for token_id in token_ids]

    def get_encoder_info(self) -> Dict[str, Any]:
        """Get information about this encoder."""
        return {
            "encoder_type": "FlatEventEncoder",
            "format": "flat_tokens",
            "vocab_size": self.get_vocab_size(),
            "drop_file_path": self.config.drop_file_path,
            "mouse_move_bins": self.config.mouse_move_bins,
            "screen_size": self.config.screen_size,
        }
