import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from mcap_owa.highlevel.reader import McapMessage
from owa.core.time import TimeUnits
from owa.msgs.desktop.keyboard import KeyboardEvent
from owa.msgs.desktop.mouse import MouseEvent
from owa.msgs.desktop.screen import ScreenCaptured

from .base_encoder import BaseEventEncoder, BaseEventEncoderConfig


@dataclass
class HierarchicalEventEncoderConfig(BaseEventEncoderConfig):
    """Configuration for HierarchicalEventEncoder."""

    # -2 to +2 seconds
    timestamp_range_ns: int = 4 * TimeUnits.SECOND
    # 4 seconds in 10ms intervals
    timestamp_bases: List[int] = field(default_factory=lambda: [4, 10, 10])
    # coordinate quantization bases (like hex: base-16)
    mouse_coord_bases: List[int] = field(default_factory=lambda: [16, 16, 16])
    screen_size: Tuple[int, int] = (1920, 1080)


def quantize_to_digits(value: float, bases: List[int]) -> List[int]:
    """
    Quantize a normalized value (0.0-1.0) to multi-level digits.

    Args:
        value: Normalized value between 0.0 and 1.0
        bases: List of bases for each quantization level (e.g., [16, 16, 16] for 3-level hex)

    Returns:
        List of digits for each level

    Example:
        >>> quantize_to_digits(0.6875, [16, 16, 16])
        [11, 0, 0]  # 0xB00 in hex = 0.6875
    """
    digits = []
    remaining = max(0.0, min(1.0, value))  # Clamp to [0, 1]

    for base in bases:
        digit = int(remaining * base)
        digits.append(digit)
        remaining = remaining * base - digit

    return digits


def digits_to_value(digits: List[int], bases: List[int]) -> float:
    """
    Reconstruct normalized value from multi-level digits.

    Args:
        digits: List of digits for each level
        bases: List of bases for each quantization level

    Returns:
        Reconstructed normalized value between 0.0 and 1.0

    Example:
        >>> digits_to_value([11, 0, 0], [16, 16, 16])
        0.6875  # 0xB00 in hex
    """
    if len(digits) != len(bases):
        raise ValueError(f"Digits length {len(digits)} must match bases length {len(bases)}")

    value = 0.0
    for i in reversed(range(len(digits))):
        digit = digits[i]
        base = bases[i]
        value = (value + digit) / base

    return value


def _generate_vocab(
    image_token: str = "<image>",
    image_token_prefix: str = "<fake_token_around_image><global-img>",
    image_token_suffix: str = "<fake_token_around_image>",
) -> Set[str]:
    """Generate the hierarchical token vocabulary."""
    vocab = [
        "<EVENT_START>",
        "<EVENT_END>",
        "<TIMESTAMP>",
        "<KEYBOARD>",
        "<MOUSE>",
        image_token,
        image_token_prefix,
        image_token_suffix,
    ]

    # Numbers 0-255 for various parameters
    vocab.extend(f"<{i}>" for i in range(256))

    # Action types and mouse buttons
    vocab.extend(["<press>", "<release>", "<move>", "<click>", "<scroll>"])
    vocab.extend(["<left>", "<right>", "<middle>", "<unknown>"])

    # Negative numbers for scroll deltas
    vocab.extend(f"<{i}>" for i in range(-10, 11))

    return set(vocab)


class HierarchicalEventEncoder(BaseEventEncoder):
    """
    Hierarchical event encoder for VLA training with compositional token structure.

    This encoder converts raw events to hierarchical token sequences that are more
    efficient and learnable than traditional flat token approaches.
    """

    def __init__(self, config: HierarchicalEventEncoderConfig = HierarchicalEventEncoderConfig(), **kwargs):
        """Initialize the hierarchical event encoder."""
        self.config = HierarchicalEventEncoderConfig(**(config.__dict__ | kwargs))

    def _encode_timestamp(self, timestamp_ns: int) -> List[str]:
        """Encode timestamp with multi-level quantization: [<TIMESTAMP>, <digit1>, <digit2>, ...]"""
        # Normalize timestamp to [0, 1] range within the configured range
        mod_timestamp = timestamp_ns % self.config.timestamp_range_ns
        norm_timestamp = mod_timestamp / self.config.timestamp_range_ns

        # Quantize to digits
        digits = quantize_to_digits(norm_timestamp, self.config.timestamp_bases)

        # Create tokens
        tokens = ["<TIMESTAMP>"] + [f"<{digit}>" for digit in digits]
        return tokens

    def _encode_keyboard(self, event: KeyboardEvent) -> List[str]:
        """Encode keyboard event: [<KEYBOARD>, <vk>, <action>]"""
        return ["<KEYBOARD>", f"<{event.vk}>", f"<{event.event_type}>"]

    def _encode_mouse(self, event: MouseEvent) -> List[str]:
        """Encode mouse event with multi-level coordinate quantization."""
        x, y = event.x, event.y
        norm_x = x / self.config.screen_size[0]
        norm_y = y / self.config.screen_size[1]

        tokens = ["<MOUSE>", "<move>"]

        # Quantize coordinates to digits
        digits_x = quantize_to_digits(norm_x, self.config.mouse_coord_bases)
        digits_y = quantize_to_digits(norm_y, self.config.mouse_coord_bases)

        # Interleave x,y digit pairs
        for digit_x, digit_y in zip(digits_x, digits_y):
            tokens.extend([f"<{digit_x}>", f"<{digit_y}>"])

        # Add action-specific tokens
        if event.event_type == "click":
            button = event.button or "unknown"
            action = "press" if bool(event.pressed) else "release"
            tokens.extend([f"<{button}>", f"<{action}>"])
        elif event.event_type == "scroll":
            dx = event.dx if event.dx is not None else 0
            dy = event.dy if event.dy is not None else 0
            tokens.extend([f"<{dx}>", f"<{dy}>"])

        return tokens

    def _decode_mouse_coords(
        self, tokens: List[str], screen_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        """Decode quantized mouse coordinates."""
        if screen_size is None:
            screen_size = self.config.screen_size

        coord_tokens = tokens[2:]  # Skip <MOUSE> and <move>
        if len(coord_tokens) != len(self.config.mouse_coord_bases) * 2:
            raise ValueError(f"Expected {len(self.config.mouse_coord_bases) * 2} coordinate tokens")

        # Parse digit pairs from tokens
        digits_x, digits_y = [], []
        for i in range(0, len(coord_tokens), 2):
            x_token = coord_tokens[i]
            y_token = coord_tokens[i + 1]

            x_match = re.match(r"<(\d+)>", x_token)
            y_match = re.match(r"<(\d+)>", y_token)
            if not x_match or not y_match:
                raise ValueError(f"Invalid coordinate tokens: {x_token}, {y_token}")

            digits_x.append(int(x_match.group(1)))
            digits_y.append(int(y_match.group(1)))

        # Reconstruct normalized coordinates from digits
        norm_x = digits_to_value(digits_x, self.config.mouse_coord_bases)
        norm_y = digits_to_value(digits_y, self.config.mouse_coord_bases)

        return int(round(norm_x * (screen_size[0] - 1))), int(round(norm_y * (screen_size[1] - 1)))

    def encode(self, mcap_message: McapMessage) -> Tuple[str, List[ScreenCaptured]]:
        """Encode a single McapMessage object to hierarchical token format."""
        mcap_message = mcap_message if isinstance(mcap_message, McapMessage) else McapMessage(**mcap_message)

        tokens = self._encode_timestamp(mcap_message.timestamp)
        images = []

        # Parse message content
        try:
            msg_data = json.loads(
                mcap_message.message.decode("utf-8")
                if isinstance(mcap_message.message, bytes)
                else mcap_message.message
            )
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Failed to parse message content: {e}")

        # Encode based on event type
        if mcap_message.topic == "keyboard":
            keyboard_event = KeyboardEvent(**msg_data)
            tokens.extend(self._encode_keyboard(keyboard_event))
        elif mcap_message.topic == "mouse":
            mouse_event = MouseEvent(**msg_data)
            tokens.extend(self._encode_mouse(mouse_event))
        elif mcap_message.topic == "screen":
            screen_event = ScreenCaptured(**msg_data)
            tokens.extend([self.config.image_token_prefix, self.config.image_token, self.config.image_token_suffix])
            images.append(screen_event)
        else:
            raise ValueError(f"Unsupported event type: {mcap_message.topic}")

        return f"<EVENT_START>{''.join(tokens)}<EVENT_END>", images

    def _decode_timestamp(self, tokens: List[str]) -> int:
        """Decode timestamp tokens back to nanoseconds."""
        if len(tokens) != len(self.config.timestamp_bases) + 1 or tokens[0] != "<TIMESTAMP>":
            raise ValueError(f"Invalid timestamp tokens: {tokens}")

        # Parse digits from tokens
        digits = []
        for i in range(1, len(tokens)):
            digit_match = re.match(r"<(\d+)>", tokens[i])
            if not digit_match:
                raise ValueError(f"Invalid timestamp digit token: {tokens[i]}")
            digits.append(int(digit_match.group(1)))

        # Reconstruct normalized timestamp
        norm_timestamp = digits_to_value(digits, self.config.timestamp_bases)

        # Convert back to nanoseconds
        return int(norm_timestamp * self.config.timestamp_range_ns)

    def _decode_keyboard(self, tokens: List[str]) -> KeyboardEvent:
        """Decode keyboard tokens back to KeyboardEvent."""
        if len(tokens) != 3 or tokens[0] != "<KEYBOARD>":
            raise ValueError(f"Invalid keyboard tokens: {tokens}")
        vk_match = re.match(r"<(\d+)>", tokens[1])
        action_match = re.match(r"<(\w+)>", tokens[2])
        if not vk_match or not action_match:
            raise ValueError(f"Invalid keyboard tokens: {tokens}")
        return KeyboardEvent(event_type=action_match.group(1), vk=int(vk_match.group(1)))

    def _decode_mouse(self, tokens: List[str], screen_size: Optional[Tuple[int, int]] = None) -> MouseEvent:
        """Decode mouse tokens back to MouseEvent."""
        if len(tokens) < 2 or tokens[0] != "<MOUSE>":
            raise ValueError(f"Invalid mouse tokens: {tokens}")

        # Decode coordinates
        move_end_idx = 2 + len(self.config.mouse_coord_bases) * 2
        x, y = self._decode_mouse_coords(tokens[:move_end_idx], screen_size)

        # Determine event type and additional parameters
        if len(tokens) == move_end_idx:
            return MouseEvent(event_type="move", x=x, y=y)
        elif len(tokens) >= move_end_idx + 2:
            if "<left>" in tokens or "<right>" in tokens or "<middle>" in tokens:
                # Click event
                button_token = tokens[move_end_idx]
                action_token = tokens[move_end_idx + 1]
                button_match = re.match(r"<(\w+)>", button_token)
                action_match = re.match(r"<(\w+)>", action_token)
                if not button_match or not action_match:
                    raise ValueError(f"Invalid click tokens: {button_token}, {action_token}")
                button = button_match.group(1)
                pressed = action_match.group(1) == "press"
                return MouseEvent(event_type="click", x=x, y=y, button=button, pressed=pressed)
            else:
                # Scroll event
                dx_token = tokens[move_end_idx]
                dy_token = tokens[move_end_idx + 1]
                dx_match = re.match(r"<(-?\d+)>", dx_token)
                dy_match = re.match(r"<(-?\d+)>", dy_token)
                if not dx_match or not dy_match:
                    raise ValueError(f"Invalid scroll tokens: {dx_token}, {dy_token}")
                dx = int(dx_match.group(1))
                dy = int(dy_match.group(1))
                return MouseEvent(event_type="scroll", x=x, y=y, dx=dx, dy=dy)

        raise ValueError(f"Invalid mouse token sequence: {tokens}")

    def decode(
        self,
        encoded_data: str,
        images: Optional[List[ScreenCaptured]] = None,
        screen_size: Optional[Tuple[int, int]] = None,
    ) -> McapMessage:
        """Decode hierarchical tokens back to original raw event format."""
        if not encoded_data.startswith("<EVENT_START>") or not encoded_data.endswith("<EVENT_END>"):
            raise ValueError("Invalid encoded format: missing <EVENT_START> or <EVENT_END> tokens")

        token_content = encoded_data[len("<EVENT_START>") : -len("<EVENT_END>")].strip()
        tokens = re.findall(r"<[^>]*>", token_content) if token_content else []

        timestamp_token_count = len(self.config.timestamp_bases) + 1
        if len(tokens) < timestamp_token_count + 1:
            raise ValueError("Token sequence too short")

        timestamp_ns = self._decode_timestamp(tokens[:timestamp_token_count])
        event_type_token = tokens[timestamp_token_count]

        if event_type_token == "<KEYBOARD>":
            keyboard_event = self._decode_keyboard(tokens[timestamp_token_count : timestamp_token_count + 3])
            msg_data = {"event_type": keyboard_event.event_type, "vk": keyboard_event.vk}
            return McapMessage(
                topic="keyboard",
                timestamp=timestamp_ns,
                message_type="desktop/KeyboardEvent",
                message=json.dumps(msg_data).encode("utf-8"),
            )
        elif event_type_token == "<MOUSE>":
            mouse_event = self._decode_mouse(tokens[timestamp_token_count:], screen_size)
            msg_data = {"event_type": mouse_event.event_type, "x": mouse_event.x, "y": mouse_event.y}
            if mouse_event.button:
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
        elif event_type_token == self.config.image_token_prefix:
            # Check if we have enough tokens for the full image token sequence
            if (
                len(tokens) < timestamp_token_count + 3
                or tokens[timestamp_token_count + 1] != self.config.image_token
                or tokens[timestamp_token_count + 2] != self.config.image_token_suffix
            ):
                raise ValueError(
                    f"Invalid image token sequence: expected prefix, token, suffix but got {tokens[timestamp_token_count : timestamp_token_count + 3]}"
                )

            if not images:
                raise ValueError("Screen event requires image data but none provided")
            image_data = images[0]
            msg = image_data.model_dump_json(exclude={"frame_arr"})
            return McapMessage(
                topic="screen",
                timestamp=timestamp_ns,
                message_type="desktop/ScreenCaptured",
                message=msg.encode("utf-8"),
            )
        else:
            raise ValueError(f"Unknown event type token: {event_type_token}")

    def encode_batch(self, mcap_messages: List[McapMessage]) -> Tuple[List[str], List[List[ScreenCaptured]]]:
        """Encode a batch of McapMessage objects."""
        all_tokens, all_images = [], []
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
        return [
            self.decode(encoded_data, images, screen_size) for encoded_data, images in zip(encoded_batch, all_images)
        ]

    def get_vocab(self) -> Set[str]:
        """Get all tokens in the vocabulary."""
        return _generate_vocab(self.config.image_token, self.config.image_token_prefix, self.config.image_token_suffix)
