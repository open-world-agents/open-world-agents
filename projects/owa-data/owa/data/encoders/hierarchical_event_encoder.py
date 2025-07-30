import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from mcap_owa.highlevel.reader import McapMessage
from owa.core.time import TimeUnits
from owa.msgs.desktop.keyboard import KeyboardEvent
from owa.msgs.desktop.mouse import RawMouseEvent
from owa.msgs.desktop.screen import ScreenCaptured

from .base_encoder import BaseEventEncoder, BaseEventEncoderConfig


@dataclass
class HierarchicalEventEncoderConfig(BaseEventEncoderConfig):
    """Configuration for HierarchicalEventEncoder."""

    # -2 to +2 seconds
    timestamp_range_ns: int = 4 * TimeUnits.SECOND
    # 4 seconds in 10ms intervals
    timestamp_bases: List[int] = field(default_factory=lambda: [4, 10, 10])
    # raw mouse delta quantization bases
    mouse_delta_bases: List[int] = field(default_factory=lambda: [10, 10, 10])
    # maximum dx value for normalization
    max_mouse_delta_x: int = 1000
    # maximum dy value for normalization
    max_mouse_delta_y: int = 1000


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
    """Hierarchical event encoder with simple token structure."""

    def __init__(self, config: Optional[HierarchicalEventEncoderConfig] = None, **kwargs):
        if config is None:
            config = HierarchicalEventEncoderConfig()
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

    def _encode_mouse(self, event: RawMouseEvent) -> List[str]:
        """Encode raw mouse event with multi-level delta quantization."""
        # Normalize dx, dy to [0, 1] range using separate max_mouse_delta values
        # Clamp values to [-max_mouse_delta_x, max_mouse_delta_x] and [-max_mouse_delta_y, max_mouse_delta_y] then normalize to [0, 1]
        dx_clamped = max(-self.config.max_mouse_delta_x, min(self.config.max_mouse_delta_x, event.dx))
        dy_clamped = max(-self.config.max_mouse_delta_y, min(self.config.max_mouse_delta_y, event.dy))

        norm_dx = (dx_clamped + self.config.max_mouse_delta_x) / (2 * self.config.max_mouse_delta_x)
        norm_dy = (dy_clamped + self.config.max_mouse_delta_y) / (2 * self.config.max_mouse_delta_y)

        tokens = ["<MOUSE>"]

        # Quantize deltas to digits
        digits_dx = quantize_to_digits(norm_dx, self.config.mouse_delta_bases)
        digits_dy = quantize_to_digits(norm_dy, self.config.mouse_delta_bases)

        # Interleave dx,dy digit pairs
        for digit_dx, digit_dy in zip(digits_dx, digits_dy):
            tokens.extend([f"<{digit_dx}>", f"<{digit_dy}>"])

        # Add button flags as a single token
        tokens.append(f"<{int(event.button_flags)}>")

        # Add button data if non-zero (for wheel events)
        if event.button_data != 0:
            # NOTE: button_data is USHORT and is multiple if 120=WHEEL_DELTA. See: https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-rawmouse
            button_data = event.button_data
            if button_data >= 32768:
                button_data -= 65536
            button_data //= 120
            tokens.append(f"<{button_data}>")

        return tokens

    def _decode_mouse_deltas(self, tokens: List[str]) -> Tuple[int, int]:
        """Decode quantized mouse deltas."""
        delta_tokens = tokens[1:]  # Skip <MOUSE>

        # Extract delta digit pairs (before button flags)
        expected_delta_tokens = len(self.config.mouse_delta_bases) * 2
        if len(delta_tokens) < expected_delta_tokens:
            raise ValueError(f"Expected at least {expected_delta_tokens} delta tokens")

        # Parse digit pairs from tokens
        digits_dx, digits_dy = [], []
        for i in range(0, expected_delta_tokens, 2):
            dx_token = delta_tokens[i]
            dy_token = delta_tokens[i + 1]

            dx_match = re.match(r"<(\d+)>", dx_token)
            dy_match = re.match(r"<(\d+)>", dy_token)
            if not dx_match or not dy_match:
                raise ValueError(f"Invalid delta tokens: {dx_token}, {dy_token}")

            digits_dx.append(int(dx_match.group(1)))
            digits_dy.append(int(dy_match.group(1)))

        # Reconstruct normalized deltas from digits
        norm_dx = digits_to_value(digits_dx, self.config.mouse_delta_bases)
        norm_dy = digits_to_value(digits_dy, self.config.mouse_delta_bases)

        # Convert back to actual delta values using separate max values
        dx = int(round(norm_dx * (2 * self.config.max_mouse_delta_x) - self.config.max_mouse_delta_x))
        dy = int(round(norm_dy * (2 * self.config.max_mouse_delta_y) - self.config.max_mouse_delta_y))

        return dx, dy

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
        elif mcap_message.topic == "mouse" or mcap_message.topic == "mouse/raw":
            raw_mouse_event = RawMouseEvent(**msg_data)
            tokens.extend(self._encode_mouse(raw_mouse_event))
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

    def _decode_mouse(self, tokens: List[str]) -> RawMouseEvent:
        """Decode mouse tokens back to RawMouseEvent."""
        if len(tokens) < 2 or tokens[0] != "<MOUSE>":
            raise ValueError(f"Invalid mouse tokens: {tokens}")

        # Decode deltas
        dx, dy = self._decode_mouse_deltas(tokens)

        # Extract button flags
        delta_token_count = len(self.config.mouse_delta_bases) * 2
        button_flags_idx = 1 + delta_token_count

        if len(tokens) <= button_flags_idx:
            raise ValueError("Missing button flags token")

        button_flags_token = tokens[button_flags_idx]
        button_flags_match = re.match(r"<(\d+)>", button_flags_token)
        if not button_flags_match:
            raise ValueError(f"Invalid button flags token: {button_flags_token}")
        button_flags = int(button_flags_match.group(1))

        # Extract button data if present
        button_data = 0
        if len(tokens) > button_flags_idx + 1:
            button_data_token = tokens[button_flags_idx + 1]
            button_data_match = re.match(r"<(-?\d+)>", button_data_token)
            if button_data_match:
                button_data = int(button_data_match.group(1))

        return RawMouseEvent(
            dx=dx, dy=dy, button_flags=RawMouseEvent.ButtonFlags(button_flags), button_data=button_data
        )

    def decode(
        self,
        encoded_data: str,
        images: Optional[List[ScreenCaptured]] = None,
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
            raw_mouse_event = self._decode_mouse(tokens[timestamp_token_count:])
            msg_data = {
                "dx": raw_mouse_event.dx,
                "dy": raw_mouse_event.dy,
                "button_flags": int(raw_mouse_event.button_flags),
                "button_data": raw_mouse_event.button_data,
            }
            if raw_mouse_event.device_handle is not None:
                msg_data["device_handle"] = raw_mouse_event.device_handle
            if raw_mouse_event.timestamp is not None:
                msg_data["timestamp"] = raw_mouse_event.timestamp
            return McapMessage(
                topic="mouse/raw",
                timestamp=timestamp_ns,
                message_type="desktop/RawMouseEvent",
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

    def get_vocab(self) -> Set[str]:
        """Get all tokens in the vocabulary."""
        return _generate_vocab(self.config.image_token, self.config.image_token_prefix, self.config.image_token_suffix)
