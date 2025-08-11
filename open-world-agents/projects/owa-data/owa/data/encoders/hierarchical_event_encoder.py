import re
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import orjson

from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.core.time import TimeUnits
from owa.msgs.desktop.keyboard import KeyboardEvent
from owa.msgs.desktop.mouse import RawMouseEvent
from owa.msgs.desktop.screen import ScreenCaptured

from .base_encoder import BaseEventEncoder, BaseEventEncoderConfig


@dataclass
class HierarchicalEventEncoderConfig(BaseEventEncoderConfig):
    """Configuration for HierarchicalEventEncoder."""

    # Minimal timestamp unit (default: 10ms)
    timestamp_unit_ns: int = 10 * TimeUnits.MSECOND
    # 16 seconds in 10ms intervals
    timestamp_bases: List[int] = field(default_factory=lambda: [16, 10, 10])
    # Mouse delta encoding bases
    mouse_delta_bases: List[int] = field(default_factory=lambda: [20, 10, 10])
    # Mouse scroll encoding bases
    mouse_scroll_bases: List[int] = field(default_factory=lambda: [10])

    def _signed_range(self, bases: List[int]) -> Tuple[int, int]:
        """Calculate signed range from bases."""
        total_range = 1
        for base in bases:
            total_range *= base
        min_val, max_val = -total_range, total_range - 1
        return min_val, max_val

    @property
    def mouse_delta_range(self) -> Tuple[int, int]:
        """Calculate valid mouse delta range from bases."""
        return self._signed_range(self.mouse_delta_bases)

    @property
    def mouse_scroll_range(self) -> Tuple[int, int]:
        """Calculate valid mouse scroll range from bases."""
        return self._signed_range(self.mouse_scroll_bases)


def quantize_to_digits(value: int, bases: List[int]) -> List[int]:
    """
    Quantize an integer to multi-level digits using modulo operations.

    Accepts any integer value. Negative values and values exceeding the base range
    are handled naturally through modulo arithmetic.

    Args:
        value: Integer to quantize
        bases: List of bases for each quantization level
               For signed representation, add [2] to front of bases

    Returns:
        List of digits (len(bases) total)

    Examples:
        >>> quantize_to_digits(64, [10, 10, 10])
        [0, 6, 4]
        >>> quantize_to_digits(1234, [10, 10, 10])
        [2, 3, 4]  # Values exceeding range wrap via modulo
        >>> quantize_to_digits(-3, [2, 10, 10, 10])
        [1, 9, 9, 7]  # Negative with signed: add [2] at front for sign bit
    """
    # Convert to digits
    digits = []
    remaining = value
    for base in reversed(bases):
        digit = remaining % base
        digits.insert(0, digit)
        remaining //= base

    return digits


def digits_to_value(digits: List[int], bases: List[int], *, signed: bool | None = None) -> int:
    """
    Reconstruct integer from digits.

    Args:
        digits: List of digits (len(bases) total)
        bases: List of bases for each quantization level
        signed: Whether the value is signed
                If None, infer from bases (signed if bases[0] == 2)

    Returns:
        Reconstructed integer

    Examples:
        # Unsigned representation
        >>> digits_to_value([0, 6, 4], [10, 10, 10])
        64  # <0><6><4> -> 64

        # Signed representation
        >>> digits_to_value([0, 0, 6, 4], [2, 10, 10, 10])
        64  # <0><0><6><4> -> 64 (positive)
        >>> digits_to_value([1, 9, 9, 7], [2, 10, 10, 10])
        -3  # <1><9><9><7> -> -3 (negative, 1997-2000=-3)
    """
    if len(digits) != len(bases):
        raise ValueError(f"Digits length {len(digits)} must match bases length {len(bases)}")

    if signed is None:
        signed = bases[0] == 2

    # Reconstruct the encoded value from digits
    encoded_value = 0
    for digit, base in zip(digits, bases):
        encoded_value = encoded_value * base + digit

    if signed:
        # Calculate total range for signed conversion
        total_range = 1
        for base in bases:
            total_range *= base
        # For signed representation, if encoded_value >= total_range//2, it's negative
        if encoded_value >= total_range // 2:
            return encoded_value - total_range
        else:
            return encoded_value
    else:
        return encoded_value


def _generate_vocab() -> Set[str]:
    """Generate the hierarchical token vocabulary.

    Note: fake_image_placeholder is NOT included in vocab as it's not a real token,
    just an internal placeholder used during encoding.
    """
    vocab = [
        "<EVENT_START>",
        "<EVENT_END>",
        "<KEYBOARD>",
        "<press>",
        "<release>",
        "<MOUSE>",
        # fake_image_placeholder deliberately excluded - it's not a real token
    ]

    # Numbers 0-255 for various parameters. TODO: verify this is enough
    vocab.extend(f"<{i}>" for i in range(256))
    return set(vocab)


class HierarchicalEventEncoder(BaseEventEncoder):
    """Hierarchical event encoder with simple token structure."""

    def __init__(self, config: Optional[HierarchicalEventEncoderConfig] = None, **kwargs):
        if config is None:
            config = HierarchicalEventEncoderConfig()
        self.config = HierarchicalEventEncoderConfig(**(config.__dict__ | kwargs))

    def _encode_timestamp(self, timestamp_ns: int) -> List[str]:
        """Encode timestamp with multi-level quantization: [<digit1>, <digit2>, ...]"""
        # Convert to timestamp units (e.g., 10ms units)
        timestamp_units = timestamp_ns // self.config.timestamp_unit_ns

        # Quantize to digits using integer approach
        digits = quantize_to_digits(timestamp_units, self.config.timestamp_bases)

        # Create tokens
        tokens = [f"<{digit}>" for digit in digits]
        return tokens

    def _encode_keyboard(self, event: KeyboardEvent) -> List[str]:
        """Encode keyboard event: [<KEYBOARD>, <vk>, <action>]"""
        return ["<KEYBOARD>", f"<{event.vk}>", f"<{event.event_type}>"]

    def _encode_mouse(self, event: RawMouseEvent) -> List[str]:
        """
        Encode raw mouse event as: <MOUSE><dx0><dy0><dx1><dy1>...<flag0><flag1><flag2><optional_button_data>

        Each flag is encoded as a hex digit (0-15).
        """
        # Warn if mouse delta values are outside acceptable range
        min_delta, max_delta = self.config.mouse_delta_range
        if not (min_delta <= event.dx <= max_delta) or not (min_delta <= event.dy <= max_delta):
            warnings.warn(
                f"Mouse delta value ({event.dx},{event.dy}) is outside valid range ({min_delta}, {max_delta}). Clamping."
            )

        tokens = ["<MOUSE>"]

        # Use signed bases (add [2] to front for sign bit)
        signed_bases = [2] + self.config.mouse_delta_bases
        digits_dx = quantize_to_digits(event.dx, signed_bases)
        digits_dy = quantize_to_digits(event.dy, signed_bases)

        # Interleave dx,dy digit pairs
        for digit_dx, digit_dy in zip(digits_dx, digits_dy):
            tokens.extend([f"<{digit_dx}>", f"<{digit_dy}>"])

        # Encode button flags as hex digits (0-15)
        flag_value = int(event.button_flags)

        # Convert to hex and pad to 3 digits, then split into individual hex digits
        hex_str = f"{flag_value:03x}"  # 3 hex digits, e.g., "401" for 0x401
        for hex_digit in hex_str:
            tokens.append(f"<{int(hex_digit, 16)}>")  # Convert hex char to int (0-15)

        # Add button data if non-zero (for wheel events)
        if event.button_data != 0:
            # NOTE: button_data is USHORT and is multiple if 120=WHEEL_DELTA. See: https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-rawmouse
            button_data = event.button_data
            if button_data >= 32768:
                button_data -= 65536
            button_data //= 120

            # Validate scroll range
            min_scroll, max_scroll = self.config.mouse_scroll_range
            if not (min_scroll <= button_data <= max_scroll):
                raise ValueError(
                    f"Mouse scroll value {button_data} is outside valid range [{min_scroll}, {max_scroll}]"
                )

            # Use signed bases for scroll encoding
            signed_bases = [2] + self.config.mouse_scroll_bases
            digits = quantize_to_digits(button_data, signed_bases)
            tokens.extend(f"<{digit}>" for digit in digits)

        return tokens

    def _decode_mouse_deltas(self, tokens: List[str]) -> Tuple[int, int]:
        """Decode quantized mouse deltas."""
        delta_tokens = tokens[1:]  # Skip <MOUSE>

        # Check for enough tokens: 2 pairs of deltas (dx, dy) with sign bits. Flag digits are handled separately.
        expected_delta_tokens = len(self.config.mouse_delta_bases) * 2 + 2
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

        # Use signed bases (add [2] to front for sign bit)
        signed_bases = [2] + self.config.mouse_delta_bases

        # Reconstruct signed deltas from digits
        dx = digits_to_value(digits_dx, signed_bases, signed=True)
        dy = digits_to_value(digits_dy, signed_bases, signed=True)

        return dx, dy

    def encode(self, mcap_message: McapMessage) -> Tuple[str, List[ScreenCaptured]]:
        """Encode a single McapMessage object to hierarchical token format."""
        mcap_message = mcap_message if isinstance(mcap_message, McapMessage) else McapMessage(**mcap_message)

        base_tokens = self._encode_timestamp(mcap_message.timestamp)
        images = []

        # Parse message content
        try:
            msg_data = orjson.loads(mcap_message.message.decode("utf-8"))
        except (orjson.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Failed to parse message content: {e}")

        # Encode based on event type
        if mcap_message.topic == "keyboard":
            keyboard_event = KeyboardEvent(**msg_data)
            event_tokens = base_tokens + self._encode_keyboard(keyboard_event)
        elif mcap_message.topic == "mouse" or mcap_message.topic == "mouse/raw":
            raw_mouse_event = RawMouseEvent(**msg_data)
            mouse_tokens = self._encode_mouse(raw_mouse_event)
            event_tokens = base_tokens + mouse_tokens
        elif mcap_message.topic == "screen":
            screen_event = ScreenCaptured(**msg_data)
            # Insert a single placeholder token - EpisodeTokenizer will handle prefix/suffix/repetition
            event_tokens = base_tokens + [self.config.fake_image_placeholder]
            images.append(screen_event)
        else:
            raise ValueError(f"Unsupported event type: {mcap_message.topic}")

        encoded_event = f"<EVENT_START>{''.join(event_tokens)}<EVENT_END>"
        return encoded_event, images

    def _decode_timestamp(self, tokens: List[str]) -> int:
        """Decode timestamp tokens back to nanoseconds."""
        if len(tokens) != len(self.config.timestamp_bases):
            raise ValueError(f"Invalid timestamp tokens: {tokens}")

        # Parse digits from tokens
        digits = []
        for i in range(len(tokens)):
            digit_match = re.match(r"<(\d+)>", tokens[i])
            if not digit_match:
                raise ValueError(f"Invalid timestamp digit token: {tokens[i]}")
            digits.append(int(digit_match.group(1)))

        # Reconstruct timestamp units using integer approach
        timestamp_units = digits_to_value(digits, self.config.timestamp_bases, signed=False)

        # Convert back to nanoseconds
        timestamp_ns = timestamp_units * self.config.timestamp_unit_ns
        return timestamp_ns

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

        # Extract button flags from hex digits
        delta_token_count = len(self.config.mouse_delta_bases) * 2 + 2
        flag_start_idx = 1 + delta_token_count

        if len(tokens) < flag_start_idx + 3:
            raise ValueError("Missing flag tokens")

        # Extract 3 hex digits for flags
        hex_digits = []
        for i in range(3):
            flag_token = tokens[flag_start_idx + i]
            flag_match = re.match(r"<(\d+)>", flag_token)
            if not flag_match:
                raise ValueError(f"Invalid flag token: {flag_token}")
            digit = int(flag_match.group(1))
            if digit > 15:
                raise ValueError(f"Invalid hex digit: {digit}")
            hex_digits.append(f"{digit:x}")

        # Reconstruct flag value from hex digits
        hex_str = "".join(hex_digits)
        button_flags = int(hex_str, 16)

        # Extract button data if present
        button_data = 0
        button_data_idx = flag_start_idx + 3
        if len(tokens) > button_data_idx:
            signed_bases = [2] + self.config.mouse_scroll_bases
            expected_tokens = len(signed_bases)

            # Validate exact token count
            if len(tokens) != button_data_idx + expected_tokens:
                raise ValueError(
                    f"Invalid scroll token count: expected {expected_tokens}, got {len(tokens) - button_data_idx}"
                )

            # Parse scroll tokens
            digits = []
            for i in range(expected_tokens):
                token = tokens[button_data_idx + i]
                match = re.match(r"<(\d+)>", token)
                if not match:
                    raise ValueError(f"Invalid scroll token: {token}")
                digits.append(int(match.group(1)))

            scroll_value = digits_to_value(digits, signed_bases, signed=True)
            button_data = scroll_value * 120

        return RawMouseEvent(
            last_x=dx, last_y=dy, button_flags=RawMouseEvent.ButtonFlags(button_flags), button_data=button_data
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

        timestamp_token_count = len(self.config.timestamp_bases)
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
                message=orjson.dumps(msg_data),
            )
        elif event_type_token == "<MOUSE>":
            raw_mouse_event = self._decode_mouse(tokens[timestamp_token_count:])
            msg_data = {
                "last_x": raw_mouse_event.last_x,
                "last_y": raw_mouse_event.last_y,
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
                message=orjson.dumps(msg_data),
            )
        elif event_type_token == self.config.fake_image_placeholder:
            # Simple image token - EpisodeTokenizer handles prefix/suffix/repetition
            if not images:
                warnings.warn("No image data provided for screen event", UserWarning)
                images = [ScreenCaptured(utc_ns=timestamp_ns, media_ref={"uri": "placeholder"})]
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
        return _generate_vocab()
