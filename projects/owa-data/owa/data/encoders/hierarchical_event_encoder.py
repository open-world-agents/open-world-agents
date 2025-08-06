import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Set, Tuple, Union

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

    # -8 to +8 seconds (half-range: 8 seconds)
    max_timestamp_range_ns: int = 8 * TimeUnits.SECOND
    # 16 seconds in 10ms intervals
    timestamp_bases: List[int] = field(default_factory=lambda: [16, 10, 10])
    # Mouse delta max ranges: (max_x, max_y) for -max_x to +max_x and -max_y to +max_y
    max_mouse_delta: Tuple[int, int] = (1000, 1000)
    # -1000 to +1000 in 1 pixel unit intervals
    mouse_delta_bases: List[int] = field(default_factory=lambda: [20, 10, 10])


def quantize_to_digits(value: Union[float, Fraction], bases: List[int]) -> List[int]:
    """
    Quantize a normalized value (0.0-1.0) to multi-level digits.

    Args:
        value: Normalized value between 0.0 and 1.0 (can be float or Fraction for precision)
        bases: List of bases for each quantization level (e.g., [16, 16, 16] for 3-level hex)

    Returns:
        List of digits for each level

    Example:
        >>> quantize_to_digits(0.6875, [16, 16, 16])
        [11, 0, 0]  # 0xB00 in hex = 0.6875
    """
    digits = []

    # Convert to Fraction for exact arithmetic if not already
    if isinstance(value, Fraction):
        remaining = value
    else:
        remaining = Fraction(value).limit_denominator()

    # Clamp to [0, 1] using fractions
    remaining = max(Fraction(0), min(Fraction(1), remaining))

    for base in bases:
        # Calculate digit using exact arithmetic
        digit_fraction = remaining * base
        digit = int(digit_fraction)
        digits.append(digit)
        # Update remaining using exact subtraction
        remaining = digit_fraction - digit

    return digits


def digits_to_value(digits: List[int], bases: List[int]) -> Fraction:
    """
    Reconstruct normalized value from multi-level digits.

    Args:
        digits: List of digits for each level
        bases: List of bases for each quantization level

    Returns:
        Reconstructed normalized value between 0 and 1 as an accurate Fraction

    Example:
        >>> digits_to_value([11, 0, 0], [16, 16, 16])
        Fraction(11, 16)  # 0xB00 in hex = 11/16
    """
    if len(digits) != len(bases):
        raise ValueError(f"Digits length {len(digits)} must match bases length {len(bases)}")

    value = Fraction(0)
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
        # max_timestamp_range_ns is half-range, so total range is 2 * max_timestamp_range_ns
        total_range = 2 * self.config.max_timestamp_range_ns
        mod_timestamp = timestamp_ns % total_range
        norm_timestamp = mod_timestamp / total_range

        # Quantize to digits
        digits = quantize_to_digits(norm_timestamp, self.config.timestamp_bases)

        # Create tokens
        tokens = ["<TIMESTAMP>"] + [f"<{digit}>" for digit in digits]
        return tokens

    def _encode_keyboard(self, event: KeyboardEvent) -> List[str]:
        """Encode keyboard event: [<KEYBOARD>, <vk>, <action>]"""
        return ["<KEYBOARD>", f"<{event.vk}>", f"<{event.event_type}>"]

    def _encode_mouse(self, event: RawMouseEvent) -> List[str]:
        """
        Encode raw mouse event as: <MOUSE><dx0><dy0><dx1><dy1><flag0><flag1><flag2><optional_button_data>

        Each flag is encoded as a hex digit (0-15).
        """
        # Normalize dx, dy to [0, 1] range using max_mouse_delta tuple (half-ranges)
        max_x, max_y = self.config.max_mouse_delta
        dx_clamped = max(-max_x, min(max_x, event.dx))
        dy_clamped = max(-max_y, min(max_y, event.dy))

        # Use fractions for exact normalization to avoid floating-point precision loss
        norm_dx = Fraction(dx_clamped + max_x) / Fraction(2 * max_x)
        norm_dy = Fraction(dy_clamped + max_y) / Fraction(2 * max_y)

        tokens = ["<MOUSE>"]

        # Quantize deltas to digits using exact Fraction values
        digits_dx = quantize_to_digits(norm_dx, self.config.mouse_delta_bases)
        digits_dy = quantize_to_digits(norm_dy, self.config.mouse_delta_bases)

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
            tokens.append(f"<{button_data}>")

        return tokens

    def _decode_mouse_deltas(self, tokens: List[str]) -> Tuple[int, int]:
        """Decode quantized mouse deltas."""
        delta_tokens = tokens[1:]  # Skip <MOUSE>

        # Extract delta digit pairs
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

        # Convert back to actual delta values using max_mouse_delta tuple (half-ranges)
        max_x, max_y = self.config.max_mouse_delta
        dx = int(round(norm_dx * (2 * max_x) - max_x))
        dy = int(round(norm_dy * (2 * max_y) - max_y))

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
            encoded_event = f"<EVENT_START>{''.join(event_tokens)}<EVENT_END>"
        elif mcap_message.topic == "mouse" or mcap_message.topic == "mouse/raw":
            raw_mouse_event = RawMouseEvent(**msg_data)
            mouse_tokens = self._encode_mouse(raw_mouse_event)
            event_tokens = base_tokens + mouse_tokens
            encoded_event = f"<EVENT_START>{''.join(event_tokens)}<EVENT_END>"
        elif mcap_message.topic == "screen":
            screen_event = ScreenCaptured(**msg_data)
            event_tokens = base_tokens + [
                self.config.image_token_prefix,
                self.config.image_token,
                self.config.image_token_suffix,
            ]
            encoded_event = f"<EVENT_START>{''.join(event_tokens)}<EVENT_END>"
            images.append(screen_event)
        else:
            raise ValueError(f"Unsupported event type: {mcap_message.topic}")

        return encoded_event, images

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
        # max_timestamp_range_ns is half-range, so total range is 2 * max_timestamp_range_ns
        total_range = 2 * self.config.max_timestamp_range_ns
        return int(norm_timestamp * total_range)

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
        delta_token_count = len(self.config.mouse_delta_bases) * 2
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
            button_data_token = tokens[button_data_idx]
            button_data_match = re.match(r"<(-?\d+)>", button_data_token)
            if button_data_match:
                # Multiply by 120 to restore original button_data value (reverse of encoding division)
                button_data = int(button_data_match.group(1)) * 120

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
        else:
            # Check if this is the start of an image token sequence
            # Parse the image_token_prefix to get individual tokens
            prefix_tokens = re.findall(r"<[^>]*>", self.config.image_token_prefix)

            if prefix_tokens and event_type_token == prefix_tokens[0]:
                # This looks like the start of an image sequence
                expected_tokens = (
                    prefix_tokens + [self.config.image_token] + re.findall(r"<[^>]*>", self.config.image_token_suffix)
                )

                if len(tokens) < timestamp_token_count + len(expected_tokens):
                    raise ValueError("Not enough tokens for image sequence")

                # Verify the complete image token sequence
                actual_tokens = tokens[timestamp_token_count : timestamp_token_count + len(expected_tokens)]
                if actual_tokens != expected_tokens:
                    raise ValueError(
                        f"Invalid image token sequence: expected {expected_tokens} but got {actual_tokens}"
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
