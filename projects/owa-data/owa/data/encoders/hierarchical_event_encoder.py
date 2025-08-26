import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple

from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.core.time import TimeUnits
from owa.msgs.desktop.keyboard import KeyboardEvent
from owa.msgs.desktop.mouse import RawMouseEvent
from owa.msgs.desktop.screen import ScreenCaptured

from .base_encoder import BaseEventEncoder, BaseEventEncoderConfig
from .exceptions import (
    InvalidInputError,
    InvalidTokenError,
    UnsupportedInputError,
    UnsupportedTokenError,
)


class EventToken(str, Enum):
    """Event tokens for hierarchical encoding."""

    EVENT_START = "<EVENT_START>"
    EVENT_END = "<EVENT_END>"
    KEYBOARD = "<KEYBOARD>"
    MOUSE = "<MOUSE>"
    SCREEN = "<SCREEN>"
    PRESS = "<press>"
    RELEASE = "<release>"


@dataclass
class HierarchicalEventEncoderConfig(BaseEventEncoderConfig):
    """Configuration for HierarchicalEventEncoder."""

    # Minimal timestamp unit (default: 10ms)
    # Provides 16-second range: 16*10*10 = 1600 units * 10ms = 16 seconds
    timestamp_unit_ns: int = 10 * TimeUnits.MSECOND

    # Timestamp encoding bases: [16, 10, 10] = 1600 total units
    # Range: 0 to 15.99 seconds in 10ms increments
    timestamp_bases: List[int] = field(default_factory=lambda: [16, 10, 10])

    # Mouse delta encoding bases: [20, 10, 10] = 2000 total values
    # With sign bit [2, 20, 10, 10] = 4000 total, range: -2000 to +1999 pixels
    # Accommodates large mouse movements (-1000+ per tick)
    mouse_delta_bases: List[int] = field(default_factory=lambda: [20, 10, 10])

    # Mouse scroll encoding bases: [10] = 10 total values
    # With sign bit [2, 10] = 20 total, range: -10 to +9 scroll units
    # Each unit represents 120 (WHEEL_DELTA) in button_data
    mouse_scroll_bases: List[int] = field(default_factory=lambda: [10])

    def _signed_range(self, bases: List[int]) -> Tuple[int, int]:
        """Calculate signed range from bases."""
        total_range = 1
        for base in bases:
            total_range *= base
        min_val, max_val = -total_range, total_range - 1
        return min_val, max_val

    @property
    def timestamp_range(self) -> int:
        """Calculate valid timestamp range from bases."""
        total_range = 1
        for base in self.timestamp_bases:
            total_range *= base
        return total_range * self.timestamp_unit_ns

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
    digits = []
    remaining = value
    # Process bases from least to most significant (reverse order)
    for base in reversed(bases):
        digit = remaining % base  # Extract digit for this base
        digits.insert(0, digit)  # Insert at front to maintain order
        remaining //= base  # Move to next significance level
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
        signed = bases[0] == 2  # Auto-detect: sign bit if first base is 2

    # Reconstruct value using positional notation (most to least significant)
    encoded_value = 0
    for digit, base in zip(digits, bases):
        encoded_value = encoded_value * base + digit

    # Convert from unsigned to signed representation if needed
    if signed:
        total_range = 1
        for base in bases:
            total_range *= base
        # If value is in upper half of range, it represents a negative number
        if encoded_value >= total_range // 2:
            return encoded_value - total_range
    return encoded_value


def _generate_vocab() -> Set[str]:
    """
    Generate the hierarchical token vocabulary.

    Includes:
    - Event structure tokens (START, END, KEYBOARD, MOUSE, SCREEN, etc.)
    - Numeric tokens <0> to <255> for:
      * VK codes (0-255)
      * Quantized timestamp digits
      * Quantized mouse delta digits
      * Quantized scroll digits
      * Button flag hex digits (0-15)
    """
    return {
        EventToken.EVENT_START.value,
        EventToken.EVENT_END.value,
        EventToken.KEYBOARD.value,
        EventToken.PRESS.value,
        EventToken.RELEASE.value,
        EventToken.MOUSE.value,
        EventToken.SCREEN.value,
        *[f"<{i}>" for i in range(256)],
    }


# Helper function for token parsing
def _extract_digit(token: str) -> int:
    """Extract digit from token format <digit>."""
    match = re.match(r"<(\d+)>", token)
    if not match:
        raise InvalidTokenError(f"Invalid token format: {token}")
    return int(match.group(1))


class HierarchicalEventEncoder(BaseEventEncoder):
    """Hierarchical event encoder: <EVENT_START><TYPE><TIMESTAMP><DATA><EVENT_END>"""

    def __init__(self, config: Optional[HierarchicalEventEncoderConfig] = None, **kwargs):
        if config is None:
            config = HierarchicalEventEncoderConfig()
        # Merge config with any keyword overrides
        self.config = HierarchicalEventEncoderConfig(**(config.__dict__ | kwargs))

        # Verify token ranges once during initialization to catch config errors early
        self._verify_configuration()

    def _verify_configuration(self) -> None:
        """Verify that all configuration values will produce valid tokens."""
        # Check that all bases produce digits within token range <0> to <255>
        for name, bases in [
            ("timestamp", self.config.timestamp_bases),
            ("mouse_delta", [2] + self.config.mouse_delta_bases),  # Include sign bit
            ("mouse_scroll", [2] + self.config.mouse_scroll_bases),  # Include sign bit
        ]:
            if max(bases) > 256:
                raise InvalidInputError(f"{name} base {max(bases)} produces digits > 255")

    @property
    def vocab(self) -> Set[str]:
        return _generate_vocab()

    # ============================================================================
    # TIMESTAMP ENCODING/DECODING
    # ============================================================================

    def _encode_timestamp(self, timestamp_ns: int) -> List[str]:
        """Encode timestamp as tokens."""
        units = timestamp_ns // self.config.timestamp_unit_ns
        return [f"<{d}>" for d in quantize_to_digits(units, self.config.timestamp_bases)]

    def _decode_timestamp(self, tokens: List[str]) -> int:
        """Decode timestamp tokens back to nanoseconds."""
        digits = [_extract_digit(token) for token in tokens]
        units = digits_to_value(digits, self.config.timestamp_bases)
        return units * self.config.timestamp_unit_ns

    # ============================================================================
    # KEYBOARD ENCODING/DECODING
    # ============================================================================

    def _encode_keyboard_data(self, event: KeyboardEvent) -> List[str]:
        """Encode keyboard data: [<vk>, <action>]"""
        return [f"<{event.vk}>", f"<{event.event_type}>"]

    def _decode_keyboard_data(self, tokens: List[str]) -> KeyboardEvent:
        """Decode keyboard data tokens."""
        if len(tokens) != 2:
            raise InvalidTokenError(f"Expected 2 keyboard tokens, got {len(tokens)}")

        vk = _extract_digit(tokens[0])

        # Extract event type
        event_type_match = re.match(r"<(\w+)>", tokens[1])
        if not event_type_match:
            raise InvalidTokenError(f"Invalid event type token: {tokens[1]}")
        event_type = event_type_match.group(1)

        if event_type not in ("press", "release"):
            raise UnsupportedTokenError(f"Unsupported event type: {event_type}")

        return KeyboardEvent(event_type=event_type, vk=vk)

    # ============================================================================
    # MOUSE ENCODING/DECODING
    # ============================================================================

    def _encode_mouse_data(self, event: RawMouseEvent) -> List[str]:
        """Encode mouse data: movement + flags + optional scroll."""
        # Validate mouse delta range
        min_delta, max_delta = self.config.mouse_delta_range
        if not (min_delta <= event.dx <= max_delta) or not (min_delta <= event.dy <= max_delta):
            warnings.warn(
                f"Mouse delta value ({event.dx},{event.dy}) is outside valid range ({min_delta}, {max_delta}). Clamping."
            )
            event.last_x = max(min_delta, min(max_delta, event.last_x))
            event.last_y = max(min_delta, min(max_delta, event.last_y))

        tokens = []

        # Encode movement deltas with sign bit
        signed_bases = [2] + self.config.mouse_delta_bases
        digits_dx = quantize_to_digits(event.dx, signed_bases)
        digits_dy = quantize_to_digits(event.dy, signed_bases)
        # Interleave dx and dy digits: <dx0><dy0><dx1><dy1>...
        for digit_dx, digit_dy in zip(digits_dx, digits_dy):
            tokens.extend([f"<{digit_dx}>", f"<{digit_dy}>"])

        # Encode button flags as 3-digit hex (0x000-0xFFF range)
        # Each hex digit (0-15) becomes a separate token
        flag_value = int(event.button_flags)
        hex_str = f"{flag_value:03x}"  # Always 3 digits: 000-FFF
        for hex_digit in hex_str:
            hex_digit_int = int(hex_digit, 16)
            tokens.append(f"<{hex_digit_int}>")

        # Encode scroll data if present
        # button_data is USHORT (0-65535), convert to signed and scale by WHEEL_DELTA (120)
        # See: https://docs.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-rawmouse
        if event.button_data != 0:
            button_data = event.button_data
            if button_data >= 32768:  # Convert USHORT to signed
                button_data -= 65536
            button_data //= 120  # Scale by WHEEL_DELTA

            # Validate scroll range
            min_scroll, max_scroll = self.config.mouse_scroll_range
            if not (min_scroll <= button_data <= max_scroll):
                raise InvalidInputError(
                    f"Mouse scroll value {button_data} is outside valid range [{min_scroll}, {max_scroll}]"
                )

            signed_bases = [2] + self.config.mouse_scroll_bases
            digits = quantize_to_digits(button_data, signed_bases)
            tokens.extend(f"<{digit}>" for digit in digits)

        return tokens

    def _decode_mouse_data(self, tokens: List[str]) -> RawMouseEvent:
        """Decode mouse data tokens."""
        # Calculate minimum required tokens: deltas + button flags
        delta_tokens_needed = len(self.config.mouse_delta_bases) * 2 + 2  # *2 for dx/dy, +2 for sign bits
        button_flags_needed = 3  # 3 hex digits for button flags
        min_tokens = delta_tokens_needed + button_flags_needed
        max_tokens = min_tokens + 1 + len(self.config.mouse_scroll_bases)  # +1 for scroll sign bit

        if not (min_tokens <= len(tokens) <= max_tokens):
            raise InvalidTokenError(f"Expected {min_tokens} to {max_tokens} mouse tokens, got {len(tokens)}")

        # Decode movement deltas from interleaved dx/dy tokens
        dx, dy = self._decode_mouse_deltas(tokens)

        # Decode button flags from 3 hex digit tokens
        # Each token represents one hex digit (0-15), reconstruct 3-digit hex value
        flag_start = len(self.config.mouse_delta_bases) * 2 + 2
        hex_digits = "".join(f"{_extract_digit(tokens[flag_start + i]):x}" for i in range(3))
        button_flags = int(hex_digits, 16)  # Convert back to integer (0x000-0xFFF)

        # Decode scroll data if present
        # Reverse the encoding: multiply by WHEEL_DELTA (120) and convert back to USHORT
        button_data = 0
        scroll_start = flag_start + 3
        if len(tokens) > scroll_start:
            bases = [2] + self.config.mouse_scroll_bases
            digits = [_extract_digit(tokens[scroll_start + i]) for i in range(len(bases))]
            button_data = digits_to_value(digits, bases, signed=True) * 120  # Restore WHEEL_DELTA scaling

        return RawMouseEvent(
            last_x=dx, last_y=dy, button_flags=RawMouseEvent.ButtonFlags(button_flags), button_data=button_data
        )

    def _decode_mouse_deltas(self, delta_tokens: List[str]) -> Tuple[int, int]:
        """Decode quantized mouse deltas from interleaved token pairs."""
        expected = len(self.config.mouse_delta_bases) * 2 + 2  # *2 for dx/dy, +2 for sign bits

        # De-interleave dx and dy digits: even indices=dx, odd indices=dy
        digits_dx = [_extract_digit(delta_tokens[i]) for i in range(0, expected, 2)]
        digits_dy = [_extract_digit(delta_tokens[i]) for i in range(1, expected, 2)]

        # Reconstruct signed values using sign bit + quantization bases
        bases = [2] + self.config.mouse_delta_bases
        return digits_to_value(digits_dx, bases, signed=True), digits_to_value(digits_dy, bases, signed=True)

    # ============================================================================
    # MAIN ENCODE/DECODE METHODS
    # ============================================================================

    def encode(self, mcap_message: McapMessage) -> Tuple[str, List[ScreenCaptured]]:
        """Encode message to: <EVENT_START><TYPE><TIMESTAMP><DATA><EVENT_END>"""
        try:
            # Use decoded message directly
            decoded = mcap_message.decoded
        except Exception as e:
            raise InvalidInputError(f"Failed to decode message: {e}") from e

        # Encode timestamp to quantized tokens
        timestamp_tokens = self._encode_timestamp(mcap_message.timestamp)

        # Encode by topic type - each has different data structure
        if mcap_message.topic == "keyboard":
            data_tokens = self._encode_keyboard_data(decoded)
            event_tokens = [EventToken.KEYBOARD.value] + timestamp_tokens + data_tokens
            return f"{EventToken.EVENT_START.value}{''.join(event_tokens)}{EventToken.EVENT_END.value}", []
        elif mcap_message.topic in ("mouse", "mouse/raw"):
            data_tokens = self._encode_mouse_data(decoded)
            event_tokens = [EventToken.MOUSE.value] + timestamp_tokens + data_tokens
            return f"{EventToken.EVENT_START.value}{''.join(event_tokens)}{EventToken.EVENT_END.value}", []
        elif mcap_message.topic == "screen":
            # Screen events: <EVENT_START><SCREEN><TIMESTAMP><fake_image_placeholder><EVENT_END>
            event_tokens = [EventToken.SCREEN.value] + timestamp_tokens + [self.config.fake_image_placeholder]
            return f"{EventToken.EVENT_START.value}{''.join(event_tokens)}{EventToken.EVENT_END.value}", [decoded]
        else:
            raise UnsupportedInputError(f"Unsupported topic: {mcap_message.topic}")

    def decode(self, encoded_data: str, images: Optional[List[ScreenCaptured]] = None) -> McapMessage:
        """Decode: <EVENT_START><TYPE><TIMESTAMP><DATA><EVENT_END>"""
        # Validate and parse token structure
        if not (
            encoded_data.startswith(EventToken.EVENT_START.value) and encoded_data.endswith(EventToken.EVENT_END.value)
        ):
            raise InvalidTokenError("Missing EVENT_START or EVENT_END tokens")

        # Extract content between start/end tokens and parse individual tokens
        content = encoded_data[len(EventToken.EVENT_START.value) : -len(EventToken.EVENT_END.value)]
        tokens = re.findall(r"<[^>]*>", content)

        # Split tokens into components: [TYPE][TIMESTAMP][DATA]
        ts_len = len(self.config.timestamp_bases)
        if len(tokens) < 1 + ts_len:  # Need at least type + timestamp
            raise InvalidTokenError("Token sequence too short")
        timestamp_ns = self._decode_timestamp(tokens[1 : 1 + ts_len])
        data_tokens = tokens[1 + ts_len :]  # Remaining tokens are event-specific data

        # Decode by type
        event_type = tokens[0]
        if event_type == EventToken.KEYBOARD.value:
            event = self._decode_keyboard_data(data_tokens)
            return McapMessage(
                topic="keyboard",
                timestamp=timestamp_ns,
                message_type="desktop/KeyboardEvent",
                message=event.model_dump_json().encode(),
            )
        elif event_type == EventToken.MOUSE.value:
            event = self._decode_mouse_data(data_tokens)
            return McapMessage(
                topic="mouse/raw",
                timestamp=timestamp_ns,
                message_type="desktop/RawMouseEvent",
                message=event.model_dump_json().encode(),
            )
        elif event_type == EventToken.SCREEN.value:
            # Screen events should have exactly one data token: fake_image_placeholder
            if len(data_tokens) != 1 or data_tokens[0] != self.config.fake_image_placeholder:
                raise InvalidTokenError(
                    f"Screen event should have exactly one fake_image_placeholder token, got {data_tokens}"
                )

            if not images:
                warnings.warn("No image data provided for screen event", UserWarning)
                from owa.msgs.desktop.screen import MediaRef

                images = [ScreenCaptured(utc_ns=timestamp_ns, media_ref=MediaRef(uri="placeholder"))]
            return McapMessage(
                topic="screen",
                timestamp=timestamp_ns,
                message_type="desktop/ScreenCaptured",
                message=images[0].model_dump_json().encode(),
            )
        else:
            raise UnsupportedTokenError(f"Unknown event type: {event_type}")

    def get_vocab(self) -> Set[str]:
        """Get all tokens in the vocabulary."""
        return _generate_vocab()
