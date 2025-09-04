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
    """Event tokens for factorized encoding."""

    EVENT_START = "<EVENT_START>"
    EVENT_END = "<EVENT_END>"
    KEYBOARD = "<KEYBOARD>"
    MOUSE = "<MOUSE>"
    SCREEN = "<SCREEN>"
    PRESS = "<press>"
    RELEASE = "<release>"
    SIGN_PLUS = "<SIGN_PLUS>"
    SIGN_MINUS = "<SIGN_MINUS>"


def _generate_keyboard_tokens() -> Set[str]:
    """Generate keyboard-specific VK code tokens."""
    return {f"<VK_{i}>" for i in range(256)}


def _generate_mouse_button_tokens() -> Set[str]:
    """Generate mouse-specific button flag tokens."""
    return {f"<MB_{i}>" for i in range(16)}  # Hex digits 0-15 for button flags


@dataclass
class FactorizedEventEncoderConfig(BaseEventEncoderConfig):
    """Configuration for FactorizedEventEncoder."""

    # Minimal timestamp unit (default: 10ms)
    # Provides 10-second range: 10*10*10 = 1000 units * 10ms = 10 seconds
    timestamp_unit_ns: int = 10 * TimeUnits.MSECOND

    # Timestamp encoding bases: [10, 10, 10] = 1000 total units
    # Range: 0 to 9.99 seconds in 10ms increments
    timestamp_bases: List[int] = field(default_factory=lambda: [10, 10, 10])

    # Mouse delta encoding bases: [2, 10, 10, 10] = 2000 total values
    # With sign token [SIGN_PLUS/MINUS, 2, 10, 10, 10] = 2000 total, range: -1999 to +1999 pixels
    # Accommodates large mouse movements while using only digits 0-9
    mouse_delta_bases: List[int] = field(default_factory=lambda: [2, 10, 10, 10])

    # Mouse scroll encoding bases: [10] = 10 total values
    # With sign token [SIGN_PLUS/MINUS, 10] = 20 total, range: -9 to +9 scroll units
    # Each unit represents 120 (WHEEL_DELTA) in button_data
    mouse_scroll_bases: List[int] = field(default_factory=lambda: [10])

    def _signed_range(self, bases: List[int]) -> Tuple[int, int]:
        """Calculate signed range from bases with separate sign tokens."""
        # Calculate magnitude range (0 to total_range-1)
        total_range = 1
        for base in bases:
            total_range *= base
        max_magnitude = total_range - 1
        # With separate sign tokens, range is -max_magnitude to +max_magnitude
        min_val, max_val = -max_magnitude, max_magnitude
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


def digits_to_value(digits: List[int], bases: List[int]) -> int:
    """
    Reconstruct integer from digits.

    Args:
        digits: List of digits (len(bases) total)
        bases: List of bases for each quantization level

    Returns:
        Reconstructed integer (always unsigned since we use separate sign tokens)

    Examples:
        # Unsigned representation (current system)
        >>> digits_to_value([0, 6, 4], [10, 10, 10])
        64  # <0><6><4> -> 64
        >>> digits_to_value([1, 9, 9, 9], [2, 10, 10, 10])
        1999  # <1><9><9><9> -> 1999 (magnitude only, sign handled separately)
    """
    if len(digits) != len(bases):
        raise ValueError(f"Digits length {len(digits)} must match bases length {len(bases)}")

    # Always treat as unsigned since we use separate sign tokens
    # Reconstruct value using positional notation (most to least significant)
    encoded_value = 0
    for digit, base in zip(digits, bases):
        encoded_value = encoded_value * base + digit

    return encoded_value


def _generate_vocab() -> Set[str]:
    """
    Generate the factorized token vocabulary.

    Includes:
    - Event structure tokens (START, END, KEYBOARD, MOUSE, SCREEN, etc.)
    - Sign tokens (SIGN_PLUS, SIGN_MINUS) for signed values
    - Keyboard-specific tokens <VK_0> to <VK_255> for VK codes
    - Mouse-specific tokens <MB_0> to <MB_15> for button flag hex digits
    - Numeric tokens <0> to <9> for:
      * Timestamp digits
      * Quantized mouse delta digits
      * Quantized scroll digits
    """
    return {
        EventToken.EVENT_START.value,
        EventToken.EVENT_END.value,
        EventToken.KEYBOARD.value,
        EventToken.PRESS.value,
        EventToken.RELEASE.value,
        EventToken.MOUSE.value,
        EventToken.SCREEN.value,
        EventToken.SIGN_PLUS.value,
        EventToken.SIGN_MINUS.value,
        *_generate_keyboard_tokens(),
        *_generate_mouse_button_tokens(),
        *[f"<{i}>" for i in range(10)],  # Only digits 0-9
    }


# Helper function for token parsing
def _extract_digit(token: str) -> int:
    """Extract digit from token format <digit>."""
    match = re.match(r"<(\d+)>", token)
    if not match:
        raise InvalidTokenError(f"Invalid token format: {token}")
    return int(match.group(1))


class FactorizedEventEncoder(BaseEventEncoder):
    """Factorized event encoder: <EVENT_START><TYPE><TIMESTAMP><DATA><EVENT_END>"""

    def __init__(self, config: Optional[FactorizedEventEncoderConfig] = None, **kwargs):
        if config is None:
            config = FactorizedEventEncoderConfig()
        # Merge config with any keyword overrides
        self.config = FactorizedEventEncoderConfig(**(config.__dict__ | kwargs))

        # Verify token ranges once during initialization to catch config errors early
        self._verify_configuration()

    def _verify_configuration(self) -> None:
        """Verify that all configuration values will produce valid tokens."""
        # Check that all bases produce digits within token range <0> to <9>
        for name, bases in [
            ("timestamp", self.config.timestamp_bases),
            ("mouse_delta", self.config.mouse_delta_bases),
            ("mouse_scroll", self.config.mouse_scroll_bases),
        ]:
            if max(bases) > 10:
                raise InvalidInputError(f"{name} base {max(bases)} produces digits > 9, which is not supported.")

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
        """Encode keyboard data: [<VK_vk>, <action>]"""
        return [f"<VK_{event.vk}>", f"<{event.event_type}>"]

    def _decode_keyboard_data(self, tokens: List[str]) -> KeyboardEvent:
        """Decode keyboard data tokens."""
        if len(tokens) != 2:
            raise InvalidTokenError(f"Expected 2 keyboard tokens, got {len(tokens)}")

        # Extract VK code from <VK_xxx> format
        vk_match = re.match(r"<VK_(\d+)>", tokens[0])
        if not vk_match:
            raise InvalidTokenError(f"Invalid VK token format: {tokens[0]}")
        vk = int(vk_match.group(1))

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
        # Validate mouse delta range and clamp values without modifying input
        min_delta, max_delta = self.config.mouse_delta_range
        if not (min_delta <= event.dx <= max_delta) or not (min_delta <= event.dy <= max_delta):
            warnings.warn(
                f"Mouse delta value ({event.dx},{event.dy}) is outside valid range ({min_delta}, {max_delta}). Clamping."
            )
            # Use clamped values for encoding without modifying the input event
            dx = max(min_delta, min(max_delta, event.dx))
            dy = max(min_delta, min(max_delta, event.dy))
        else:
            # Use original values when no clamping is needed
            dx = event.dx
            dy = event.dy

        tokens = []

        # Encode movement deltas sequentially: dx first, then dy
        # dx encoding: sign token + magnitude digits
        if dx >= 0:
            tokens.append(EventToken.SIGN_PLUS.value)
            dx_magnitude = dx
        else:
            tokens.append(EventToken.SIGN_MINUS.value)
            dx_magnitude = -dx

        # Encode dx magnitude digits
        digits_dx = quantize_to_digits(dx_magnitude, self.config.mouse_delta_bases)
        tokens.extend([f"<{digit}>" for digit in digits_dx])

        # dy encoding: sign token + magnitude digits
        if dy >= 0:
            tokens.append(EventToken.SIGN_PLUS.value)
            dy_magnitude = dy
        else:
            tokens.append(EventToken.SIGN_MINUS.value)
            dy_magnitude = -dy

        # Encode dy magnitude digits
        digits_dy = quantize_to_digits(dy_magnitude, self.config.mouse_delta_bases)
        tokens.extend([f"<{digit}>" for digit in digits_dy])

        # Encode button flags as 3-digit hex (0x000-0xFFF range)
        # Each hex digit (0-15) becomes a separate mouse button token
        flag_value = int(event.button_flags)
        hex_str = f"{flag_value:03x}"  # Always 3 digits: 000-FFF
        for hex_digit in hex_str:
            hex_digit_int = int(hex_digit, 16)
            tokens.append(f"<MB_{hex_digit_int}>")

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

            # Encode scroll with separate sign token
            if button_data >= 0:
                tokens.append(EventToken.SIGN_PLUS.value)
                scroll_magnitude = button_data
            else:
                tokens.append(EventToken.SIGN_MINUS.value)
                scroll_magnitude = -button_data

            # Encode magnitude using scroll bases
            digits = quantize_to_digits(scroll_magnitude, self.config.mouse_scroll_bases)
            tokens.extend(f"<{digit}>" for digit in digits)

        return tokens

    def _decode_mouse_data(self, tokens: List[str]) -> RawMouseEvent:
        """Decode mouse data tokens."""
        # Calculate minimum required tokens: deltas + button flags
        delta_tokens_needed = (
            len(self.config.mouse_delta_bases) * 2 + 2
        )  # *2 for dx/dy magnitude digits, +2 for sign tokens
        button_flags_needed = 3  # 3 hex digits for button flags
        min_tokens = delta_tokens_needed + button_flags_needed
        max_tokens = min_tokens + 1 + len(self.config.mouse_scroll_bases)  # +1 for scroll sign bit

        if not (min_tokens <= len(tokens) <= max_tokens):
            raise InvalidTokenError(f"Expected {min_tokens} to {max_tokens} mouse tokens, got {len(tokens)}")

        # Decode movement deltas from sign tokens + magnitude tokens
        delta_token_count = len(self.config.mouse_delta_bases) * 2 + 2  # *2 for dx/dy, +2 for sign tokens
        delta_tokens = tokens[:delta_token_count]
        dx, dy = self._decode_mouse_deltas(delta_tokens)

        # Decode button flags from 3 mouse button tokens
        # Each token represents one hex digit (0-15), reconstruct 3-digit hex value
        flag_start = delta_token_count
        hex_digits = ""
        for i in range(3):
            mb_match = re.match(r"<MB_(\d+)>", tokens[flag_start + i])
            if not mb_match:
                raise InvalidTokenError(f"Invalid mouse button token format: {tokens[flag_start + i]}")
            hex_digit_int = int(mb_match.group(1))
            hex_digits += f"{hex_digit_int:x}"
        button_flags = int(hex_digits, 16)  # Convert back to integer (0x000-0xFFF)

        # Decode scroll data if present
        # Reverse the encoding: multiply by WHEEL_DELTA (120) and convert back to USHORT
        button_data = 0
        scroll_start = flag_start + 3
        if len(tokens) > scroll_start:
            # Extract scroll sign token
            scroll_sign_token = tokens[scroll_start]
            if scroll_sign_token == EventToken.SIGN_PLUS.value:
                scroll_sign = 1
            elif scroll_sign_token == EventToken.SIGN_MINUS.value:
                scroll_sign = -1
            else:
                raise InvalidTokenError(f"Invalid scroll sign token: {scroll_sign_token}")

            # Extract scroll magnitude digits
            scroll_magnitude_tokens = tokens[scroll_start + 1 : scroll_start + 1 + len(self.config.mouse_scroll_bases)]
            digits = [_extract_digit(token) for token in scroll_magnitude_tokens]
            scroll_magnitude = digits_to_value(digits, self.config.mouse_scroll_bases)
            button_data = scroll_sign * scroll_magnitude * 120  # Restore WHEEL_DELTA scaling

        return RawMouseEvent(
            last_x=dx, last_y=dy, button_flags=RawMouseEvent.ButtonFlags(button_flags), button_data=button_data
        )

    def _decode_mouse_deltas(self, delta_tokens: List[str]) -> Tuple[int, int]:
        """Decode quantized mouse deltas from sequential sign + magnitude tokens."""
        # Expected format: <dx_sign><dx_digits...><dy_sign><dy_digits...>
        num_magnitude_digits = len(self.config.mouse_delta_bases)

        # Extract dx sign token
        dx_sign_token = delta_tokens[0]
        if dx_sign_token == EventToken.SIGN_PLUS.value:
            dx_sign = 1
        elif dx_sign_token == EventToken.SIGN_MINUS.value:
            dx_sign = -1
        else:
            raise InvalidTokenError(f"Invalid dx sign token: {dx_sign_token}")

        # Extract dx magnitude digits
        dx_start = 1
        dx_end = dx_start + num_magnitude_digits
        dx_magnitude_tokens = delta_tokens[dx_start:dx_end]
        digits_dx = [_extract_digit(token) for token in dx_magnitude_tokens]
        dx_magnitude = digits_to_value(digits_dx, self.config.mouse_delta_bases)

        # Extract dy sign token
        dy_sign_token = delta_tokens[dx_end]
        if dy_sign_token == EventToken.SIGN_PLUS.value:
            dy_sign = 1
        elif dy_sign_token == EventToken.SIGN_MINUS.value:
            dy_sign = -1
        else:
            raise InvalidTokenError(f"Invalid dy sign token: {dy_sign_token}")

        # Extract dy magnitude digits
        dy_start = dx_end + 1
        dy_end = dy_start + num_magnitude_digits
        dy_magnitude_tokens = delta_tokens[dy_start:dy_end]
        digits_dy = [_extract_digit(token) for token in dy_magnitude_tokens]
        dy_magnitude = digits_to_value(digits_dy, self.config.mouse_delta_bases)

        # Apply signs and return
        return dx_sign * dx_magnitude, dy_sign * dy_magnitude

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
