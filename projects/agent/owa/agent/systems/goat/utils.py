import re
from typing import Any, List, Optional, Sequence, Tuple

from pydantic import Field
from pydantic.dataclasses import dataclass

from owa.agent.core import Event
from owa.core.time import TimeUnits
from owa.env.desktop.msg import KeyboardEvent, MouseEvent


@dataclass
class EventProcessorConfig:
    timestamp_min_ns: int = -2 * TimeUnits.SECOND
    timestamp_max_ns: int = 2 * TimeUnits.SECOND
    timestamp_interval_ns: int = 33 * TimeUnits.MSECOND  # 33ms == 30fps
    timestamp_token_format: str = "<TIMESTAMP_{idx}>"
    # all token count = (timestamp_max_ns - timestamp_min_ns) / timestamp_interval_ns + 1

    keyboard_vk: int = 256
    keyboard_token_format: str = "<KEYBOARD_{vk}_{pressed}>"
    # e.g. <KEYBOARD_65_press>, <KEYBOARD_65_release>
    # all token count = (keyboard_vk + 1) * 2

    mouse_tokens: list[int] = Field(default_factory=lambda: [256, 256, 256])
    screen_size: tuple[int, int] = (1920, 1080)
    mouse_move_format: str = "<MOUSE_move_{level}_{idx_x}_{idx_y}>"
    mouse_click_format: str = "<MOUSE_click_{button}_{pressed}>"
    mouse_scroll_format: str = "<MOUSE_scroll_{dx}_{dy}>"
    # For move, we utilize residual-quantize. [256, 256, 256] can quantize 2D space into 256*256*256 with 3 token, which fits to 16M pixels ~= 12.3% of FHD
    # Also within quantization, regard the screen_size. normalize the x, y coordinate to [0, 1] and then quantize (x, y) to [0, 255]
    # e.g. <MOUSE_move><RQ_0_253_45><RQ_1_255_100><RQ_2_88_201></MOUSE_move>
    # <MOUSE_click_left_press>, <MOUSE_scroll_0_1>


class MouseTokenizer:
    def __init__(self, config: EventProcessorConfig):
        self.cfg = config

    def move_tokens(self, event: MouseEvent, screen_size: Optional[Tuple[int, int]] = None) -> List[str]:
        """
        Quantize (x, y) with residual quantization (multi-level, jointly as a pair).
        Returns: [<MOUSE_move>, <move_level_0>, ... <move_level_n>, </MOUSE_move>]
        """
        if screen_size is None:
            screen_size = self.cfg.screen_size
        x, y = event.x, event.y
        fx = x / max(screen_size[0] - 1, 1)
        fy = y / max(screen_size[1] - 1, 1)

        # Jointly quantize the pair (x, y) repeatedly at each level
        vx, vy = fx, fy
        tokens = []
        for i, nbins in enumerate(self.cfg.mouse_tokens):
            idx_x = int(round(vx * (nbins - 1)))
            idx_y = int(round(vy * (nbins - 1)))
            tokens.append(self.cfg.mouse_move_format.format(level=i, idx_x=idx_x, idx_y=idx_y))
            # Calculate residuals for next level and normalize into [0, 1]
            vx = min(max(vx * (nbins - 1) - idx_x + 0.5, 0.0), 1.0)
            vy = min(max(vy * (nbins - 1) - idx_y + 0.5, 0.0), 1.0)
        return ["<MOUSE_move>"] + tokens + ["</MOUSE_move>"]

    def move_inv(self, tokens: Sequence[str], screen_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Inverse of move_tokens - reconstruct (x, y).
        """
        if screen_size is None:
            screen_size = self.cfg.screen_size

        # Extract indices from tokens
        indices = []
        for token in tokens:
            m = re.match(r"<MOUSE_move_(\d+)_(\d+)_(\d+)>", token)
            if m:
                level, idx_x, idx_y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                indices.append((level, idx_x, idx_y))

        # Sort by level to ensure correct order
        indices.sort(key=lambda x: x[0])

        if not indices:
            return 0, 0  # Fallback if no valid tokens

        # The residual quantization reconstruction
        x_normalized, y_normalized = 0.0, 0.0
        factor = 1.0

        for i, (_, idx_x, idx_y) in enumerate(indices):
            nbins = self.cfg.mouse_tokens[i]
            bin_size = 1.0 / (nbins - 1) if nbins > 1 else 1.0

            # Add the contribution from this level
            x_normalized += idx_x * bin_size * factor
            y_normalized += idx_y * bin_size * factor

            # Reduce contribution factor for next level
            factor *= bin_size

        # Convert normalized coordinates to pixel coordinates
        pix_x = int(round(x_normalized * (screen_size[0] - 1)))
        pix_y = int(round(y_normalized * (screen_size[1] - 1)))

        return pix_x, pix_y

    def click_token(self, event: MouseEvent) -> str:
        """
        Compose a click token from MouseEvent.
        """
        button = event.button or "unknown"
        press_str = "press" if bool(event.pressed) else "release"
        return self.cfg.mouse_click_format.format(button=button, pressed=press_str)

    def scroll_token(self, event: MouseEvent) -> str:
        """
        Compose a scroll token from MouseEvent.
        """
        dx = event.dx if event.dx is not None else 0
        dy = event.dy if event.dy is not None else 0
        return self.cfg.mouse_scroll_format.format(dx=dx, dy=dy)

    def to_tokens(self, event: MouseEvent, screen_size: Optional[Tuple[int, int]] = None) -> List[str]:
        """
        Encode MouseEvent to sequence of tokens, always including move tokens for position.
        """
        tokens = self.move_tokens(event, screen_size)
        if event.event_type == "move":
            return tokens
        elif event.event_type == "click":
            return tokens + [self.click_token(event)]
        elif event.event_type == "scroll":
            return tokens + [self.scroll_token(event)]
        else:
            return tokens + ["<MOUSE_unknown>"]

    def from_tokens(
        self, tokens: Sequence[str], idx: int = 0, screen_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[Optional[MouseEvent], int]:
        """
        Parse MouseEvent from tokens starting at idx.
        Returns (MouseEvent, new idx)
        """
        if screen_size is None:
            screen_size = self.cfg.screen_size

        # move tokens always first
        if idx >= len(tokens) or tokens[idx] != "<MOUSE_move>":
            return None, idx + 1

        move_tokens = []
        j = idx + 1
        while j < len(tokens) and tokens[j] != "</MOUSE_move>":
            move_tokens.append(tokens[j])
            j += 1

        if j >= len(tokens):
            return None, j

        j += 1  # skip </MOUSE_move>
        x, y = self.move_inv(move_tokens, screen_size)

        # Next token: click or scroll or unknown or nothing (move)
        if j < len(tokens):
            if tokens[j].startswith("<MOUSE_click_"):
                m = re.match(r"<MOUSE_click_(\w+)_(\w+)>", tokens[j])
                if m:
                    button, pressed_str = m.group(1), m.group(2)
                    pressed = pressed_str == "press"
                    return MouseEvent(event_type="click", x=x, y=y, button=button, pressed=pressed), j + 1
            elif tokens[j].startswith("<MOUSE_scroll_"):
                m = re.match(r"<MOUSE_scroll_(-?\d+)_(-?\d+)>", tokens[j])
                if m:
                    dx, dy = int(m.group(1)), int(m.group(2))
                    return MouseEvent(event_type="scroll", x=x, y=y, dx=dx, dy=dy), j + 1

        # Only move tokens present
        return MouseEvent(event_type="move", x=x, y=y), j


class EventProcessor:
    def __init__(self, config: EventProcessorConfig):
        self.config = config
        self.mouse_tokenizer = MouseTokenizer(config)

    def _tokenize_timestamp(self, timestamp: int) -> str:
        cfg = self.config
        clipped = min(max(timestamp, cfg.timestamp_min_ns), cfg.timestamp_max_ns)
        idx = int((clipped - cfg.timestamp_min_ns) // cfg.timestamp_interval_ns)
        return cfg.timestamp_token_format.format(idx=idx)

    def _detokenize_timestamp(self, token: str) -> int:
        m = re.match(r"<TIMESTAMP_(\d+)>", token)
        if not m:
            raise ValueError(f"Malformed timestamp token: {token}")
        idx = int(m.group(1))
        ts = self.config.timestamp_min_ns + idx * self.config.timestamp_interval_ns
        return ts

    def _tokenize_keyboard(self, ev: KeyboardEvent) -> str:
        cfg = self.config
        return cfg.keyboard_token_format.format(vk=ev.vk, pressed=ev.event_type)

    def _detokenize_keyboard(self, token: str) -> Optional[KeyboardEvent]:
        m = re.match(r"<KEYBOARD_(\d+)_(\w+)>", token)
        if m:
            vk = int(m.group(1))
            event_type = m.group(2)
            if event_type in ("press", "release"):
                return KeyboardEvent(event_type=event_type, vk=vk)
        return None

    def tokenize(self, events: List[Event], screen_size: Optional[Tuple[int, int]] = None) -> List[str]:
        result = []
        for event in events:
            tokens = [self._tokenize_timestamp(event.timestamp)]
            msg = event.msg
            if isinstance(msg, KeyboardEvent):
                tokens.append(self._tokenize_keyboard(msg))
            elif isinstance(msg, MouseEvent):
                tokens += self.mouse_tokenizer.to_tokens(msg, screen_size or self.config.screen_size)
            else:
                tokens.append("<UNKNOWN_EVENT>")
            result.append("".join(tokens))
        return result

    def detokenize(self, token_strs: List[str], screen_size: Optional[Tuple[int, int]] = None) -> List[Event]:
        events = []
        screen_size = screen_size if screen_size is not None else self.config.screen_size
        for s in token_strs:
            tokens = re.findall(r"<[^>]+>", s)
            if not tokens or not tokens[0].startswith("<TIMESTAMP_"):
                continue

            timestamp = self._detokenize_timestamp(tokens[0])
            if len(tokens) < 2:
                continue

            token = tokens[1]
            keyboard_ev = self._detokenize_keyboard(token)
            if keyboard_ev:
                events.append(Event(timestamp=timestamp, topic="keyboard", msg=keyboard_ev))
            elif token == "<MOUSE_move>":
                mouse_ev, _ = self.mouse_tokenizer.from_tokens(tokens, 1, screen_size)
                if mouse_ev:
                    events.append(Event(timestamp=timestamp, topic="mouse", msg=mouse_ev))
        return events


# --- Example usage ---

if __name__ == "__main__":
    processor = EventProcessor(EventProcessorConfig())
    events = [
        Event(timestamp=-50 * TimeUnits.MSECOND, topic="keyboard", msg=KeyboardEvent(event_type="press", vk=65)),
        Event(timestamp=-10 * TimeUnits.MSECOND, topic="mouse", msg=MouseEvent(event_type="move", x=10, y=20)),
        Event(
            timestamp=125 * TimeUnits.MSECOND,
            topic="mouse",
            msg=MouseEvent(event_type="click", x=11, y=21, button="left", pressed=True),
        ),
        Event(
            timestamp=126 * TimeUnits.MSECOND,
            topic="mouse",
            msg=MouseEvent(event_type="scroll", x=15, y=25, dx=0, dy=-1),
        ),
        Event(
            timestamp=500 * TimeUnits.MSECOND,
            topic="click",
            msg=MouseEvent(event_type="click", x=1504, y=1027, button="right", pressed=False),
        ),
    ]
    token_strs = processor.tokenize(events)
    print("Tokenized:")
    for t in token_strs:
        print(f"  {t}")

    print("\nDetokenized:")
    events_restored = processor.detokenize(token_strs)
    for e in events_restored:
        print(f"  {e}")

    print("\nComparing original with restored:")
    for i, (orig, restored) in enumerate(zip(events, events_restored)):
        print(f"Event {i}:")
        print(f"  Original:  {orig}")
        print(f"  Restored:  {restored}")
        if isinstance(orig.msg, MouseEvent) and isinstance(restored.msg, MouseEvent):
            orig_pos = (orig.msg.x, orig.msg.y)
            restored_pos = (restored.msg.x, restored.msg.y)
            print(f"  Position:  {orig_pos} -> {restored_pos}")
