import re
from typing import List, Optional, Sequence, Tuple

from pydantic import Field
from pydantic.dataclasses import dataclass

from owa.agent.core import Event
from owa.core.time import TimeUnits
from owa.env.desktop.msg import KeyboardEvent, MouseButton, MouseEvent
from owa.env.gst.msg import ScreenEmitted


@dataclass
class EventProcessorConfig:
    # Timestamp config
    timestamp_min_ns: int = -2 * TimeUnits.SECOND
    timestamp_max_ns: int = 2 * TimeUnits.SECOND
    timestamp_interval_ns: int = 20 * TimeUnits.MSECOND  # 20ms == 50fps
    timestamp_token_format: str = "<TIMESTAMP_{idx}>"
    # Each timestamp interval encodes a moment in time within the allowed window.

    # Keyboard config
    keyboard_vk_count: int = 256
    keyboard_token_format: str = "<KEYBOARD_{vk}_{pressed}>"
    # e.g. <KEYBOARD_65_press>, <KEYBOARD_65_release>

    # Mouse config
    mouse_move_bins: list[int] = Field(default_factory=lambda: [16, 16, 16])
    screen_size: tuple[int, int] = (1920, 1080)
    mouse_move_token_format: str = "<MOUSE_move_{level}_{idx_x}_{idx_y}>"
    mouse_click_token_format: str = "<MOUSE_click_{button}_{pressed}>"
    mouse_scroll_token_format: str = "<MOUSE_scroll_{dx}_{dy}>"
    # For move, we utilize residual-quantize; e.g., [16,16,16] means three levels, 16 bins per level. This fits 4096 pixel wide, which is wider than FHD(1920)
    # See MouseProcessor for token scheme.

    screen_token: str = "<image>"

    @property
    def timestamp_tokens(self) -> list[str]:
        """
        All unique timestamp tokens over the configured window and interval.

        Returns:
            list[str]: Each of the tokens in the interval, with index.
        """
        count = ((self.timestamp_max_ns - self.timestamp_min_ns) // self.timestamp_interval_ns) + 1
        return [self.timestamp_token_format.format(idx=idx) for idx in range(count)]

    @property
    def keyboard_tokens(self) -> list[str]:
        """
        All unique keyboard event tokens (one for press, one for release per key).

        Returns:
            list[str]: Each in <KEYBOARD_{vk}_{press|release}> form.
        """
        tokens = []
        for vk in range(self.keyboard_vk_count):
            tokens.append(self.keyboard_token_format.format(vk=vk, pressed="press"))
            tokens.append(self.keyboard_token_format.format(vk=vk, pressed="release"))
        return tokens

    @property
    def mouse_move_tokens(self) -> list[str]:
        """
        All unique mouse move tokens (token types) for all quantization levels.

        For each quantization level, all (idx_x,idx_y) pairs are tokenized; sum over all levels.
        Special tokens for <MOUSE_move> and </MOUSE_move> not included.

        Returns:
            list[str]: Each as <MOUSE_move_{level}_{idx_x}_{idx_y}>
        Example:
            mouse_move_bins = [256,256,256] => 256*256*3 = 196,608 tokens
        """
        tokens = []
        for level, bins in enumerate(self.mouse_move_bins):
            for idx_x in range(bins):
                for idx_y in range(bins):
                    tokens.append(self.mouse_move_token_format.format(level=level, idx_x=idx_x, idx_y=idx_y))
        return tokens

    @property
    def mouse_click_tokens(self) -> list[str]:
        """
        All unique mouse click tokens, for all buttons and both press/release states.

        Returns:
            list[str]: Each as <MOUSE_click_{button}_{press|release}>
        """
        tokens = []
        buttons = MouseButton.__args__
        for button in buttons:
            for pressed in ["press", "release"]:
                tokens.append(self.mouse_click_token_format.format(button=button, pressed=pressed))
        return tokens

    @property
    def mouse_scroll_tokens(self) -> list[str]:
        """
        All mouse scroll tokens, i.e. possible (dx, dy) directions.
        In practice, token set is limited (e.g., dx/dy in [-1,0,1]).

        Returns:
            list[str]: Each as <MOUSE_scroll_{dx}_{dy}>
        """
        tokens = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                tokens.append(self.mouse_scroll_token_format.format(dx=dx, dy=dy))
        return tokens

    @property
    def screen_tokens(self) -> list[str]:
        """
        All unique screen marker tokens.
        Usually just 1: "<image>".

        Returns:
            list[str]: [screen_token]
        """
        return [self.screen_token]

    def summary(self) -> str:
        """
        Returns a string summary of unique token counts for each event/token type,
        under the current configuration.
        """
        return (
            f"Timestamp tokens: {len(self.timestamp_tokens)}\n"
            f"Keyboard tokens: {len(self.keyboard_tokens)}\n"
            f"Mouse move tokens: {len(self.mouse_move_tokens)}\n"
            f"Mouse click tokens: {len(self.mouse_click_tokens)}\n"
            f"Mouse scroll tokens: {len(self.mouse_scroll_tokens)}\n"
            f"Screen tokens: {len(self.screen_tokens)}"
        )


def limit(x, low=0, high=1):
    """
    Limit x to the range [low, high].
    """
    return max(low, min(x, high))


class MouseProcessor:
    def __init__(self, config: EventProcessorConfig):
        self.config = config

    def move_tokens(self, event: MouseEvent, screen_size: Optional[Tuple[int, int]] = None) -> List[str]:
        """
        Quantize (x, y) with residual quantization (multi-level, jointly as a pair).
        Returns: [<MOUSE_move>, <move_level_0>, ... <move_level_n>, </MOUSE_move>]
        """
        if screen_size is None:
            screen_size = self.config.screen_size
        x, y = event.x, event.y
        fx = limit(x / screen_size[0])
        fy = limit(y / screen_size[1])

        # Jointly quantize the pair (x, y) repeatedly at each level
        vx, vy = fx, fy
        tokens = []
        for i, nbins in enumerate(self.config.mouse_move_bins):
            # Using floor instead of round for better accuracy
            idx_x = int(vx * nbins)
            idx_y = int(vy * nbins)
            tokens.append(self.config.mouse_move_token_format.format(level=i, idx_x=idx_x, idx_y=idx_y))
            # Calculate residuals for next level and normalize into [0, 1]
            vx = vx * nbins - idx_x
            vy = vy * nbins - idx_y
            # vx and vy should already be in [0, 1) due to using floor
        return ["<MOUSE_move>"] + tokens + ["</MOUSE_move>"]

    def move_inv(self, tokens: Sequence[str], screen_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Inverse of move_tokens - reconstruct (x, y).
        """
        if screen_size is None:
            screen_size = self.config.screen_size

        # Extract indices from tokens
        indices = []
        for token in tokens:
            m = re.match(r"<MOUSE_move_(\d+)_(\d+)_(\d+)>", token)
            if m:
                level, idx_x, idx_y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                indices.append((level, idx_x, idx_y))

        # Sort by level to ensure correct order
        indices.sort(key=lambda x: x[0])

        if len(indices) != len(self.config.mouse_move_bins):
            raise ValueError(
                f"Expected {len(self.config.mouse_move_bins)} levels, but got {len(indices)} levels in tokens."
            )

        fx = fy = 0
        # Apply refinements from subsequent levels
        for i in reversed(range(len(indices))):
            level, idx_x, idx_y = indices[i]
            nbins = self.config.mouse_move_bins[i]

            # Calculate the residuals
            fx = (fx + idx_x) / nbins
            fy = (fy + idx_y) / nbins

        # Convert normalized coordinates to pixel coordinates
        pix_x = int(round(fx * (screen_size[0] - 1)))
        pix_y = int(round(fy * (screen_size[1] - 1)))

        return pix_x, pix_y

    def click_token(self, event: MouseEvent) -> str:
        """
        Compose a click token from MouseEvent.
        """
        button = event.button or "unknown"
        press_str = "press" if bool(event.pressed) else "release"
        return self.config.mouse_click_token_format.format(button=button, pressed=press_str)

    def scroll_token(self, event: MouseEvent) -> str:
        """
        Compose a scroll token from MouseEvent.
        """
        dx = event.dx if event.dx is not None else 0
        dy = event.dy if event.dy is not None else 0
        return self.config.mouse_scroll_token_format.format(dx=dx, dy=dy)

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
            screen_size = self.config.screen_size

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
    def __init__(self, config: EventProcessorConfig = EventProcessorConfig()):
        self.config = config
        self.mouse_processor = MouseProcessor(config)

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

    def tokenize(self, events: List[Event], now: int = 0, screen_size: Optional[Tuple[int, int]] = None) -> List[str]:
        result = []
        for event in events:
            tokens = [self._tokenize_timestamp(event.timestamp - now)]
            msg = event.msg
            if isinstance(msg, KeyboardEvent):
                tokens.append(self._tokenize_keyboard(msg))
            elif isinstance(msg, MouseEvent):
                tokens += self.mouse_processor.to_tokens(msg, screen_size or self.config.screen_size)
            elif isinstance(msg, ScreenEmitted):
                tokens += [self.config.screen_token]
            else:
                tokens.append("<UNKNOWN_EVENT>")
            result.append("".join(tokens))
        return result

    def detokenize(self, token_strs: List[str], screen_size: Optional[Tuple[int, int]] = None) -> List[Event]:
        """
        Accepts either a list of per-event token strings, or a single string containing
        a concatenation of multiple tokenized events (e.g. "".join(token_strs)).
        Returns a list of restored events.
        """
        events = []
        screen_size = screen_size if screen_size is not None else self.config.screen_size

        # If input is a single long string, split at every <TIMESTAMP_\d+>
        if len(token_strs) == 1 and token_strs[0].count("<TIMESTAMP_") > 1:
            s = token_strs[0]
            # Find all indices of <TIMESTAMP_\d+>
            matches = list(re.finditer(r"<TIMESTAMP_\d+>", s))
            event_strings = []
            for i, m in enumerate(matches):
                start = m.start()
                # event goes from start to start of *next* match, or to end of string
                end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
                event_strings.append(s[start:end])
        else:
            event_strings = token_strs  # already split

        for s in event_strings:
            # Extract all tokens (e.g. <SOMETHING>)
            tokens = re.findall(r"<[^>]+>", s)
            if not tokens or not tokens[0].startswith("<TIMESTAMP_"):
                continue

            timestamp = self._detokenize_timestamp(tokens[0])
            if len(tokens) < 2:
                events.append(Event(timestamp=timestamp, topic="unknown", msg=None))
                continue

            token = tokens[1]
            if token == "<UNKNOWN_EVENT>":
                events.append(Event(timestamp=timestamp, topic="unknown", msg=None))
                continue

            keyboard_ev = self._detokenize_keyboard(token)
            if keyboard_ev:
                events.append(Event(timestamp=timestamp, topic="keyboard", msg=keyboard_ev))
            elif token == "<MOUSE_move>":
                mouse_ev, _ = self.mouse_processor.from_tokens(tokens, 1, screen_size)
                if mouse_ev:
                    events.append(Event(timestamp=timestamp, topic="mouse", msg=mouse_ev))
            elif token == self.config.screen_token:
                # This ScreenEmitted is always reconstructed as path="", pts=0
                events.append(Event(timestamp=timestamp, topic="screen", msg=ScreenEmitted(path="", pts=0)))

        return events


# --- Example usage ---

if __name__ == "__main__":
    config = EventProcessorConfig()
    print(config.summary())
    processor = EventProcessor(config=config)
    events = [
        Event(timestamp=-50 * TimeUnits.MSECOND, topic="keyboard", msg=KeyboardEvent(event_type="press", vk=65)),
        Event(timestamp=-10 * TimeUnits.MSECOND, topic="mouse", msg=MouseEvent(event_type="move", x=10, y=20)),
        Event(timestamp=0, topic="screen", msg=ScreenEmitted(path="output.mkv", pts=123 * TimeUnits.MSECOND)),
        Event(
            timestamp=125 * TimeUnits.MSECOND,
            topic="mouse",
            msg=MouseEvent(event_type="click", x=11, y=21, button="left", pressed=True),
        ),
        Event(
            timestamp=126 * TimeUnits.MSECOND,
            topic="mouse",
            msg=MouseEvent(event_type="scroll", x=15, y=889, dx=0, dy=-1),
        ),
        Event(
            timestamp=500 * TimeUnits.MSECOND,
            topic="click",
            msg=MouseEvent(event_type="click", x=1504, y=1027, button="right", pressed=False),
        ),
    ]
    # events = [Event(timestamp=-10 * TimeUnits.MSECOND, topic="mouse", msg=MouseEvent(event_type="move", x=10, y=20))]
    token_strs = processor.tokenize(events)
    print("Tokenized:")
    for t in token_strs:
        print(f"  {t}")
    print("".join(token_strs))

    print("\nDetokenized:")
    # token_strs = ["".join(token_strs)]
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
