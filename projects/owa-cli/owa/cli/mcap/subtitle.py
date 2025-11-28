"""Generate subtitle files (SRT/ASS) from MCAP recordings."""

from collections import namedtuple
from enum import Enum
from pathlib import Path

import typer
from tqdm import tqdm
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapReader
from owa.env.desktop.constants import VK
from owa.msgs.desktop.mouse import RawMouseEvent

# Constants
MIN_DURATION_NS = 500_000_000  # 500ms
CompletedEvent = namedtuple("CompletedEvent", ["timestamp", "content"])

BUTTON_PRESS_FLAGS = {
    RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_DOWN: "left",
    RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_DOWN: "right",
    RawMouseEvent.ButtonFlags.RI_MOUSE_MIDDLE_BUTTON_DOWN: "middle",
}
BUTTON_RELEASE_FLAGS = {
    RawMouseEvent.ButtonFlags.RI_MOUSE_LEFT_BUTTON_UP: "left",
    RawMouseEvent.ButtonFlags.RI_MOUSE_RIGHT_BUTTON_UP: "right",
    RawMouseEvent.ButtonFlags.RI_MOUSE_MIDDLE_BUTTON_UP: "middle",
}

VK_TO_LABEL = {
    VK.ESCAPE: "ESC",
    VK.BACK: "BACK",
    VK.TAB: "TAB",
    VK.RETURN: "ENTER",
    VK.CAPITAL: "CAPS",
    VK.LSHIFT: "SHIFT",
    VK.RSHIFT: "SHIFT",
    VK.LCONTROL: "CTRL",
    VK.RCONTROL: "CTRL",
    VK.LMENU: "ALT",
    VK.RMENU: "ALT",
    VK.LWIN: "WIN",
    VK.RWIN: "WIN",
    VK.SPACE: "SPACE",
    VK.UP: "↑",
    VK.DOWN: "↓",
    VK.LEFT: "←",
    VK.RIGHT: "→",
    **{getattr(VK, f"F{i}"): f"F{i}" for i in range(1, 13)},
}

CURSOR_SAMPLE_INTERVAL_NS = 100_000_000  # 100ms

ASS_HEADER = """[Script Info]
Title: OWA Keyboard/Mouse Overlay
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: KeyPressed,Arial,24,&H00FFFFFF,&H000000FF,&H0050B0AB,&H8050B0AB,1,0,0,0,100,100,0,0,3,2,0,7,10,10,10,1
Style: Cursor,Arial,28,&H0000FFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


class SubtitleFormat(str, Enum):
    SRT = "srt"
    ASS = "ass"


class KeyState:
    """Tracks state of a single key."""

    def __init__(self, vk: int):
        self.vk = vk
        self.is_pressed = False
        self.press_time = None
        self.release_time = None

    def press(self, timestamp: int) -> bool:
        if not self.is_pressed:
            self.is_pressed = True
            self.press_time = timestamp
            self.release_time = None
            return True
        return False

    def release(self, timestamp: int) -> bool:
        if self.is_pressed:
            self.is_pressed = False
            self.release_time = timestamp
            return True
        return False

    def get_duration(self) -> tuple[int, int]:
        if self.press_time is None:
            return (0, 0)
        end = self.press_time + max((self.release_time or self.press_time) - self.press_time, MIN_DURATION_NS)
        return (self.press_time, end)


class KeyStateManager:
    """Manages keyboard state and generates subtitle events."""

    def __init__(self):
        self.states = {}  # vk -> KeyState
        self.pending = []  # (KeyState, label)
        self.completed = []  # (start, end, label)

    def handle_event(self, event_type: str, vk: int, timestamp: int):
        if vk not in self.states:
            self.states[vk] = KeyState(vk)
        state = self.states[vk]

        if event_type == "press" and state.press(timestamp):
            label = get_key_label(vk)
            self.pending.append((state, label))
        elif event_type == "release" and state.release(timestamp):
            for i, (s, label) in enumerate(self.pending):
                if s is state:
                    self.completed.append((*state.get_duration(), label))
                    self.pending.pop(i)
                    break

    def finalize(self):
        for state, label in self.pending:
            self.completed.append((*state.get_duration(), label))
        self.pending.clear()


def get_key_label(vk: int) -> str:
    try:
        vk_enum = VK(vk)
        if vk_enum in VK_TO_LABEL:
            return VK_TO_LABEL[vk_enum]
        name = vk_enum.name
        return name[4:] if name.startswith("KEY_") else name
    except ValueError:
        return f"?{vk}"


def format_srt_time(ns: int) -> str:
    s = ns / 1e9
    h, m, sec, ms = int(s // 3600), int((s % 3600) // 60), int(s % 60), int((s * 1000) % 1000)
    return f"{h:02}:{m:02}:{sec:02},{ms:03}"


def format_ass_time(ns: int) -> str:
    s = ns / 1e9
    return f"{int(s // 3600)}:{int((s % 3600) // 60):02}:{s % 60:05.2f}"


def subtitle(
    input_file: Annotated[Path, typer.Argument(help="Input MCAP file")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output file path")] = None,
    format: Annotated[SubtitleFormat | None, typer.Option("--format", "-f", help="Output format")] = None,
    width: Annotated[int, typer.Option(help="Video width (ASS only)")] = 1920,
    height: Annotated[int, typer.Option(help="Video height (ASS only)")] = 1080,
):
    """Generate subtitle file from MCAP recording for playback verification."""
    # Auto-detect format from output extension, default to ASS
    if format is None:
        if output and output.suffix.lower() in (".srt", ".ass"):
            format = SubtitleFormat(output.suffix.lower()[1:])
        else:
            format = SubtitleFormat.ASS
    output = output or input_file.with_suffix(f".{format.value}")

    with OWAMcapReader(input_file) as reader:
        # Get start time from first screen message
        start_time = None
        for msg in reader.iter_messages(topics=["screen"]):
            start_time = msg.timestamp
            if hasattr(msg.decoded, "media_ref") and msg.decoded.media_ref:
                pts_ns = getattr(msg.decoded.media_ref, "pts_ns", None)
                if pts_ns is not None:
                    start_time -= pts_ns
            break
        if start_time is None:
            typer.echo("No screen messages found.")
            raise typer.Exit(1)

        # Read all messages
        all_messages = list(
            tqdm(
                reader.iter_messages(topics=["keyboard", "mouse/raw", "mouse"], start_time=start_time),
                desc="Reading messages",
                unit="msg",
            )
        )

        # Process events
        key_manager = KeyStateManager()
        mouse_events = []  # (timestamp, button, is_press)
        mouse_positions = {}  # timestamp -> (x, y)
        abs_x, abs_y = width // 2, height // 2

        for msg in tqdm(all_messages, desc="Processing events", unit="msg"):
            if msg.topic == "keyboard":
                if hasattr(msg.decoded, "event_type") and hasattr(msg.decoded, "vk"):
                    key_manager.handle_event(msg.decoded.event_type, msg.decoded.vk, msg.timestamp)

            elif msg.topic == "mouse/raw":
                if hasattr(msg.decoded, "button_flags"):
                    flags = msg.decoded.button_flags
                    for flag, btn in BUTTON_PRESS_FLAGS.items():
                        if flags & flag:
                            mouse_events.append((msg.timestamp, btn, True))
                    for flag, btn in BUTTON_RELEASE_FLAGS.items():
                        if flags & flag:
                            mouse_events.append((msg.timestamp, btn, False))
                if hasattr(msg.decoded, "last_x") and hasattr(msg.decoded, "last_y"):
                    abs_x = max(0, min(width - 1, abs_x + msg.decoded.last_x))
                    abs_y = max(0, min(height - 1, abs_y + msg.decoded.last_y))
                    mouse_positions[msg.timestamp] = (abs_x, abs_y)

            elif msg.topic == "mouse":
                if hasattr(msg.decoded, "x") and hasattr(msg.decoded, "y"):
                    abs_x, abs_y = msg.decoded.x, msg.decoded.y
                    mouse_positions[msg.timestamp] = (abs_x, abs_y)

        key_manager.finalize()

    # Generate output based on format
    if format == SubtitleFormat.SRT:
        content = generate_srt(start_time, key_manager.completed, mouse_events)
    else:
        content = generate_ass(start_time, key_manager.completed, mouse_events, mouse_positions, width, height)

    output.write_text(content, encoding="utf-8")
    typer.echo(f"Subtitle saved: {output}")
    if format == SubtitleFormat.ASS:
        typer.echo(f"Play with: mpv {input_file.with_suffix('.mkv')} --sub-file={output}")


def generate_srt(start_time: int, keyboard_events: list, mouse_events: list) -> str:
    """Generate SRT format subtitle."""
    events = []

    # Add keyboard events
    for press_time, release_time, label in keyboard_events:
        start = format_srt_time(press_time - start_time)
        end = format_srt_time(release_time - start_time)
        events.append(CompletedEvent(press_time, f"{start} --> {end}\n[keyboard] press {label}"))

    # Process mouse click events
    mouse_state = {}
    for ts, button, is_press in sorted(mouse_events):
        if is_press:
            mouse_state[button] = ts
        elif button in mouse_state:
            press_ts = mouse_state.pop(button)
            start = format_srt_time(press_ts - start_time)
            end_ts = press_ts + max(ts - press_ts, MIN_DURATION_NS)
            end = format_srt_time(end_ts - start_time)
            events.append(CompletedEvent(press_ts, f"{start} --> {end}\n[mouse] {button} click"))

    events.sort(key=lambda e: e.timestamp)
    return "\n".join(f"{i}\n{e.content}\n" for i, e in enumerate(events, 1))


def generate_ass(
    start_time: int, keyboard_events: list, mouse_events: list, mouse_positions: dict, width: int, height: int
) -> str:
    """Generate ASS format subtitle with cursor movement."""
    lines = [ASS_HEADER.format(width=width, height=height)]
    pos_y = height - 50

    # Build timeline of active keys/buttons
    all_events = []
    for press_time, release_time, label in keyboard_events:
        all_events.append((press_time, "key", label, True))
        all_events.append((release_time, "key", label, False))

    mouse_state = {}
    for ts, button, is_press in sorted(mouse_events):
        if is_press:
            mouse_state[button] = ts
        elif button in mouse_state:
            press_ts = mouse_state.pop(button)
            all_events.append((press_ts, "mouse", button.upper(), True))
            all_events.append((ts, "mouse", button.upper(), False))

    all_events.sort(key=lambda x: (x[0], not x[3]))

    # Generate state-based subtitle events
    active_keys, active_mouse = set(), set()
    last_time = None

    for time_ns, evt_type, label, is_start in all_events:
        if last_time is not None and time_ns != last_time and (active_keys or active_mouse):
            t1, t2 = format_ass_time(last_time - start_time), format_ass_time(time_ns - start_time)
            parts = list(active_keys) + [f"[{m}]" for m in active_mouse]
            lines.append(f"Dialogue: 0,{t1},{t2},KeyPressed,,0,0,0,,{{\\pos(20,{pos_y})}}{' + '.join(sorted(parts))}")

        if evt_type == "key":
            (active_keys.add if is_start else active_keys.discard)(label)
        else:
            (active_mouse.add if is_start else active_mouse.discard)(label)
        last_time = time_ns

    # Emit final state
    if last_time and (active_keys or active_mouse):
        t1, t2 = format_ass_time(last_time - start_time), format_ass_time(last_time - start_time + MIN_DURATION_NS)
        parts = list(active_keys) + [f"[{m}]" for m in active_mouse]
        lines.append(f"Dialogue: 0,{t1},{t2},KeyPressed,,0,0,0,,{{\\pos(20,{pos_y})}}{' + '.join(sorted(parts))}")

    # Generate cursor movement
    if mouse_positions:
        sorted_pos = sorted(mouse_positions.items())
        sampled = []
        next_time = sorted_pos[0][0]
        for ts, pos in sorted_pos:
            if ts >= next_time:
                sampled.append((ts, pos))
                next_time = ts + CURSOR_SAMPLE_INTERVAL_NS

        for i in range(len(sampled) - 1):
            t1_ns, (x1, y1) = sampled[i]
            t2_ns, (x2, y2) = sampled[i + 1]
            t1, t2 = format_ass_time(t1_ns - start_time), format_ass_time(t2_ns - start_time)
            lines.append(f"Dialogue: 1,{t1},{t2},Cursor,,0,0,0,,{{\\move({int(x1)},{int(y1)},{int(x2)},{int(y2)})}}●")

    return "\n".join(lines)


if __name__ == "__main__":
    typer.run(subtitle)
