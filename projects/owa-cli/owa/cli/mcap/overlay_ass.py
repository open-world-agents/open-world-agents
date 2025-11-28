"""Generate ASS subtitle overlay for keyboard/mouse visualization without video re-encoding."""

from pathlib import Path

import typer
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapReader
from owa.cli.mcap.convert import KeyStateManager
from owa.env.desktop.constants import VK
from owa.msgs.desktop.mouse import RawMouseEvent

# VK code to display label mapping
VK_TO_LABEL = {
    VK.ESCAPE: "ESC",
    VK.F1: "F1",
    VK.F2: "F2",
    VK.F3: "F3",
    VK.F4: "F4",
    VK.F5: "F5",
    VK.F6: "F6",
    VK.F7: "F7",
    VK.F8: "F8",
    VK.F9: "F9",
    VK.F10: "F10",
    VK.F11: "F11",
    VK.F12: "F12",
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
}

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

ASS_HEADER = """[Script Info]
Title: OWA Keyboard/Mouse Overlay
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Key,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,2,0,7,10,10,10,1
Style: KeyPressed,Arial,24,&H00FFFFFF,&H000000FF,&H0050B0AB,&H8050B0AB,1,0,0,0,100,100,0,0,3,2,0,7,10,10,10,1
Style: Mouse,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,2,0,7,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def format_ass_time(ns: int) -> str:
    """Convert nanoseconds to ASS time format (H:MM:SS.cc)."""
    total_seconds = ns / 1e9
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02}:{seconds:05.2f}"


def get_key_label(vk: int) -> str:
    """Get display label for a virtual key code."""
    try:
        vk_enum = VK(vk)
        if vk_enum in VK_TO_LABEL:
            return VK_TO_LABEL[vk_enum]
        name = vk_enum.name
        if name.startswith("KEY_"):
            return name[4:]
        if name.startswith("OEM_"):
            return name[4:]
        return name
    except ValueError:
        return f"?{vk}"


def overlay_ass(
    input_file: Annotated[Path, typer.Argument(help="Input MCAP file")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output ASS file (default: <input>.ass)")
    ] = None,
    width: Annotated[int, typer.Option(help="Video width for positioning")] = 1920,
    height: Annotated[int, typer.Option(help="Video height for positioning")] = 1080,
):
    """Generate ASS subtitle with keyboard/mouse overlay. Play with: mpv video.mkv --sub-file=overlay.ass"""
    output = output or input_file.with_suffix(".ass")

    with OWAMcapReader(input_file) as reader:
        # Get recording start time from first screen message
        start_time = None
        for mcap_msg in reader.iter_messages(topics=["screen"]):
            start_time = mcap_msg.timestamp
            if hasattr(mcap_msg.decoded, "media_ref") and mcap_msg.decoded.media_ref:
                pts_ns = getattr(mcap_msg.decoded.media_ref, "pts_ns", None)
                if pts_ns is not None:
                    start_time -= pts_ns
            break

        if start_time is None:
            typer.echo("No screen messages found.")
            raise typer.Exit(1)

        # Collect keyboard and mouse events
        key_manager = KeyStateManager()
        mouse_events = []  # (timestamp, button, is_press)

        for mcap_msg in reader.iter_messages(topics=["keyboard", "mouse/raw"], start_time=start_time):
            if mcap_msg.topic == "keyboard":
                if hasattr(mcap_msg.decoded, "event_type") and hasattr(mcap_msg.decoded, "vk"):
                    key_manager.handle_key_event(mcap_msg.decoded.event_type, mcap_msg.decoded.vk, mcap_msg.timestamp)
            elif mcap_msg.topic == "mouse/raw":
                if hasattr(mcap_msg.decoded, "button_flags"):
                    flags = mcap_msg.decoded.button_flags
                    for flag, btn in BUTTON_PRESS_FLAGS.items():
                        if flags & flag:
                            mouse_events.append((mcap_msg.timestamp, btn, True))
                    for flag, btn in BUTTON_RELEASE_FLAGS.items():
                        if flags & flag:
                            mouse_events.append((mcap_msg.timestamp, btn, False))

        key_manager.finalize_remaining_subtitles()

    # Build timeline of active keys/buttons at each moment
    # Collect all events with their times
    all_events = []  # (time_ns, event_type, key_or_button, is_start)

    for press_time, release_time, message in key_manager.get_completed_subtitles():
        if not message.startswith("press "):
            continue
        vk_name = message[6:]
        try:
            vk = VK[vk_name].value
            label = get_key_label(vk)
        except KeyError:
            continue
        all_events.append((press_time, "key", label, True))
        all_events.append((release_time, "key", label, False))

    # Process mouse events into press/release pairs
    mouse_state = {}
    for ts, button, is_press in sorted(mouse_events):
        if is_press:
            mouse_state[button] = ts
        elif button in mouse_state:
            press_ts = mouse_state.pop(button)
            all_events.append((press_ts, "mouse", button.upper(), True))
            all_events.append((ts, "mouse", button.upper(), False))

    # Sort all events by time
    all_events.sort(key=lambda x: (x[0], not x[3]))  # End events before start events at same time

    # Generate state changes and create dialogue lines
    lines = [ASS_HEADER.format(width=width, height=height)]
    pos_y = height - 50

    active_keys = set()
    active_mouse = set()
    last_change_time = None

    def emit_state(end_time_ns):
        """Emit a dialogue line for the current state."""
        if last_change_time is None:
            return
        if not active_keys and not active_mouse:
            return
        t1 = format_ass_time(last_change_time - start_time)
        t2 = format_ass_time(end_time_ns - start_time)
        # Combine all active keys and mouse buttons
        parts = list(active_keys) + [f"[{m}]" for m in active_mouse]
        text = " + ".join(sorted(parts))
        lines.append(f"Dialogue: 0,{t1},{t2},KeyPressed,,0,0,0,,{{\\pos(20,{pos_y})}}{text}")

    for time_ns, evt_type, label, is_start in all_events:
        if last_change_time is not None and time_ns != last_change_time:
            emit_state(time_ns)

        if evt_type == "key":
            if is_start:
                active_keys.add(label)
            else:
                active_keys.discard(label)
        else:  # mouse
            if is_start:
                active_mouse.add(label)
            else:
                active_mouse.discard(label)

        last_change_time = time_ns

    # Emit final state if any
    if last_change_time is not None and (active_keys or active_mouse):
        emit_state(last_change_time + 500_000_000)  # 500ms after last event

    output.write_text("\n".join(lines), encoding="utf-8")
    typer.echo(f"ASS overlay saved: {output}")
    typer.echo(f"Play with: mpv {input_file.with_suffix('.mkv')} --sub-file={output}")


if __name__ == "__main__":
    typer.run(overlay_ass)
