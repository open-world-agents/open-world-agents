from collections import namedtuple
from pathlib import Path

import typer
from tqdm import tqdm
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapReader
from owa.env.desktop.constants import VK
from owa.msgs.desktop.mouse import RawMouseEvent

# Constants
MIN_MOUSE_CLICK_DURATION_NS = 500_000_000  # 500ms in nanoseconds
MIN_KEY_PRESS_DURATION_NS = 500_000_000  # 500ms in nanoseconds


class KeyState:
    """Represents the state of a single key."""

    def __init__(self, vk: int):
        self.vk = vk
        self.is_pressed = False
        self.first_press_timestamp = None
        self.last_press_timestamp = None
        self.release_timestamp = None
        self.press_count = 0

    def press(self, timestamp: int) -> bool:
        """Handle key press. Returns True if this is the first press."""
        if not self.is_pressed:
            self.is_pressed = True
            self.first_press_timestamp = timestamp
            self.last_press_timestamp = timestamp
            self.press_count = 1
            self.release_timestamp = None
            return True
        self.last_press_timestamp = timestamp
        self.press_count += 1
        return False

    def release(self, timestamp: int) -> bool:
        """Handle key release. Returns True if key was pressed."""
        if self.is_pressed:
            self.is_pressed = False
            self.release_timestamp = timestamp
            return True
        return False

    def get_subtitle_duration(self) -> tuple[int, int]:
        """Get start and end timestamps for the subtitle."""
        if self.first_press_timestamp is None:
            return (0, 0)

        start_time = self.first_press_timestamp
        if self.release_timestamp is not None:
            actual_duration = self.release_timestamp - self.first_press_timestamp
            end_time = self.first_press_timestamp + max(actual_duration, MIN_KEY_PRESS_DURATION_NS)
        else:
            end_time = self.first_press_timestamp + MIN_KEY_PRESS_DURATION_NS

        return (start_time, end_time)


class KeyStateManager:
    """Manages the state of all keyboard keys."""

    def __init__(self):
        self.key_states = {}  # vk -> KeyState
        self.pending_subtitles = []  # (KeyState, message_content)
        self.completed_subtitles = []  # (start_time, end_time, message_content)

    def handle_key_event(self, event_type: str, vk: int, timestamp: int) -> None:
        """Handle a keyboard event (press or release)."""
        if vk not in self.key_states:
            self.key_states[vk] = KeyState(vk)

        key_state = self.key_states[vk]

        if event_type == "press":
            if key_state.press(timestamp):
                try:
                    key_name = VK(vk).name
                except ValueError:
                    key_name = f"VK_{vk}"
                self.pending_subtitles.append((key_state, f"press {key_name}"))

        elif event_type == "release":
            if key_state.release(timestamp):
                for i, (pending_key_state, message_content) in enumerate(self.pending_subtitles):
                    if pending_key_state is key_state:
                        start_time, end_time = key_state.get_subtitle_duration()
                        self.completed_subtitles.append((start_time, end_time, message_content))
                        self.pending_subtitles.pop(i)
                        break

    def finalize_remaining_subtitles(self) -> None:
        """Finalize pending subtitles for keys that were never released."""
        for key_state, message_content in self.pending_subtitles:
            start_time, end_time = key_state.get_subtitle_duration()
            self.completed_subtitles.append((start_time, end_time, message_content))
        self.pending_subtitles.clear()

    def get_completed_subtitles(self) -> list[tuple[int, int, str]]:
        """Get all completed subtitles."""
        return self.completed_subtitles.copy()


def format_timestamp(timestamp_ns: int) -> str:
    """Convert nanosecond timestamp to SRT timestamp format (HH:MM:SS,mmm)."""
    timestamp_s = timestamp_ns / 1e9
    hours = int(timestamp_s // 3600)
    minutes = int((timestamp_s % 3600) // 60)
    seconds = int(timestamp_s % 60)
    milliseconds = int((timestamp_s * 1000) % 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


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

CompletedEvent = namedtuple("CompletedEvent", ["timestamp", "content"])


def convert(
    input_file: Annotated[Path, typer.Argument(help="Input MCAP file")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output SRT file (default: <input>.srt)")
    ] = None,
    topics: Annotated[list[str], typer.Option(help="MCAP topics to process")] = ["mouse/raw", "mouse", "keyboard"],
):
    """Convert MCAP file to SRT subtitle file for playback verification."""
    output = output or input_file.with_suffix(".srt")
    subtitles = []

    with OWAMcapReader(input_file) as reader:
        # Get start time from first screen message
        start_time = None
        for mcap_msg in reader.iter_messages(topics=["screen"]):
            start_time = mcap_msg.timestamp
            if (
                hasattr(mcap_msg.decoded, "media_ref")
                and mcap_msg.decoded.media_ref
                and hasattr(mcap_msg.decoded.media_ref, "pts_ns")
                and mcap_msg.decoded.media_ref.pts_ns is not None
            ):
                start_time -= mcap_msg.decoded.media_ref.pts_ns
            else:
                typer.echo("Warning: No pts_ns found. Subtitle timing may be off.")
            break
        else:
            typer.echo("No screen messages found.")
            raise typer.Exit()

        all_messages = list(
            tqdm(
                reader.iter_messages(topics=topics, start_time=start_time),
                desc="Reading messages",
                unit="msg",
            )
        )
        mouse_button_states = {}  # button_name -> press_timestamp
        pending_mouse_events = []  # (press_timestamp, button_name, message_content)
        key_state_manager = KeyStateManager()
        completed_events: list[CompletedEvent] = []

        def handle_mouse_press(button_name: str, timestamp: int):
            mouse_button_states[button_name] = timestamp
            pending_mouse_events.append((timestamp, button_name, f"{button_name} click"))

        def handle_mouse_release(button_name: str, timestamp: int):
            if button_name not in mouse_button_states:
                return
            press_timestamp = mouse_button_states.pop(button_name)
            for i, (press_ts, btn_name, msg_content) in enumerate(pending_mouse_events):
                if press_ts == press_timestamp and btn_name == button_name:
                    start = format_timestamp(press_timestamp - start_time)
                    end_timestamp = press_timestamp + max(timestamp - press_timestamp, MIN_MOUSE_CLICK_DURATION_NS)
                    end = format_timestamp(end_timestamp - start_time)
                    completed_events.append(
                        CompletedEvent(press_timestamp, f"{start} --> {end}\n[mouse] {msg_content}")
                    )
                    pending_mouse_events.pop(i)
                    break

        for mcap_msg in tqdm(all_messages, desc="Processing events", unit="msg"):
            if mcap_msg.topic == "mouse/raw":
                if hasattr(mcap_msg.decoded, "button_flags"):
                    button_flags = mcap_msg.decoded.button_flags
                    for flag, button_name in BUTTON_PRESS_FLAGS.items():
                        if button_flags & flag:
                            handle_mouse_press(button_name, mcap_msg.timestamp)
                            break
                    for flag, button_name in BUTTON_RELEASE_FLAGS.items():
                        if button_flags & flag:
                            handle_mouse_release(button_name, mcap_msg.timestamp)
                            break

            elif mcap_msg.topic == "mouse":
                if (
                    getattr(mcap_msg.decoded, "event_type", None) == "click"
                    and mcap_msg.decoded.button is not None
                    and mcap_msg.decoded.pressed is not None
                ):
                    if mcap_msg.decoded.pressed:
                        handle_mouse_press(mcap_msg.decoded.button, mcap_msg.timestamp)
                    else:
                        handle_mouse_release(mcap_msg.decoded.button, mcap_msg.timestamp)

            elif mcap_msg.topic == "keyboard":
                if hasattr(mcap_msg.decoded, "event_type") and hasattr(mcap_msg.decoded, "vk"):
                    key_state_manager.handle_key_event(
                        mcap_msg.decoded.event_type, mcap_msg.decoded.vk, mcap_msg.timestamp
                    )

        # Handle remaining unpaired mouse events
        for press_timestamp, button_name, message_content in pending_mouse_events:
            start = format_timestamp(press_timestamp - start_time)
            end = format_timestamp(press_timestamp - start_time + MIN_MOUSE_CLICK_DURATION_NS)
            completed_events.append(CompletedEvent(press_timestamp, f"{start} --> {end}\n[mouse] {message_content}"))

        # Finalize keyboard events
        key_state_manager.finalize_remaining_subtitles()
        for key_start_time, key_end_time, key_message_content in key_state_manager.get_completed_subtitles():
            start = format_timestamp(key_start_time - start_time)
            end = format_timestamp(key_end_time - start_time)
            completed_events.append(
                CompletedEvent(key_start_time, f"{start} --> {end}\n[keyboard] {key_message_content}")
            )

        # Sort and generate subtitles
        completed_events.sort(key=lambda event: event.timestamp)
        for i, event in enumerate(completed_events, 1):
            subtitles.append(f"{i}\n{event.content}\n")

    output.write_text("\n".join(subtitles), encoding="utf-8")
    typer.echo(f"Subtitle file saved: {output}")


if __name__ == "__main__":
    typer.run(convert)
