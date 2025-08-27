from pathlib import Path

import typer
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapReader
from owa.env.desktop.constants import VK

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
        """
        Handle a key press event.

        Returns:
            bool: True if this is the first press (should create subtitle), False otherwise
        """
        if not self.is_pressed:
            # First press - transition from released to pressed
            self.is_pressed = True
            self.first_press_timestamp = timestamp
            self.last_press_timestamp = timestamp
            self.press_count = 1
            self.release_timestamp = None
            return True
        else:
            # Key is already pressed - just update the last press timestamp
            self.last_press_timestamp = timestamp
            self.press_count += 1
            return False

    def release(self, timestamp: int) -> bool:
        """
        Handle a key release event.

        Returns:
            bool: True if key was pressed and should finalize subtitle, False otherwise
        """
        if self.is_pressed:
            self.is_pressed = False
            self.release_timestamp = timestamp
            return True
        return False

    def get_subtitle_duration(self) -> tuple[int, int]:
        """
        Get the start and end timestamps for the subtitle.

        Returns:
            tuple[int, int]: (start_timestamp, end_timestamp)
        """
        if self.first_press_timestamp is None:
            return (0, 0)

        start_time = self.first_press_timestamp

        if self.release_timestamp is not None:
            # Key was released - use actual duration but ensure minimum
            actual_duration = self.release_timestamp - self.first_press_timestamp
            end_time = self.first_press_timestamp + max(actual_duration, MIN_KEY_PRESS_DURATION_NS)
        else:
            # Key is still pressed or no release recorded - use minimum duration
            end_time = self.first_press_timestamp + MIN_KEY_PRESS_DURATION_NS

        return (start_time, end_time)


class KeyStateManager:
    """Manages the state of all keyboard keys."""

    def __init__(self):
        self.key_states = {}  # vk -> KeyState
        self.pending_subtitles = []  # List of (KeyState, message_content) for keys that need subtitles
        self.completed_subtitles = []  # List of (start_time, end_time, message_content) for finalized subtitles

    def handle_key_event(self, event_type: str, vk: int, timestamp: int) -> None:
        """
        Handle a keyboard event (press or release).

        Args:
            event_type: "press" or "release"
            vk: Virtual key code
            timestamp: Event timestamp in nanoseconds
        """
        if vk not in self.key_states:
            self.key_states[vk] = KeyState(vk)

        key_state = self.key_states[vk]

        if event_type == "press":
            should_create_subtitle = key_state.press(timestamp)
            if should_create_subtitle:
                # Create message content
                try:
                    key_name = VK(vk).name
                except ValueError:
                    key_name = f"VK_{vk}"
                message_content = f"press {key_name}"
                self.pending_subtitles.append((key_state, message_content))

        elif event_type == "release":
            should_finalize_subtitle = key_state.release(timestamp)
            if should_finalize_subtitle:
                # Find and finalize the corresponding pending subtitle
                for i, (pending_key_state, message_content) in enumerate(self.pending_subtitles):
                    if pending_key_state is key_state:
                        start_time, end_time = key_state.get_subtitle_duration()
                        self.completed_subtitles.append((start_time, end_time, message_content))
                        self.pending_subtitles.pop(i)
                        break

    def finalize_remaining_subtitles(self) -> None:
        """
        Finalize any remaining pending subtitles (for keys that were never released).
        """
        for key_state, message_content in self.pending_subtitles:
            start_time, end_time = key_state.get_subtitle_duration()
            self.completed_subtitles.append((start_time, end_time, message_content))
        self.pending_subtitles.clear()

    def get_completed_subtitles(self) -> list[tuple[int, int, str]]:
        """
        Get all completed subtitles.

        Returns:
            list[tuple[int, int, str]]: List of (start_time, end_time, message_content)
        """
        return self.completed_subtitles.copy()


def format_timestamp(timestamp_ns: int) -> str:
    """Convert nanosecond timestamp to SRT timestamp format (HH:MM:SS,mmm)."""
    timestamp_s = timestamp_ns / 1e9
    hours = int(timestamp_s // 3600)
    minutes = int((timestamp_s % 3600) // 60)
    seconds = int(timestamp_s % 60)
    milliseconds = int((timestamp_s * 1000) % 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def convert(
    mcap_path: Annotated[Path, typer.Argument(help="Path to the input .mcap file")],
    topics: Annotated[
        list[str], typer.Option(help="Comma-separated list of topics to include in the subtitle file")
    ] = ["mouse/raw", "keyboard"],
    output_srt: Annotated[Path | None, typer.Argument(help="Path to the output .srt file")] = None,
):
    """
    Convert an `.mcap` file into an `.srt` subtitle file. After the conversion, you may play `.mkv` file and verify the sanity of data.
    """
    if output_srt is None:
        output_srt = mcap_path.with_suffix(".srt")

    subtitles = []

    with OWAMcapReader(mcap_path) as reader:
        for mcap_msg in reader.iter_messages(topics=["screen"]):
            start_time = mcap_msg.timestamp
            # Add pts_ns from the first screen message if available
            if (
                hasattr(mcap_msg.decoded, "media_ref")
                and mcap_msg.decoded.media_ref
                and hasattr(mcap_msg.decoded.media_ref, "pts_ns")
                and mcap_msg.decoded.media_ref.pts_ns is not None
            ):
                start_time -= mcap_msg.decoded.media_ref.pts_ns
            else:
                print("No pts_ns found in the first screen message. Subtitle timing may be off.")
            break
        else:
            typer.echo("No screen messages found in the .mcap file.")
            raise typer.Exit()
        # Collect all messages first to pair press/release events
        all_messages = list(reader.iter_messages(topics=topics, start_time=start_time))

        # Track mouse button states and pending press events
        mouse_button_states = {}  # button_name -> press_timestamp
        pending_mouse_events = []  # List of (press_timestamp, button_name, message_content)

        # Initialize key state manager
        key_state_manager = KeyStateManager()

        subtitle_counter = 0

        def handle_mouse_button_press(button_name: str, timestamp: int):
            """Handle mouse button press event."""
            mouse_button_states[button_name] = timestamp
            message_content = f"{button_name} click"
            pending_mouse_events.append((timestamp, button_name, message_content))

        def handle_mouse_button_release(button_name: str, timestamp: int):
            """Handle mouse button release event."""
            nonlocal subtitle_counter
            if button_name in mouse_button_states:
                press_timestamp = mouse_button_states[button_name]
                # Find the corresponding press event
                for i, (press_ts, btn_name, msg_content) in enumerate(pending_mouse_events):
                    if press_ts == press_timestamp and btn_name == button_name:
                        subtitle_counter += 1
                        start = format_timestamp(press_timestamp - start_time)
                        # Ensure minimum duration for mouse clicks
                        actual_duration = timestamp - press_timestamp
                        end_timestamp = press_timestamp + max(actual_duration, MIN_MOUSE_CLICK_DURATION_NS)
                        end = format_timestamp(end_timestamp - start_time)
                        subtitles.append(f"{subtitle_counter}\n{start} --> {end}\n[mouse] {msg_content}\n")
                        pending_mouse_events.pop(i)
                        break
                del mouse_button_states[button_name]

        # Button flag mappings for RawMouseEvent
        BUTTON_PRESS_FLAGS = {
            0x0001: "left",  # RI_MOUSE_LEFT_BUTTON_DOWN
            0x0004: "right",  # RI_MOUSE_RIGHT_BUTTON_DOWN
            0x0010: "middle",  # RI_MOUSE_MIDDLE_BUTTON_DOWN
        }

        BUTTON_RELEASE_FLAGS = {
            0x0002: "left",  # RI_MOUSE_LEFT_BUTTON_UP
            0x0008: "right",  # RI_MOUSE_RIGHT_BUTTON_UP
            0x0020: "middle",  # RI_MOUSE_MIDDLE_BUTTON_UP
        }

        for mcap_msg in all_messages:
            # Handle mouse events with press/release pairing
            if mcap_msg.topic == "mouse/raw":
                if hasattr(mcap_msg.decoded, "button_flags"):
                    button_flags = mcap_msg.decoded.button_flags

                    # Check for button press events
                    for flag, button_name in BUTTON_PRESS_FLAGS.items():
                        if button_flags & flag:
                            handle_mouse_button_press(button_name, mcap_msg.timestamp)
                            break

                    # Check for button release events
                    for flag, button_name in BUTTON_RELEASE_FLAGS.items():
                        if button_flags & flag:
                            handle_mouse_button_release(button_name, mcap_msg.timestamp)
                            break

            # Handle keyboard events with state management
            elif mcap_msg.topic == "keyboard":
                if hasattr(mcap_msg.decoded, "event_type") and hasattr(mcap_msg.decoded, "vk"):
                    key_state_manager.handle_key_event(
                        mcap_msg.decoded.event_type, mcap_msg.decoded.vk, mcap_msg.timestamp
                    )

        # Handle any remaining unpaired mouse press events (use default duration)
        for press_timestamp, button_name, message_content in pending_mouse_events:
            subtitle_counter += 1
            start = format_timestamp(press_timestamp - start_time)
            end = format_timestamp(press_timestamp - start_time + MIN_MOUSE_CLICK_DURATION_NS)
            subtitles.append(f"{subtitle_counter}\n{start} --> {end}\n[mouse] {message_content}\n")

        # Finalize any remaining keyboard key states and add keyboard subtitles
        key_state_manager.finalize_remaining_subtitles()
        keyboard_subtitles = key_state_manager.get_completed_subtitles()

        # Add keyboard subtitles to the main list
        for key_start_time, key_end_time, key_message_content in keyboard_subtitles:
            subtitle_counter += 1
            start = format_timestamp(key_start_time - start_time)
            end = format_timestamp(key_end_time - start_time)
            subtitles.append(f"{subtitle_counter}\n{start} --> {end}\n[keyboard] {key_message_content}\n")

    output_srt.write_text("\n".join(subtitles), encoding="utf-8")
    print(f"Subtitle file saved as {output_srt}")


if __name__ == "__main__":
    typer.run(convert)
