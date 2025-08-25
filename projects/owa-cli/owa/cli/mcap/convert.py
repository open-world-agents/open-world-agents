from pathlib import Path
from typing import Iterable

import typer
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapReader
from owa.env.desktop.constants import VK

# Constants
MIN_MOUSE_CLICK_DURATION_NS = 500_000_000  # 500ms in nanoseconds


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
            if hasattr(mcap_msg.decoded, 'media_ref') and mcap_msg.decoded.media_ref and hasattr(mcap_msg.decoded.media_ref, 'pts_ns') and mcap_msg.decoded.media_ref.pts_ns is not None:
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
            0x0001: "left",    # RI_MOUSE_LEFT_BUTTON_DOWN
            0x0004: "right",   # RI_MOUSE_RIGHT_BUTTON_DOWN
            0x0010: "middle",  # RI_MOUSE_MIDDLE_BUTTON_DOWN
        }

        BUTTON_RELEASE_FLAGS = {
            0x0002: "left",    # RI_MOUSE_LEFT_BUTTON_UP
            0x0008: "right",   # RI_MOUSE_RIGHT_BUTTON_UP
            0x0020: "middle",  # RI_MOUSE_MIDDLE_BUTTON_UP
        }

        for mcap_msg in all_messages:
            # Handle mouse events with press/release pairing
            if mcap_msg.topic == "mouse/raw":
                if hasattr(mcap_msg.decoded, 'button_flags'):
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

            # Handle keyboard events (unchanged logic)
            elif mcap_msg.topic == "keyboard":
                should_log = True
                if should_log:
                    subtitle_counter += 1
                    start = format_timestamp(mcap_msg.timestamp - start_time)
                    end = format_timestamp(mcap_msg.timestamp - start_time + MIN_MOUSE_CLICK_DURATION_NS)

                    if hasattr(mcap_msg.decoded, 'event_type') and hasattr(mcap_msg.decoded, 'vk'):
                        message_content = f"{mcap_msg.decoded.event_type} {VK(mcap_msg.decoded.vk).name}"
                    else:
                        message_content = str(mcap_msg.decoded)

                    subtitles.append(f"{subtitle_counter}\n{start} --> {end}\n[keyboard] {message_content}\n")

        # Handle any remaining unpaired mouse press events (use default duration)
        for press_timestamp, button_name, message_content in pending_mouse_events:
            subtitle_counter += 1
            start = format_timestamp(press_timestamp - start_time)
            end = format_timestamp(press_timestamp - start_time + MIN_MOUSE_CLICK_DURATION_NS)
            subtitles.append(f"{subtitle_counter}\n{start} --> {end}\n[mouse] {message_content}\n")

    output_srt.write_text("\n".join(subtitles), encoding="utf-8")
    print(f"Subtitle file saved as {output_srt}")


if __name__ == "__main__":
    typer.run(convert)
