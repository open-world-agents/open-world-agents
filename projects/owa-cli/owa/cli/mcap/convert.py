from pathlib import Path
from typing import Iterable

import typer
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapReader
from owa.env.desktop.constants import VK


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
    ] = ["mouse", "keyboard"],
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

        # Track keyboard key states and pending press events
        keyboard_key_states = {}  # vk -> press_timestamp
        pending_keyboard_events = []  # List of (press_timestamp, vk, press_count)

        subtitle_counter = 0

        for mcap_msg in all_messages:
            # Handle mouse events with press/release pairing
            if mcap_msg.topic == "mouse":
                if hasattr(mcap_msg.decoded, 'event_type') and hasattr(mcap_msg.decoded, 'pressed') and hasattr(mcap_msg.decoded, 'button'):
                    if mcap_msg.decoded.event_type == "click":
                        button_name = mcap_msg.decoded.button
                        if mcap_msg.decoded.pressed:
                            # Button press - store the press event
                            mouse_button_states[button_name] = mcap_msg.timestamp
                            message_content = f"{button_name} click"
                            pending_mouse_events.append((mcap_msg.timestamp, button_name, message_content))
                        else:
                            # Button release - create subtitle with proper duration
                            if button_name in mouse_button_states:
                                press_timestamp = mouse_button_states[button_name]
                                # Find the corresponding press event
                                for i, (press_ts, btn_name, msg_content) in enumerate(pending_mouse_events):
                                    if press_ts == press_timestamp and btn_name == button_name:
                                        subtitle_counter += 1
                                        start = format_timestamp(press_timestamp - start_time)
                                        end = format_timestamp(mcap_msg.timestamp - start_time)
                                        subtitles.append(f"{subtitle_counter}\n{start} --> {end}\n[mouse] {msg_content}\n")
                                        pending_mouse_events.pop(i)
                                        break
                                del mouse_button_states[button_name]

            # Handle keyboard events (unchanged logic)
            elif mcap_msg.topic == "keyboard":
                should_log = True
                if should_log:
                    subtitle_counter += 1
                    start = format_timestamp(mcap_msg.timestamp - start_time)
                    end = format_timestamp(mcap_msg.timestamp - start_time + 500_000_000)

                    if hasattr(mcap_msg.decoded, 'event_type') and hasattr(mcap_msg.decoded, 'vk'):
                        message_content = f"{mcap_msg.decoded.event_type} {VK(mcap_msg.decoded.vk).name}"
                    else:
                        message_content = str(mcap_msg.decoded)

                    subtitles.append(f"{subtitle_counter}\n{start} --> {end}\n[keyboard] {message_content}\n")

        # Handle any remaining unpaired mouse press events (use default duration)
        for press_timestamp, button_name, message_content in pending_mouse_events:
            subtitle_counter += 1
            start = format_timestamp(press_timestamp - start_time)
            end = format_timestamp(press_timestamp - start_time + 500_000_000)
            subtitles.append(f"{subtitle_counter}\n{start} --> {end}\n[mouse] {message_content}\n")

    output_srt.write_text("\n".join(subtitles), encoding="utf-8")
    print(f"Subtitle file saved as {output_srt}")


if __name__ == "__main__":
    typer.run(convert)
