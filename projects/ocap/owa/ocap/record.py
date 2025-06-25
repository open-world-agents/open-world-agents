import os
import time
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from typing import Optional

import typer
from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapWriter
from owa.cli.utils import check_for_update
from owa.core import LISTENERS, get_plugin_discovery
from owa.core.time import TimeUnits

# TODO: apply https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.remove()
# how to use loguru with tqdm: https://github.com/Delgan/loguru/issues/135
logger.add(lambda msg: tqdm.write(msg, end=""), filter={"owa.ocap": "DEBUG", "owa.env.gst": "INFO"}, colorize=True)

event_queue = Queue()
MCAP_LOCATION = None


def enqueue_event(event, *, topic):
    event_queue.put((topic, event, time.time_ns()))


def keyboard_monitor_callback(event):
    # info only for F1-F12 keys
    if 0x70 <= event.vk <= 0x7B and event.event_type == "press":
        logger.info(f"F1-F12 key pressed: F{event.vk - 0x70 + 1}")
    enqueue_event(event, topic="keyboard")


def screen_capture_callback(event):
    global MCAP_LOCATION
    event.media_ref.path = Path(event.media_ref.path).relative_to(MCAP_LOCATION.parent).as_posix()
    enqueue_event(event, topic="screen")


def check_plugin():
    plugin_discovery = get_plugin_discovery()
    success, failed = plugin_discovery.get_plugin_info(["desktop", "gst"])
    assert len(success) == 2, f"Failed to load plugins: {failed}"


USER_INSTRUCTION = """
Since this recorder records all screen/keyboard/mouse/window events, be aware NOT to record sensitive information, such as passwords, credit card numbers, etc.

Press Ctrl+C to stop recording.
"""


@contextmanager
def setup_resources(
    file_location: Path,
    record_audio: bool,
    record_video: bool,
    record_timestamp: bool,
    show_cursor: bool,
    fps: float,
    window_name: Optional[str],
    monitor_idx: Optional[int],
    width: Optional[int],
    height: Optional[int],
    additional_properties: dict,
):
    check_plugin()
    # Instantiate all listeners and recorder etc.
    recorder = LISTENERS["gst/omnimodal.appsink_recorder"]()
    keyboard_listener = LISTENERS["desktop/keyboard"]().configure(callback=keyboard_monitor_callback)
    mouse_listener = LISTENERS["desktop/mouse"]().configure(callback=lambda event: enqueue_event(event, topic="mouse"))
    window_listener = LISTENERS["desktop/window"]().configure(
        callback=lambda event: enqueue_event(event, topic="window")
    )
    keyboard_state_listener = LISTENERS["desktop/keyboard_state"]().configure(
        callback=lambda event: enqueue_event(event, topic="keyboard/state")
    )
    mouse_state_listener = LISTENERS["desktop/mouse_state"]().configure(
        callback=lambda event: enqueue_event(event, topic="mouse/state")
    )
    # Configure recorder
    recorder.configure(
        filesink_location=file_location.with_suffix(".mkv"),
        record_audio=record_audio,
        record_video=record_video,
        record_timestamp=record_timestamp,
        show_cursor=show_cursor,
        fps=fps,
        window_name=window_name,
        monitor_idx=monitor_idx,
        width=width,
        height=height,
        additional_properties=additional_properties,
        callback=screen_capture_callback,
    )

    resources = [
        (recorder, "recorder"),
        (keyboard_listener, "keyboard listener"),
        (mouse_listener, "mouse listener"),
        (window_listener, "window listener"),
        (keyboard_state_listener, "keyboard state listener"),
        (mouse_state_listener, "mouse state listener"),
    ]
    for resource, name in resources:
        resource.start()
        logger.debug(f"Started {name}")
    try:
        yield
    finally:
        for resource, name in reversed(resources):
            try:
                resource.stop()
                resource.join(timeout=5)
                assert not resource.is_alive(), f"{name} is still alive after stop"
                logger.debug(f"Stopped {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")


def parse_additional_properties(additional_args: Optional[str]) -> dict:
    additional_properties = {}
    if additional_args is not None:
        for arg in additional_args.split(","):
            key, value = arg.split("=")
            additional_properties[key] = value
    return additional_properties


def ensure_output_files_ready(file_location: Path):
    output_file = file_location.with_suffix(".mcap")
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Created directory {output_file.parent}")
    if output_file.exists() or output_file.with_suffix(".mkv").exists():
        delete = typer.confirm("The output file already exists. Do you want to delete it?")
        if not delete:
            print("The recording is aborted.")
            raise typer.Abort()
        output_file.unlink(missing_ok=True)
        output_file.with_suffix(".mkv").unlink(missing_ok=True)
        logger.warning(f"Deleted existing file {output_file}")
    return output_file


def record(
    file_location: Annotated[
        Path,
        typer.Argument(
            help="The location of the output file. If `output.mcap` is given as argument, the output file would be `output.mcap` and `output.mkv`."
        ),
    ],
    *,
    record_audio: Annotated[bool, typer.Option(help="Whether to record audio")] = True,
    record_video: Annotated[bool, typer.Option(help="Whether to record video")] = True,
    record_timestamp: Annotated[bool, typer.Option(help="Whether to record timestamp")] = True,
    show_cursor: Annotated[bool, typer.Option(help="Whether to show the cursor in the capture")] = True,
    fps: Annotated[Optional[float], typer.Option(help="The frame rate of the video. Default is 60 fps.")] = 60.0,
    window_name: Annotated[
        Optional[str], typer.Option(help="The name of the window to capture, substring of window name is supported")
    ] = None,
    monitor_idx: Annotated[Optional[int], typer.Option(help="The index of the monitor to capture")] = None,
    width: Annotated[
        Optional[int],
        typer.Option(help="The width of the video. If None, the width will be determined by the source."),
    ] = None,
    height: Annotated[
        Optional[int],
        typer.Option(help="The height of the video. If None, the height will be determined by the source."),
    ] = None,
    additional_args: Annotated[
        Optional[str],
        typer.Option(
            help="Additional arguments to pass to the pipeline. For detail, see https://gstreamer.freedesktop.org/documentation/d3d11/d3d11screencapturesrc.html"
        ),
    ] = None,
):
    """Record screen, keyboard, mouse, and window events to an `.mcap` and `.mkv` file."""
    global MCAP_LOCATION
    output_file = ensure_output_files_ready(file_location)
    MCAP_LOCATION = output_file

    if window_name is not None:
        logger.warning(
            "⚠️ WINDOW CAPTURE LIMITATION (as of 2025-03-20) ⚠️\n"
            "When capturing a specific window, mouse coordinates cannot be accurately aligned with the window content due to "
            "limitations in the Windows API (WGC).\n\n"
            "RECOMMENDATION:\n"
            "- Use FULL SCREEN capture when you need mouse event tracking\n"
            "- Full screen mode in games works well if the video output matches your monitor resolution (e.g., 1920x1080)\n"
            "- Any non-fullscreen capture will have misaligned mouse coordinates in the recording"
        )
    additional_properties = parse_additional_properties(additional_args)

    logger.info(USER_INSTRUCTION)

    with setup_resources(
        file_location=output_file,
        record_audio=record_audio,
        record_video=record_video,
        record_timestamp=record_timestamp,
        show_cursor=show_cursor,
        fps=fps,
        window_name=window_name,
        monitor_idx=monitor_idx,
        width=width,
        height=height,
        additional_properties=additional_properties,
    ):
        with OWAMcapWriter(output_file) as writer, tqdm(desc="Recording", unit="event", dynamic_ncols=True) as pbar:
            try:
                while True:
                    topic, event, publish_time = event_queue.get()
                    pbar.update()
                    latency = time.time_ns() - publish_time
                    # warn if latency is too high, i.e., > 100ms
                    if latency > 100 * TimeUnits.MSECOND:
                        logger.warning(
                            f"High latency: {latency / TimeUnits.MSECOND:.2f}ms while processing {topic} event."
                        )
                    writer.write_message(topic, event, publish_time=publish_time)
            except KeyboardInterrupt:
                logger.info("Recording stopped by user.")
            finally:
                # Resources are cleaned up by context managers
                logger.info(f"Output file saved to {output_file}")


def main():
    # Check for updates on startup (skip in CI environments)
    if not os.getenv("GITHUB_ACTIONS"):
        check_for_update("ocap")
    typer.run(record)


if __name__ == "__main__":
    main()
