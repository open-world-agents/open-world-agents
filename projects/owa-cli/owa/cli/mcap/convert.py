from pathlib import Path

import typer
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapReader


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
        str, typer.Option(help="Comma-separated list of topics to include in the subtitle file")
    ] = "window,keyboard,mouse,keyboard/state",
    output_srt: Annotated[Path, typer.Argument(help="Path to the output .srt file")] = None,
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
            break
        else:
            typer.echo("No screen messages found in the .mcap file.")
            raise typer.Exit()
        for i, mcap_msg in enumerate(reader.iter_messages(topics=topics, start_time=start_time), start=1):
            start = format_timestamp(mcap_msg.timestamp - start_time)
            end = format_timestamp(mcap_msg.timestamp - start_time + 1_000_000_000)
            subtitles.append(f"{i}\n{start} --> {end}\n{mcap_msg.decoded}\n")

    output_srt.write_text("\n".join(subtitles), encoding="utf-8")
    print(f"Subtitle file saved as {output_srt}")


if __name__ == "__main__":
    typer.run(convert)
