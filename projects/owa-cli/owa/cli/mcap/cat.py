import datetime
import json
from pathlib import Path

import typer
from typing_extensions import Annotated

from mcap_owa.highlevel import OWAMcapReader
from owa.core.time import TimeUnits


def format_timestamp(ns):
    """Convert nanoseconds since epoch to a human-readable string with timezone awareness."""
    dt = datetime.datetime.fromtimestamp(ns / TimeUnits.SECOND, datetime.timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Trim to milliseconds


def cat(
    mcap_path: Annotated[Path, typer.Argument(help="Path to the input .mcap file")],
    pretty: Annotated[bool, typer.Option(help="Pretty print JSON output")] = True,
    topics: Annotated[str, typer.Option(help="Comma-separated list of topics to include")] = None,
    exclude: Annotated[str, typer.Option(help="Comma-separated list of topics to exclude")] = None,
    start_time: Annotated[int, typer.Option(help="Start time in seconds")] = None,
    end_time: Annotated[int, typer.Option(help="End time in seconds")] = None,
    n: Annotated[int, typer.Option(help="Number of messages to print")] = None,
):
    """
    Print messages from an `.mcap` file in a readable format.
    """
    start_time = start_time * TimeUnits.SECOND if start_time is not None else None
    end_time = end_time * TimeUnits.SECOND if end_time is not None else None

    with OWAMcapReader(mcap_path) as reader:
        topics = topics.split(",") if topics else reader.topics
        topics = set(topics) - (set(exclude.split(",")) if exclude else set())
        topics = list(topics)

        for i, mcap_msg in enumerate(reader.iter_messages(topics=topics, start_time=start_time, end_time=end_time)):
            if n is not None and i >= n:
                break

            if pretty:
                formatted_time = format_timestamp(mcap_msg.timestamp)
                try:
                    # Try Pydantic model_dump_json first
                    pretty_msg = mcap_msg.decoded.model_dump_json(indent=2, exclude_none=True)
                except AttributeError:
                    # Fall back to standard JSON serialization
                    try:
                        # Handle dict-like objects (EasyDict, etc.)
                        if hasattr(mcap_msg.decoded, "__dict__"):
                            pretty_msg = json.dumps(
                                mcap_msg.decoded.__dict__, indent=2, ensure_ascii=False, default=str
                            )
                        else:
                            pretty_msg = json.dumps(mcap_msg.decoded, indent=2, ensure_ascii=False, default=str)
                    except (TypeError, ValueError):
                        # Last resort: convert to string
                        pretty_msg = str(mcap_msg.decoded)

                typer.echo(
                    typer.style(f"[{formatted_time}]", fg=typer.colors.BLUE)
                    + typer.style(f" [{mcap_msg.topic}]", fg=typer.colors.GREEN)
                    + "\n"
                    + typer.style(pretty_msg, fg=typer.colors.CYAN)
                    + "\n"
                    + "-" * 80
                )
            else:
                typer.echo(f"Topic: {mcap_msg.topic}, Timestamp: {mcap_msg.timestamp}, Message: {mcap_msg.decoded}")


if __name__ == "__main__":
    typer.run(cat)
