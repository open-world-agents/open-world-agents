"""
Trim mcap recording and all referenced MKV videos to a specific time range.

Due to ffmpeg copy mode's keyframe constraints, the output may include extra frames
before/after the requested range. Use --max-margin to control the allowed tolerance.
"""

import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from mediaref import MediaRef

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
from owa.core.utils.backup import BackupContext
from owa.msgs.desktop.screen import ScreenCaptured

from ..console import console

NS = 1_000_000_000


class MissingSubtitleError(ValueError):
    """Raised when a video file is missing the required subtitle track."""

    def __init__(self, mkv_path: Path):
        self.mkv_path = mkv_path
        super().__init__(
            f"No subtitle track found in '{mkv_path}'.\n"
            "       The trim command requires embedded UTC timestamps in subtitle track.\n\n"
            "Hint: Use --auto-subtitle to automatically generate and embed subtitles."
        )


# Type alias for MKV naming function: (src_mkv, dst_mcap) -> dst_mkv
MkvNamer = Callable[[Path, Path], Path]


def generate_utc_srt(mcap_path: Path, mkv_uri: str) -> str:
    """
    Generate SRT subtitle containing UTC timestamps from MCAP screen messages.

    Each subtitle entry contains the UTC nanosecond timestamp at that video PTS,
    which enables accurate time synchronization during trim operations.

    Args:
        mcap_path: Path to the MCAP file
        mkv_uri: The MKV URI referenced in screen messages

    Returns:
        SRT format string with UTC timestamps
    """
    entries: list[tuple[int, int]] = []  # (pts_ns, utc_ns)

    with OWAMcapReader(mcap_path) as reader:
        for msg in reader.iter_messages(topics=["screen"]):
            screen: ScreenCaptured = msg.decoded
            if screen.media_ref and screen.media_ref.uri == mkv_uri:
                pts_ns = screen.media_ref.pts_ns
                utc_ns = screen.utc_ns or msg.timestamp
                if pts_ns is not None:
                    entries.append((pts_ns, utc_ns))

    if not entries:
        raise ValueError(f"No screen messages found for MKV '{mkv_uri}' in {mcap_path}")

    # Sort by PTS and deduplicate
    entries = sorted(set(entries), key=lambda x: x[0])

    def format_srt_time(ns: int) -> str:
        """Format nanoseconds as SRT timestamp (HH:MM:SS,mmm)."""
        total_ms = ns // 1_000_000
        h = total_ms // 3600000
        m = (total_ms % 3600000) // 60000
        s = (total_ms % 60000) // 1000
        ms = total_ms % 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    srt_lines = []
    for i, (pts_ns, utc_ns) in enumerate(entries, 1):
        start = format_srt_time(pts_ns)
        # End time: next entry's PTS or 1 second after start
        if i < len(entries):
            end = format_srt_time(entries[i][0])
        else:
            end = format_srt_time(pts_ns + NS)
        srt_lines.append(f"{i}\n{start} --> {end}\n{utc_ns}\n")

    return "\n".join(srt_lines)


def embed_subtitle(mkv_path: Path, srt_content: str) -> None:
    """
    Embed SRT subtitle into MKV file using ffmpeg.

    Uses BackupContext for safe operation - original file is backed up
    and restored if any error occurs.

    Args:
        mkv_path: Path to the MKV file
        srt_content: SRT format subtitle content
    """
    with BackupContext(mkv_path, console=console) as _ctx:
        # Create temporary SRT file
        srt_path = mkv_path.with_suffix(".srt.tmp")
        try:
            srt_path.write_text(srt_content, encoding="utf-8")

            # Create temporary output file
            tmp_output = mkv_path.with_suffix(".mkv.tmp")

            # Mux subtitle into MKV
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(mkv_path),
                    "-i",
                    str(srt_path),
                    "-c",
                    "copy",
                    "-c:s",
                    "srt",
                    str(tmp_output),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed to embed subtitle: {result.stderr}")

            # Replace original with muxed file
            tmp_output.replace(mkv_path)

        finally:
            # Clean up temporary files
            srt_path.unlink(missing_ok=True)
            tmp_output = mkv_path.with_suffix(".mkv.tmp")
            if tmp_output.exists():
                tmp_output.unlink()


def ensure_subtitle(mcap_path: Path, mkv_path: Path, mkv_uri: str, auto_subtitle: bool) -> None:
    """
    Ensure the MKV has embedded subtitle. Generate and embed if needed.

    Args:
        mcap_path: Path to the MCAP file
        mkv_path: Path to the MKV file
        mkv_uri: The MKV URI referenced in screen messages
        auto_subtitle: If True, auto-generate and embed subtitle when missing

    Raises:
        MissingSubtitleError: If subtitle is missing and auto_subtitle is False
    """
    if get_video_start_utc(mkv_path) is not None:
        return  # Subtitle already exists

    if not auto_subtitle:
        raise MissingSubtitleError(mkv_path)

    console.print(f"[yellow]Generating UTC subtitle for {mkv_path.name}...[/yellow]")
    srt_content = generate_utc_srt(mcap_path, mkv_uri)

    console.print(f"[yellow]Embedding subtitle into {mkv_path.name}...[/yellow]")
    embed_subtitle(mkv_path, srt_content)

    console.print(f"[green]Subtitle embedded successfully into {mkv_path.name}[/green]")


def default_mkv_namer(src_mkvs: dict[str, Path], dst_mcap: Path) -> MkvNamer:
    """Default naming: single MKV uses mcap stem, multiple MKVs use original stem + _cut."""
    if len(src_mkvs) == 1:
        return lambda _src_mkv, dst: dst.with_suffix(".mkv")
    return lambda src_mkv, dst: dst.parent / f"{src_mkv.stem}_cut.mkv"


def find_all_mkvs(mcap_path: Path) -> dict[str, Path]:
    """Find all unique MKV files referenced in mcap. Returns {uri: resolved_path}."""
    mkvs: dict[str, Path] = {}
    with OWAMcapReader(mcap_path) as reader:
        for msg in reader.iter_messages(topics=["screen"]):
            screen: ScreenCaptured = msg.decoded
            if screen.media_ref and screen.media_ref.uri:
                uri = screen.media_ref.uri
                if uri not in mkvs:
                    for p in [mcap_path.parent / uri, Path(uri)]:
                        if p.exists():
                            mkvs[uri] = p.resolve()
                            break
    return mkvs


def get_video_start_utc(mkv: Path) -> int | None:
    """Get UTC corresponding to video PTS 0 from subtitle."""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mkv), "-map", "0:s:0", "-f", "srt", "-"], capture_output=True, text=True
    )
    if r.returncode != 0:
        return None
    for i, line in enumerate(lines := r.stdout.strip().split("\n")):
        if line.strip().isdigit() and i + 2 < len(lines):
            if m := re.match(r"(\d+):(\d+):(\d+),(\d+)", lines[i + 1]):
                h, mi, s, ms = map(int, m.groups())
                pts_ns = int((h * 3600 + mi * 60 + s + ms / 1000) * NS)
                if lines[i + 2].strip().isdigit():
                    return int(lines[i + 2].strip()) - pts_ns
    return None


def get_duration(mkv: Path) -> float:
    """Get video duration in seconds."""
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(mkv),
        ],
        capture_output=True,
        text=True,
    )
    return float(r.stdout.strip())


def cut_mkv(src: Path, dst: Path, start: float, duration: float, margin: float) -> None:
    """Cut mkv using ffmpeg copy mode."""
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            str(max(0, start - margin)),
            "-i",
            str(src),
            "-t",
            str(duration + 2 * margin),
            "-c",
            "copy",
            str(dst),
        ],
        check=True,
    )


def cut_mcap(src: Path, dst: Path, start_utc: int, end_utc: int, uri_map: dict[str, str]) -> dict[str, int]:
    """Filter mcap messages by UTC range and rewrite media_ref URIs."""
    stats = {"total": 0, "screen": 0}

    with OWAMcapReader(src) as reader:
        with OWAMcapWriter(dst) as writer:
            for msg in reader.iter_messages(start_time=start_utc, end_time=end_utc):
                new_timestamp = msg.timestamp - start_utc

                if msg.topic == "screen":
                    screen: ScreenCaptured = msg.decoded
                    if screen.media_ref:
                        old_uri = screen.media_ref.uri
                        new_uri = uri_map.get(old_uri, old_uri)
                        new_pts = (screen.utc_ns or msg.timestamp) - start_utc
                        screen.media_ref = MediaRef(uri=new_uri, pts_ns=new_pts)
                    writer.write_message(screen, topic=msg.topic, timestamp=new_timestamp)
                    stats["screen"] += 1
                else:
                    writer.write_message(msg.decoded, topic=msg.topic, timestamp=new_timestamp)
                stats["total"] += 1

    return stats


def trim_recording(
    src_mcap: Path,
    dst_mcap: Path,
    start: float,
    duration: float,
    mkv_namer: MkvNamer | None = None,
    max_margin: float = 5.0,
    auto_subtitle: bool = False,
) -> tuple[tuple[float, float], dict[str, Path], dict[str, Path]]:
    """
    Trim mcap recording and all referenced MKV files.

    Due to ffmpeg copy mode's keyframe constraints, the output video may include
    extra frames before/after the requested range. This function ensures that
    extra content stays within max_margin seconds.

    Args:
        src_mcap: Source mcap file path
        dst_mcap: Destination mcap file path
        start: Start time in seconds (relative to recording start)
        duration: Duration in seconds
        mkv_namer: Function (src_mkv, dst_mcap) -> dst_mkv. If None, uses default naming.
        max_margin: Maximum allowed extra content beyond [start, start+duration].
                    If the output contains frames further than this, raises an error.
                    This is important for privacy protection.
        auto_subtitle: If True, automatically generate and embed subtitles for MKVs
                       that are missing subtitle tracks.

    Returns:
        ((actual_before, actual_after), src_mkvs, dst_mkvs) where:
        - actual_before: extra seconds included before start
        - actual_after: extra seconds included after end
        - src_mkvs: {uri: src_path}
        - dst_mkvs: {uri: dst_path}

    Raises:
        MissingSubtitleError: If subtitle is missing and auto_subtitle is False.
        RuntimeError: If the output would contain content beyond max_margin from the target range.
    """
    # Find all MKVs referenced in the mcap
    src_mkvs = find_all_mkvs(src_mcap)
    if not src_mkvs:
        raise ValueError(f"No MKV files found in {src_mcap}")

    # Use default namer if not provided
    namer = mkv_namer or default_mkv_namer(src_mkvs, dst_mcap)

    # Ensure all MKVs have subtitles (auto-generate if requested)
    for uri, src_mkv in src_mkvs.items():
        ensure_subtitle(src_mcap, src_mkv, uri, auto_subtitle)

    # Get each MKV's start UTC
    mkv_start_utcs: dict[str, int] = {}
    for uri, src_mkv in src_mkvs.items():
        mkv_start = get_video_start_utc(src_mkv)
        if mkv_start is None:
            # This should not happen after ensure_subtitle, but handle it anyway
            raise MissingSubtitleError(src_mkv)
        mkv_start_utcs[uri] = mkv_start

    # Calculate target UTC range for each MKV based on its own start time
    def get_target_range(mkv_start: int) -> tuple[int, int]:
        return mkv_start + int(start * NS), mkv_start + int((start + duration) * NS)

    dst_mcap.parent.mkdir(parents=True, exist_ok=True)
    dst_mkvs: dict[str, Path] = {}

    # Try increasing margins until we cover the target range for all MKVs
    for try_margin in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
        dst_mkvs.clear()
        all_covered = True
        cut_ranges: list[tuple[int, int]] = []  # (cut_start, cut_end) for each MKV

        for uri, src_mkv in src_mkvs.items():
            mkv_start = mkv_start_utcs[uri]
            target_start, target_end = get_target_range(mkv_start)

            dst_mkv = namer(src_mkv, dst_mcap)
            cut_mkv(src_mkv, dst_mkv, start, duration, try_margin)

            cut_start = get_video_start_utc(dst_mkv)
            if cut_start is None:
                all_covered = False
                break
            cut_end = cut_start + int(get_duration(dst_mkv) * NS)

            if not (cut_start <= target_start and cut_end >= target_end):
                all_covered = False
                break

            dst_mkvs[uri] = dst_mkv
            cut_ranges.append((cut_start, cut_end))

        if all_covered and dst_mkvs:
            # Use the intersection of all cut ranges for mcap
            cut_start_utc = max(r[0] for r in cut_ranges)
            cut_end_utc = min(r[1] for r in cut_ranges)

            # Calculate actual margins based on first MKV's target (for user feedback)
            first_uri = next(iter(src_mkvs.keys()))
            first_target_start, first_target_end = get_target_range(mkv_start_utcs[first_uri])
            actual_before = (first_target_start - cut_start_utc) / NS
            actual_after = (cut_end_utc - first_target_end) / NS
            actual_margin = max(actual_before, actual_after)

            if actual_margin > max_margin:
                raise RuntimeError(
                    f"Privacy violation: output contains {actual_margin:.1f}s of extra content "
                    f"(max allowed: {max_margin}s). "
                    f"Before: {actual_before:.1f}s, After: {actual_after:.1f}s"
                )

            # Build uri_map for cut_mcap (old_uri -> new_filename)
            uri_map = {uri: dst.name for uri, dst in dst_mkvs.items()}
            cut_mcap(src_mcap, dst_mcap, cut_start_utc, cut_end_utc, uri_map)
            return (actual_before, actual_after), src_mkvs, dst_mkvs

    raise RuntimeError("Could not cover target range even with large margins")


def trim(
    input_mcap: Annotated[Path, typer.Argument(help="Input mcap file")],
    output_mcap: Annotated[Path, typer.Argument(help="Output mcap file path")],
    start: Annotated[float, typer.Option(help="Start time in seconds")],
    duration: Annotated[float, typer.Option(help="Duration in seconds")],
    max_margin: Annotated[
        float,
        typer.Option(
            help="Maximum allowed extra content beyond [start, start+duration] in seconds. "
            "Due to video keyframe constraints, the output may include extra frames. "
            "If extra content exceeds this limit, the operation fails to protect privacy."
        ),
    ] = 5.0,
    auto_subtitle: Annotated[
        bool,
        typer.Option(
            "--auto-subtitle",
            help="Automatically generate and embed UTC timestamps as subtitles "
            "into MKV files that are missing subtitle tracks. "
            "This modifies the original MKV file (with backup protection).",
        ),
    ] = False,
) -> None:
    """Trim mcap recording and referenced MKV files to a specific time range."""
    if not input_mcap.exists():
        console.print(f"[red]Error: Input file not found: {input_mcap}[/red]")
        raise typer.Exit(1)

    console.print(f"Input: {input_mcap}", highlight=False)
    console.print(f"Trim range: [{start}s, {start + duration}s] (max-margin: {max_margin}s)")

    try:
        (before, after), src_mkvs, dst_mkvs = trim_recording(
            input_mcap, output_mcap, start, duration, max_margin=max_margin, auto_subtitle=auto_subtitle
        )
    except (MissingSubtitleError, ValueError, RuntimeError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    actual_start = start - before
    actual_end = start + duration + after
    console.print(
        f"Actual range: [{actual_start:.1f}s, {actual_end:.1f}s] (margin: before={before:.1f}s, after={after:.1f}s)"
    )
    console.print("Output:")
    console.print(f"  {input_mcap} -> {output_mcap}", highlight=False)
    for uri in src_mkvs:
        console.print(f"  {src_mkvs[uri]} -> {dst_mkvs[uri]}", highlight=False)
