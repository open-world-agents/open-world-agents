import json
import shutil
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List

import plotext as plt
import typer

# TODO: add `gst-discoverer-1.0.exe`


def get_frame_data(video_path: str):
    """Extract all frames and their details using FFmpeg"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=pict_type,pts_time,best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        video_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse the output to find all frame types
    frames = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                # Use best_effort_timestamp if available, otherwise pts_time
                if len(parts) >= 3 and parts[0]:
                    timestamp = float(parts[0])
                else:
                    timestamp = float(parts[1])

                frame_type = parts[-1]
                frames.append({"type": frame_type, "timestamp": timestamp})
            except (ValueError, IndexError):
                continue

    return sorted(frames, key=lambda x: x["timestamp"])


def get_video_info(video_path: str) -> Dict:
    """Get comprehensive video information"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,duration",
        "-of",
        "json",
        video_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]

        # Parse the fps which is in the format "num/den"
        fps_str = stream.get("r_frame_rate", "0/1")
        if "/" in fps_str:
            num, den = map(int, fps_str.split("/"))
            fps = num / den
        else:
            fps = float(fps_str or 0)

        return {
            "codec": stream.get("codec_name", "unknown"),
            "resolution": f"{stream.get('width', 0)}x{stream.get('height', 0)}",
            "fps": fps,
            "duration": float(stream.get("duration", 0)),
        }
    except (json.JSONDecodeError, IndexError):
        return {"codec": "unknown", "resolution": "unknown", "fps": 0, "duration": 0}


def visualize_frame_pattern(frames: List[Dict], max_frames: int = 120):
    """Display a visual representation of frame patterns in terminal"""
    # Create a frame type sequence for visualization
    sequence = [frame["type"] for frame in frames]

    # Limit to max_frames to prevent terminal overflow
    if len(sequence) > max_frames:
        sequence = sequence[:max_frames]

    # Define colors for each frame type
    TYPE_COLORS = {
        "I": typer.colors.BRIGHT_RED,
        "P": typer.colors.BRIGHT_BLUE,
        "B": typer.colors.BRIGHT_GREEN,
        # Other frame types if needed
    }

    # Build the visualization string
    frame_viz = ""
    for i, frame_type in enumerate(sequence):
        color = TYPE_COLORS.get(frame_type, typer.colors.WHITE)
        frame_viz += typer.style(frame_type, fg=color)
        # Add space every 10 frames for readability
        if (i + 1) % 10 == 0:
            frame_viz += " "

    typer.echo(f"\nFrame Pattern (first {len(sequence)} frames):")
    typer.echo(frame_viz)

    # Display a legend
    legend = "Legend: "
    for frame_type, color in TYPE_COLORS.items():
        legend += typer.style(f"{frame_type}", fg=color) + " "
    typer.echo(legend)


def visualize_frame_distribution(frames: List[Dict]):
    """Visualize the distribution of frame types using plotext"""
    frame_types = [frame["type"] for frame in frames]
    counts = Counter(frame_types)

    # Calculate percentages for the pie chart
    total = len(frame_types)
    labels = []
    values = []

    for frame_type, count in sorted(counts.items()):
        percentage = (count / total) * 100
        labels.append(f"{frame_type}-frames: {count} ({percentage:.1f}%)")
        values.append(count)

    # Get terminal width for sizing
    terminal_width = shutil.get_terminal_size().columns

    # Create a bar chart
    plt.clear_figure()
    plt.bar(labels, values, orientation="horizontal")
    plt.title("Frame Type Distribution")
    plt.plotsize(min(terminal_width - 10, 100), 15)
    plt.show()


def visualize_gop_structure(frames: List[Dict], max_gops: int = 3):
    """Visualize the GOP (Group of Pictures) structure"""
    # Find I-frames to identify GOP boundaries
    gop_starts = [i for i, frame in enumerate(frames) if frame["type"] == "I"]

    # Limit to a few GOPs for clarity
    if len(gop_starts) > max_gops + 1:
        gop_starts = gop_starts[: max_gops + 1]

    # Display each GOP
    for i in range(len(gop_starts) - 1):
        gop_frames = frames[gop_starts[i] : gop_starts[i + 1]]
        gop_sequence = "".join([frame["type"] for frame in gop_frames])

        # Calculate GOP size in frames
        gop_size = len(gop_frames)

        # Time interval
        time_start = gop_frames[0]["timestamp"]
        time_end = gop_frames[-1]["timestamp"]
        time_interval = time_end - time_start

        typer.echo(f"\nGOP #{i + 1}:")
        typer.echo(f"  Size: {gop_size} frames ({time_interval:.2f} seconds)")
        typer.echo(f"  Sequence: {gop_sequence[:50]}{'...' if len(gop_sequence) > 50 else ''}")

        # Count frame types in this GOP
        type_counts = Counter([frame["type"] for frame in gop_frames])
        frame_counts = ", ".join([f"{count} {type_}" for type_, count in type_counts.items()])
        typer.echo(f"  Composition: {frame_counts}")


def analyze_iframe_intervals(frames: List[Dict], fps: float, frame_count: bool = False):
    """Analyze I-frame intervals and return statistics"""
    # Extract only I-frames
    iframes = [frame for frame in frames if frame["type"] == "I"]

    if len(iframes) <= 1:
        return None

    # Calculate intervals
    timestamps = [frame["timestamp"] for frame in iframes]
    intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

    # Convert to frame count if requested
    if frame_count and fps > 0:
        intervals = [interval * fps for interval in intervals]
        unit = "frames"
    else:
        unit = "seconds"

    # Calculate statistics
    stats = {
        "count": len(iframes),
        "min": min(intervals),
        "max": max(intervals),
        "mean": statistics.mean(intervals),
        "median": statistics.median(intervals),
        "unit": unit,
        "consistent": max(intervals) - min(intervals) < 0.1 * fps if fps > 0 else False,
    }

    return stats


def analyze_video(
    video_path: str = typer.Argument(..., help="Path to the video file"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed frame information"),
    frame_count: bool = typer.Option(False, "--frames", "-f", help="Show intervals in frames instead of seconds"),
    max_frames: int = typer.Option(200, "--max-frames", "-m", help="Maximum frames to show in pattern visualization"),
    max_gops: int = typer.Option(3, "--max-gops", "-g", help="Maximum GOPs to analyze in detail"),
):
    """Analyze frame types and patterns in a video file"""
    if not Path(video_path).exists():
        typer.echo(f"Error: File {video_path} does not exist")
        raise typer.Exit(1)

    # Title
    typer.echo(typer.style("📊 Advanced Video Frame Analysis 📊", fg=typer.colors.BRIGHT_MAGENTA, bold=True))
    typer.echo(f"Analyzing frames in {video_path}...\n")

    # Get video information
    video_info = get_video_info(video_path)
    fps = video_info["fps"]

    # Show video metadata
    typer.echo(typer.style("Video Information:", fg=typer.colors.BRIGHT_CYAN))
    typer.echo(f"  Codec: {video_info['codec']}")
    typer.echo(f"  Resolution: {video_info['resolution']}")
    typer.echo(f"  Frame Rate: {fps:.2f} fps")
    typer.echo(f"  Duration: {video_info['duration']:.2f} seconds")

    # Get all frame data
    frames = get_frame_data(video_path)

    if not frames:
        typer.echo("No frames detected or error processing the file")
        raise typer.Exit(1)

    # Count frame types
    frame_types = Counter([frame["type"] for frame in frames])

    # Display frame type counts
    typer.echo(typer.style("\nFrame Type Distribution:", fg=typer.colors.BRIGHT_CYAN))
    for frame_type, count in frame_types.items():
        percentage = (count / len(frames)) * 100
        typer.echo(f"  {frame_type}-frames: {count} ({percentage:.1f}%)")

    # Analyze I-frame intervals
    iframe_stats = analyze_iframe_intervals(frames, fps, frame_count)

    if iframe_stats:
        typer.echo(typer.style("\nI-frame Interval Analysis:", fg=typer.colors.BRIGHT_CYAN))
        typer.echo(f"  Total I-frames: {iframe_stats['count']}")
        typer.echo(f"  I-frame intervals ({iframe_stats['unit']}):")
        typer.echo(f"    Minimum: {iframe_stats['min']:.2f}")
        typer.echo(f"    Maximum: {iframe_stats['max']:.2f}")
        typer.echo(f"    Average: {iframe_stats['mean']:.2f}")
        typer.echo(f"    Median:  {iframe_stats['median']:.2f}")

        if iframe_stats["consistent"]:
            typer.echo(f"\n  Consistent GOP size detected: {iframe_stats['mean']:.2f} {iframe_stats['unit']}")
        else:
            typer.echo("\n  Variable GOP size detected")

    # Visualize frame pattern
    visualize_frame_pattern(frames, max_frames)

    # Visualize frame type distribution
    visualize_frame_distribution(frames)

    # Visualize GOP structure
    visualize_gop_structure(frames, max_gops)

    # Show detailed frame list if requested
    if detailed:
        typer.echo(typer.style("\nDetailed Frame Information:", fg=typer.colors.BRIGHT_CYAN))
        for i, frame in enumerate(frames[:max_frames]):
            typer.echo(f"  Frame {i}: Type={frame['type']}, Time={frame['timestamp']:.3f}s")

        if len(frames) > max_frames:
            typer.echo(f"  ... and {len(frames) - max_frames} more frames")


if __name__ == "__main__":
    typer.run(analyze_video)
