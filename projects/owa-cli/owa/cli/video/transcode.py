import subprocess
from pathlib import Path
from typing import Optional

import typer


def build_ffmpeg_cmd(
    input_path: Path,
    output_path: Path,
    fps: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    codec: str = "libx264",
    crf: Optional[int] = None,
    keyint: int = 30,
    min_keyint: Optional[int] = None,
    scenecut: Optional[int] = None,
) -> list[str]:
    """Build FFmpeg command with optimized settings for video compatibility."""
    cmd = ["ffmpeg", "-i", str(input_path)]

    # Force constant frame rate output for compatibility
    cmd.extend(["-vsync", "1"])

    # Build video filter chain
    filters = []
    if width or height:
        if width and height:
            filters.append(f"scale={width}:{height}")
        elif width:
            filters.append(f"scale={width}:-2")  # Maintain aspect ratio
        else:
            filters.append(f"scale=-2:{height}")  # Maintain aspect ratio

    if fps:
        filters.append(f"fps={fps}")

    if filters:
        cmd.extend(["-filter:v", ",".join(filters)])

    # Video codec configuration
    cmd.extend(["-c:v", codec])

    # Quality control via Constant Rate Factor
    if crf is not None:
        cmd.extend(["-crf", str(crf)])

    # Configure GOP structure for optimal seeking performance
    if codec in ["libx264", "libx265"]:
        params = ["bframes=0"]  # Disable B-frames for compatibility

        if keyint:
            params.append(f"keyint={keyint}")
        if min_keyint:
            params.append(f"min-keyint={min_keyint}")
        if scenecut is not None:
            params.append("no-scenecut=1" if scenecut == 0 else f"scenecut={scenecut}")

        param_flag = "-x264-params" if codec == "libx264" else "-x265-params"
        cmd.extend([param_flag, ":".join(params)])

    # Standard pixel format for universal compatibility
    cmd.extend(["-pix_fmt", "yuv420p"])

    # Audio encoding with sync correction
    cmd.extend(["-c:a", "aac", "-b:a", "192k", "-af", "aresample=async=1000"])

    # Preserve all streams
    cmd.extend(["-c:s", "copy", "-map", "0"])
    cmd.append(str(output_path))

    return cmd


def transcode(
    input_path: Path,
    output_path: Path,
    fps: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    codec: str = "libx264",
    crf: Optional[int] = None,
    keyint: int = 30,
    min_keyint: Optional[int] = None,
    scenecut: Optional[int] = None,
    dry_run: bool = False,
) -> str:
    """Execute video transcoding with specified parameters."""
    cmd = build_ffmpeg_cmd(input_path, output_path, fps, width, height, codec, crf, keyint, min_keyint, scenecut)

    if dry_run:
        return f"[DRY RUN] {' '.join(cmd)}"

    # Write directly to output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return f"✓ Transcoded {input_path.name} → {output_path.name}"
    except subprocess.CalledProcessError as e:
        return f"✗ Error: {e.stderr if e.stderr else 'Unknown error'}"


def main(
    input_path: str = typer.Argument(..., help="Input video file"),
    output_path: str = typer.Argument(..., help="Output video file"),
    fps: Optional[float] = typer.Option(None, "--fps", "-f", help="Target FPS"),
    width: Optional[int] = typer.Option(None, "--width", "-w", help="Target width"),
    height: Optional[int] = typer.Option(None, "--height", "-h", help="Target height"),
    codec: str = typer.Option("libx264", "--codec", "-c", help="Video codec"),
    crf: Optional[int] = typer.Option(None, "--crf", help="Quality (0-51, lower=better)"),
    keyint: int = typer.Option(30, "--keyint", "-k", help="Keyframe interval"),
    min_keyint: Optional[int] = typer.Option(None, "--min-keyint", help="Min keyframe interval"),
    scenecut: Optional[int] = typer.Option(None, "--scenecut", help="Scene cut threshold (0=disable)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show command only"),
):
    """
    Transcode video files with professional encoding settings.

    Options:
        fps: Target frame rate (forces constant frame rate output)
        width/height: Target resolution in pixels (maintains aspect ratio if only one specified)
        codec: Video codec (libx264 for H.264, libx265 for H.265)
        crf: Quality level (0-51, lower=better quality, 18=high, 23=default, 28=smaller file)
        keyint: Keyframe interval/GOP size (lower=better seeking, higher=better compression)
        min-keyint: Minimum keyframe interval (optional, for fine control)
        scenecut: Scene change detection (0=disable for consistent GOP, 40=default adaptive)
        dry-run: Show command without executing

    Examples:
        # Basic transcoding
        owa video transcode input.mkv output.mkv

        # Custom resolution and frame rate
        owa video transcode input.mkv output.mkv --fps 30 --width 1920 --height 1080

        # High quality encoding for archival
        owa video transcode input.mkv output.mkv --crf 18 --keyint 60 --scenecut 0

        # Streaming-optimized (consistent keyframes)
        owa video transcode input.mkv output.mkv --keyint 30 --scenecut 0

        # Preview command without execution
        owa video transcode input.mkv output.mkv --dry-run
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        typer.echo(f"Error: {input_path} not found")
        raise typer.Exit(1)

    # Basic validation
    if crf is not None and not (0 <= crf <= 51):
        typer.echo("Error: CRF must be 0-51")
        raise typer.Exit(1)

    if keyint and keyint <= 0:
        typer.echo("Error: keyint must be positive")
        raise typer.Exit(1)

    result = transcode(input_file, output_file, fps, width, height, codec, crf, keyint, min_keyint, scenecut, dry_run)
    if result.startswith("✗"):
        typer.echo(f"[bold red]{result}[/bold red]", err=True)
        raise typer.Exit(code=1)
    typer.echo(result)


if __name__ == "__main__":
    typer.run(main)
