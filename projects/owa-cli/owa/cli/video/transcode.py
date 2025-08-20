import subprocess
from pathlib import Path
from typing import Optional

import typer


def build_ffmpeg_cmd(
    input_path: Path,
    output_path: Path,
    fps: float = 60.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    codec: str = "libx264",
    crf: Optional[int] = None,
    keyint: int = 30,
    min_keyint: Optional[int] = None,
    scenecut: Optional[int] = None,
) -> list[str]:
    """
    Build ffmpeg command with proper video/audio settings for compatibility.

    Key parameters explained:
    - vsync=1: Forces constant frame rate (CFR) output for better compatibility
    - bframes=0: Disables B-frames for maximum player/editor compatibility
    - keyint: Maximum keyframe interval (GOP size) - affects seeking performance
    - scenecut: Scene change detection (0=disable for consistent GOP structure)
    - yuv420p: 4:2:0 chroma subsampling for universal device compatibility
    - aresample=async=1000: Audio sync correction to prevent A/V drift
    """
    cmd = ["ffmpeg", "-i", str(input_path)]

    # Force constant frame rate (CFR) - critical for seeking and compatibility
    cmd.extend(["-vsync", "1"])

    # Video filters for resolution and frame rate
    filters = []
    if width or height:
        if width and height:
            filters.append(f"scale={width}:{height}")  # Exact dimensions
        elif width:
            filters.append(f"scale={width}:-2")  # Keep aspect ratio, even height
        else:
            filters.append(f"scale=-2:{height}")  # Keep aspect ratio, even width

    if fps:
        filters.append(f"fps={fps}")  # Frame rate conversion

    if filters:
        cmd.extend(["-filter:v", ",".join(filters)])

    # Video codec selection
    cmd.extend(["-c:v", codec])

    # Quality control (CRF = Constant Rate Factor)
    if crf is not None:
        cmd.extend(["-crf", str(crf)])  # 18=high quality, 23=default, 28=lower quality

    # Keyframe and GOP structure settings - EXTREMELY IMPORTANT for seeking
    if codec in ["libx264", "libx265"]:
        params = ["bframes=0"]  # Disable B-frames for maximum compatibility

        if keyint:
            params.append(f"keyint={keyint}")  # Max keyframe interval (GOP size)
        if min_keyint:
            params.append(f"min-keyint={min_keyint}")  # Min keyframe interval
        if scenecut is not None:
            # Scene change detection: 0=disable for consistent GOP, >0=adaptive
            params.append("no-scenecut=1" if scenecut == 0 else f"scenecut={scenecut}")

        param_flag = "-x264-params" if codec == "libx264" else "-x265-params"
        cmd.extend([param_flag, ":".join(params)])

    # Pixel format for universal compatibility (4:2:0 chroma subsampling)
    cmd.extend(["-pix_fmt", "yuv420p"])

    # Audio encoding with explicit quality and sync correction
    cmd.extend(["-c:a", "aac", "-b:a", "192k", "-af", "aresample=async=1000"])

    # Preserve all streams: video, audio, subtitles
    cmd.extend(["-c:s", "copy", "-map", "0"])
    cmd.append(str(output_path))

    return cmd


def transcode(
    input_path: Path,
    output_path: Path,
    fps: float = 60.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    codec: str = "libx264",
    crf: Optional[int] = None,
    keyint: int = 30,
    min_keyint: Optional[int] = None,
    scenecut: Optional[int] = None,
    dry_run: bool = False,
) -> str:
    """Transcode video."""
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
    fps: float = typer.Option(60.0, "--fps", "-f", help="Target FPS"),
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
    Transcode video with fps/resolution/quality control.

    This command maintains all video, audio, and subtitle streams while applying
    professional encoding settings for maximum compatibility.

    Key Parameters:
    - fps: Target frame rate (default: 60) - higher fps for smooth motion
    - keyint: Keyframe interval (default: 30) - lower = better seeking, higher = better compression
    - scenecut: Scene change detection (0=disable for consistent GOP, 40=default adaptive)
    - crf: Quality level (18=high, 23=default, 28=lower) - lower numbers = better quality

    Examples:
      # Basic transcoding with defaults (60fps, keyint=30)
      owl video transcode input.mkv output.mkv

      # Custom resolution and frame rate
      owl video transcode input.mkv output.mkv --fps 30 --width 1920 --height 1080

      # High quality with fixed keyframes (good for streaming)
      owl video transcode input.mkv output.mkv --crf 18 --keyint 60 --scenecut 0

      # Screen recording optimization
      owl video transcode input.mkv output.mkv --fps 30 --keyint 30 --scenecut 0
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
