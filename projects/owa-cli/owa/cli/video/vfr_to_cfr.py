import concurrent.futures
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import typer


def is_vfr(file_path):
    """
    Check if the given file has Variable Frame Rate (VFR).

    Args:
        file_path (Path): Path to the mkv file

    Returns:
        bool: True if the file has VFR, False otherwise
    """
    # Use ffprobe to get frame rate information
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(file_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        data = json.loads(result.stdout)

        # Extract frame rate information
        if "streams" in data and data["streams"]:
            stream = data["streams"][0]
            avg_frame_rate = stream.get("avg_frame_rate", "")
            r_frame_rate = stream.get("r_frame_rate", "")

            # Parse the frame rates
            def parse_rate(rate):
                if not rate or rate == "0/0":
                    return 0
                num, den = map(int, rate.split("/"))
                return num / den if den != 0 else 0

            avg_fps = parse_rate(avg_frame_rate)
            r_fps = parse_rate(r_frame_rate)

            # If the rates are different or if either is 0, it's likely VFR
            if abs(avg_fps - r_fps) > 0.01 or avg_fps == 0 or r_fps == 0:
                print(
                    f"VFR detected in '{file_path.name}' at {file_path}: avg_rate={avg_frame_rate}, r_rate={r_frame_rate}"
                )
                return True
    except Exception as e:
        print(f"Error analyzing '{file_path.name}': {e}")

    return False


def convert_to_cfr(file_path):
    """
    Convert a VFR file to CFR using ffmpeg.

    Args:
        file_path (Path): Path to the mkv file

    Returns:
        str: Status message
    """
    try:
        # Create a temporary directory and file
        temp_dir = tempfile.mkdtemp()
        temp_file = Path(temp_dir) / f"cfr_{file_path.name}"

        # Use the ORIGINAL ffmpeg command with your parameters
        cmd = [
            "ffmpeg",
            "-i",
            str(file_path),
            "-vsync",
            "1",  # Ensure video sync
            "-filter:v",
            "fps=60",
            "-c:v",
            # Adopt x264 instead of x265 for 3x faster encoding
            "libx264",
            "-x264-params",
            "keyint=30:no-scenecut=1:bframes=0",
            # Re-encode audio instead of copying to maintain sync
            "-c:a",
            "aac",  # Use AAC codec for audio
            "-b:a",
            "192k",  # Good quality bitrate
            "-af",
            "aresample=async=1000",  # Fix potential audio sync issues
            "-c:s",
            "copy",
            str(temp_file),
        ]

        # Run the ffmpeg command
        process = subprocess.run(cmd, capture_output=True, text=True)

        # Check for errors
        if process.returncode != 0:
            raise Exception(f"ffmpeg error: {process.stderr}")

        # Verify the output file exists and has content
        if not temp_file.exists() or temp_file.stat().st_size < 1000:  # 1KB minimum
            raise Exception("Output file is missing or too small")

        # Make backup of original file
        backup_file = file_path.with_suffix(f"{file_path.suffix}.bak")
        shutil.copy2(file_path, backup_file)

        # Replace the original file with the temporary file
        shutil.move(str(temp_file), str(file_path))

        # Remove backup if everything worked
        if file_path.exists() and file_path.stat().st_size > 1000:
            backup_file.unlink(missing_ok=True)

        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        return f"Successfully converted '{file_path.name}' to CFR at {file_path}"

    except Exception as e:
        # Clean up temporary directory if it exists
        if "temp_dir" in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Restore from backup if needed
        backup_file = file_path.with_suffix(f"{file_path.suffix}.bak")
        if backup_file.exists() and (not file_path.exists() or file_path.stat().st_size < 1000):
            shutil.move(str(backup_file), str(file_path))

        return f"Error converting '{file_path.name}': {str(e)}"


def process_file(file_path):
    """
    Process a single .mkv file.

    Args:
        file_path (str): Path to the .mkv file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: File '{file_path.name}' does not exist at {file_path}")
        return

    if file_path.suffix.lower() != ".mkv":
        print(f"Error: '{file_path.name}' is not an MKV file")
        return

    if is_vfr(file_path):
        print(f"Converting '{file_path.name}' from VFR to CFR...")
        result = convert_to_cfr(file_path)
        print(result)
    else:
        print(f"'{file_path.name}' is already using CFR. No conversion needed.")


def process_directory(directory, max_workers=None):
    """
    Process all .mkv files in the given directory recursively.

    Args:
        directory (str): Directory to scan for .mkv files
        max_workers (int, optional): Maximum number of parallel conversions
    """
    directory_path = Path(directory)
    if not directory_path.is_dir():
        print(f"Error: '{directory}' is not a valid directory")
        return

    # Get all .mkv files recursively
    mkv_files = list(directory_path.rglob("*.mkv"))

    print(f"Found {len(mkv_files)} .mkv files in '{directory_path.name}'. Checking for VFR...")

    # Filter for VFR files
    vfr_files = [file for file in mkv_files if is_vfr(file)]

    print(f"Found {len(vfr_files)} VFR files out of {len(mkv_files)} total .mkv files.")

    if not vfr_files:
        print("No VFR files to convert. Exiting.")
        return

    # Convert VFR files to CFR in parallel
    print(f"Converting {len(vfr_files)} files to CFR...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(convert_to_cfr, file): file for file in vfr_files}

        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Error processing '{file.name}': {e}")


def main(
    path: str = typer.Argument(..., help="Path to MKV file or directory containing MKV files"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Maximum number of parallel conversions"),
):
    """
    Convert MKV files with Variable Frame Rate (VFR) to Constant Frame Rate (CFR).
    Can process either a single file or an entire directory recursively.
    """
    path_obj = Path(path)

    if path_obj.is_file():
        process_file(path)
    elif path_obj.is_dir():
        process_directory(path, workers)
    else:
        print(f"Error: '{path}' is neither a valid file nor directory")


if __name__ == "__main__":
    typer.run(main)
