import concurrent.futures
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm


def is_vfr(file_path: Path) -> bool:
    """Check if file has Variable Frame Rate."""
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

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        if not data.get("streams"):
            return False

        stream = data["streams"][0]
        avg_rate = stream.get("avg_frame_rate", "")
        r_rate = stream.get("r_frame_rate", "")

        def parse_rate(rate):
            if not rate or rate == "0/0":
                return 0
            try:
                num, den = map(int, rate.split("/"))
                return num / den if den != 0 else 0
            except (ValueError, ZeroDivisionError):
                return 0

        avg_fps = parse_rate(avg_rate)
        r_fps = parse_rate(r_rate)

        return abs(avg_fps - r_fps) > 0.01 or avg_fps == 0 or r_fps == 0

    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return False


def convert_to_cfr(file_path: Path, dry_run: bool = False) -> str:
    """Convert VFR file to CFR using ffmpeg."""
    if dry_run:
        cmd = [
            "ffmpeg",
            "-i",
            str(file_path),
            "-vsync",
            "1",
            "-filter:v",
            "fps=60",
            "-c:v",
            "libx264",
            "-x264-params",
            "keyint=30:no-scenecut=1:bframes=0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-af",
            "aresample=async=1000",
            "-c:s",
            "copy",
            f"cfr_{file_path.name}",
        ]
        return f"[DRY RUN] Would convert '{file_path.name}' to CFR using: {' '.join(cmd)}"

    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / f"cfr_{file_path.name}"
    backup_file = file_path.with_suffix(f"{file_path.suffix}.bak")

    try:
        cmd = [
            "ffmpeg",
            "-i",
            str(file_path),
            "-vsync",
            "1",
            "-filter:v",
            "fps=60",
            "-c:v",
            "libx264",
            "-x264-params",
            "keyint=30:no-scenecut=1:bframes=0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-af",
            "aresample=async=1000",
            "-c:s",
            "copy",
            str(temp_file),
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        if not temp_file.exists() or temp_file.stat().st_size < 1000:
            raise Exception("Output file missing or too small")

        if backup_file.exists():
            raise Exception(f"Backup file already exists: {backup_file}")

        shutil.copy2(file_path, backup_file)
        shutil.move(str(temp_file), str(file_path))

        if not file_path.exists() or file_path.stat().st_size < 1000:
            raise Exception("File replacement failed")

        backup_file.unlink(missing_ok=True)
        return f"Successfully converted '{file_path.name}' to CFR"

    except Exception as e:
        if backup_file.exists() and (not file_path.exists() or file_path.stat().st_size < 1000):
            shutil.move(str(backup_file), str(file_path))
            return f"Error converting '{file_path.name}': {e} (restored from backup)"
        elif backup_file.exists():
            backup_file.unlink(missing_ok=True)

        return f"Error converting '{file_path.name}': {e}"

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def process_file(file_path: Path, dry_run: bool = False):
    """Process a single .mkv file."""
    if not file_path.exists():
        print(f"Error: File does not exist: {file_path}")
        return

    if file_path.suffix.lower() != ".mkv":
        print(f"Error: Not an MKV file: {file_path.name}")
        return

    if is_vfr(file_path):
        if dry_run:
            print(f"[DRY RUN] Would convert '{file_path.name}' from VFR to CFR...")
        else:
            print(f"Converting '{file_path.name}' from VFR to CFR...")
        result = convert_to_cfr(file_path, dry_run)
        print(result)
    else:
        print(f"'{file_path.name}' is already using CFR. No conversion needed.")


def process_directory(directory_path: Path, max_workers: Optional[int] = None, dry_run: bool = False):
    """Process all .mkv files in directory recursively."""
    if not directory_path.is_dir():
        print(f"Error: '{directory_path}' is not a valid directory")
        return

    mkv_files = list(directory_path.rglob("*.mkv"))
    print(f"Found {len(mkv_files)} .mkv files. Checking for VFR...")

    # Filter for VFR files
    vfr_files = []
    with tqdm(mkv_files, desc="Analyzing files for VFR", unit="file") as pbar:
        for file in pbar:
            pbar.set_postfix_str(f"Checking {file.name}")
            if is_vfr(file):
                vfr_files.append(file)

    print(f"Found {len(vfr_files)} VFR files out of {len(mkv_files)} total files.")

    if not vfr_files:
        print("No VFR files to convert.")
        return

    # Convert files
    desc = "Analyzing conversions (dry run)" if dry_run else "Converting to CFR"
    with tqdm(total=len(vfr_files), desc=desc, unit="file") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(convert_to_cfr, file, dry_run): file for file in vfr_files}

            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    pbar.set_postfix_str(f"{'Analyzed' if dry_run else 'Completed'} {file.name}")
                    print(result)
                except Exception as e:
                    print(f"Error processing '{file.name}': {e}")
                finally:
                    pbar.update(1)


def main(
    path: str = typer.Argument(..., help="Path to MKV file or directory containing MKV files"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Maximum number of parallel conversions"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without actually converting files"),
):
    """Convert MKV files with Variable Frame Rate (VFR) to Constant Frame Rate (CFR)."""
    path_obj = Path(path)

    if path_obj.is_file():
        process_file(path_obj, dry_run)
    elif path_obj.is_dir():
        process_directory(path_obj, workers, dry_run)
    else:
        print(f"Error: '{path}' is neither a valid file nor directory")


if __name__ == "__main__":
    typer.run(main)
