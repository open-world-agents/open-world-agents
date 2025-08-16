#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcap-owa-support==0.5.5",
#   "owa-core==0.5.5",
#   "opencv-python",
#   "tqdm",
#   "rich",
# ]
# [tool.uv]
# exclude-newer = "2025-08-01T12:00:00Z"
# ///
import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
from rich import print
from tqdm import tqdm

from owa.core.io.video import VideoReader, VideoWriter

TARGET_FPS = 20.0


def test_target_fps(mp4_file_path: Path) -> None:
    """Test function to analyze frame rates and conversion."""
    mkv_file_path = mp4_file_path.with_suffix(".mkv")
    print(f"Processing {mp4_file_path=}...")

    # OpenCV frame count
    cap = cv2.VideoCapture(str(mp4_file_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"opencv {frame_count=}")

    # VideoReader without target fps
    with VideoReader(mp4_file_path, keep_av_open=False) as reader:
        frame_count = sum(1 for _ in reader.read_frames())
        print(f"no target_fps {frame_count=}")

    # Convert VFR to CFR
    with VideoReader(mp4_file_path, keep_av_open=False) as reader:
        with VideoWriter(mkv_file_path, fps=TARGET_FPS, vfr=False) as writer:
            frame_count = 0
            for frame in reader.read_frames(fps=TARGET_FPS):
                writer.write_frame(frame.to_ndarray(format="rgb24"))
                frame_count += 1
            print(f"{TARGET_FPS=} {frame_count=} duration={frame_count / TARGET_FPS:.2f}s")


def process_single_file(mp4_file_path: Path) -> None:
    """Convert a single mp4 file to mkv format with constant frame rate."""
    mkv_file_path = mp4_file_path.with_suffix(".mkv")

    with VideoReader(mp4_file_path, keep_av_open=False) as reader:
        with VideoWriter(mkv_file_path, fps=TARGET_FPS, vfr=False) as writer:
            for frame in reader.read_frames(fps=TARGET_FPS):
                writer.write_frame(frame.to_ndarray(format="rgb24"))


def _calculate_shard_slice(total_files: int, shard_index: int, shard_count: int) -> tuple[int, int]:
    """Calculate start and end indices for a shard."""
    files_per_shard = total_files // shard_count
    remainder = total_files % shard_count

    if shard_index < remainder:
        start_idx = shard_index * (files_per_shard + 1)
        end_idx = start_idx + files_per_shard + 1
    else:
        start_idx = remainder * (files_per_shard + 1) + (shard_index - remainder) * files_per_shard
        end_idx = start_idx + files_per_shard

    return start_idx, end_idx


def main(
    vpt_folder_path: Path,
    max_workers: int = 10,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> None:
    """Convert VPT mp4 files to mkv format with optional sharding."""
    print(f"Using {max_workers} workers")

    mp4_files = sorted(f for f in vpt_folder_path.iterdir() if f.suffix == ".mp4" and f.is_file())
    print(f"Found {len(mp4_files)} mp4 files in {vpt_folder_path}")

    if shard_index is not None and shard_count is not None:
        start_idx, end_idx = _calculate_shard_slice(len(mp4_files), shard_index, shard_count)
        mp4_files = mp4_files[start_idx:end_idx]
        print(f"Shard {shard_index}/{shard_count}: processing files [{start_idx}:{end_idx}] ({len(mp4_files)} files)")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, f): f for f in mp4_files}

        with tqdm(total=len(mp4_files), desc="Converting files") as pbar:
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    future.result()
                    tqdm.write(f"✓ {file_path}")
                except Exception as exc:
                    tqdm.write(f"✗ {file_path}: {exc}")
                finally:
                    pbar.update(1)


def _validate_args(args: argparse.Namespace) -> Path:
    """Validate command line arguments and return the validated folder path."""
    # Validate sharding arguments
    if (args.shard_index is None) != (args.shard_count is None):
        raise SystemExit("Error: --shard-index and --shard-count must be used together")

    if args.shard_count is not None and args.shard_count <= 0:
        raise SystemExit("Error: --shard-count must be positive")

    if args.shard_index is not None and not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(f"Error: --shard-index must be between 0 and {args.shard_count - 1}")

    # Validate and return folder path
    folder_path = args.vpt_folder_path.expanduser()
    if not folder_path.exists():
        raise SystemExit(f"Error: Path does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise SystemExit(f"Error: Path is not a directory: {folder_path}")

    return folder_path


def _run_test(folder_path: Path) -> None:
    """Run test function with a random mp4 file."""
    mp4_files = [f for f in folder_path.iterdir() if f.suffix == ".mp4" and f.is_file()]
    if not mp4_files:
        raise SystemExit(f"Error: No mp4 files found in {folder_path}")

    random_file = random.choice(mp4_files)
    print(f"Testing with random file: {random_file}")
    test_target_fps(random_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VPT mp4 files to mkv format with constant frame rate")
    parser.add_argument("vpt_folder_path", type=Path, help="Path to VPT data folder containing mp4 files")
    parser.add_argument("--max-workers", type=int, default=10, help="Number of worker processes (default: 10)")
    parser.add_argument("--test", action="store_true", help="Run test with a random file")
    parser.add_argument("--shard-index", type=int, help="Shard index (0-based, use with --shard-count)")
    parser.add_argument("--shard-count", type=int, help="Total number of shards (use with --shard-index)")

    args = parser.parse_args()
    vpt_folder_path = _validate_args(args)

    if args.test:
        _run_test(vpt_folder_path)
    else:
        main(vpt_folder_path, args.max_workers, args.shard_index, args.shard_count)
