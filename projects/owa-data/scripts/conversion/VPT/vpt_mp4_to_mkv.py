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
import cv2
import random
from pathlib import Path
from tqdm import tqdm
from rich import print
from concurrent.futures import ProcessPoolExecutor, as_completed

from owa.core.io.video import VideoReader, VideoWriter


def test_target_fps(mp4_file_path):
    mkv_file_path = mp4_file_path.with_suffix(".mkv")

    print(f"Processing {mp4_file_path=}...")

    cap = cv2.VideoCapture(mp4_file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"opencv {frame_count=}")

    with VideoReader(mp4_file_path, force_close=True) as reader:
        frame_count = 0
        for frame in reader.read_frames():  # Sample at regular intervals
            frame_array = frame.to_ndarray(format="rgb24")
            frame_count += 1
        print(f"no target_fps {frame_count=}")  # supposed to be 6000 + 1

    # Process VFR to CFR: read with fps sampling, write as CFR
    target_fps = 20.0
    with VideoReader(mp4_file_path, force_close=True) as reader:
        with VideoWriter(mkv_file_path, fps=target_fps, vfr=False) as writer:
            frame_count = 0
            for frame in reader.read_frames(fps=target_fps):  # Sample at regular intervals
                frame_array = frame.to_ndarray(format="rgb24")
                writer.write_frame(frame_array)
                frame_count += 1
            print(f"{target_fps=} {frame_count=} {frame_count/target_fps=}")


def process_single_file(mp4_file_path):
    """Process a single mp4 file and convert it to mkv format."""

    mkv_file_path = mp4_file_path.with_suffix(".mkv")

    # Process VFR to CFR: read with fps sampling, write as CFR
    target_fps = 20.0
    with VideoReader(mp4_file_path, force_close=True) as reader:
        with VideoWriter(mkv_file_path, fps=target_fps, vfr=False) as writer:
            frame_count = 0
            for frame in reader.read_frames(fps=target_fps):  # Sample at regular intervals
                frame_array = frame.to_ndarray(format="rgb24")
                writer.write_frame(frame_array)
                frame_count += 1


def main(
    vpt_folder_path: Path,
    max_workers: int = 10,
    shard_index: int | None = None,
    shard_count: int | None = None,
):
    print(f"Using {max_workers} worker processes.")

    print(f"Reading {vpt_folder_path=} for mp4 files.")

    mp4_target_list = sorted([f for f in vpt_folder_path.iterdir() if f.suffix == ".mp4" and f.is_file()])
    print(f"Found {len(mp4_target_list)} total mp4 files.")

    # Apply sharding if specified
    if shard_index is not None and shard_count is not None:
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError(f"shard_index ({shard_index}) must be between 0 and {shard_count - 1}")

        # Calculate shard boundaries
        total_files = len(mp4_target_list)
        files_per_shard = total_files // shard_count
        remainder = total_files % shard_count

        # Calculate start and end indices for this shard
        # Distribute remainder files among the first 'remainder' shards, they will have 1 more file than the others
        if shard_index < remainder:
            start_idx = shard_index * (files_per_shard + 1)
            end_idx = start_idx + files_per_shard + 1
        else:
            start_idx = remainder * (files_per_shard + 1) + (shard_index - remainder) * files_per_shard
            end_idx = start_idx + files_per_shard

        mp4_target_list = mp4_target_list[start_idx:end_idx]
        print(
            f"Shard {shard_index=}/{shard_count=}: Processing files \[{start_idx=}:{end_idx=}] ({len(mp4_target_list)} files)"
        )
    else:
        print(f"We will convert {len(mp4_target_list)} mp4 files.")

    # Use ProcessPoolExecutor for multiprocessing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, mp4_file_path): mp4_file_path for mp4_file_path in mp4_target_list
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(mp4_target_list), desc="Converting files") as pbar:
            for future in as_completed(future_to_file):
                mp4_file_path = future_to_file[future]
                try:
                    future.result()  # Get the result (or raise exception if there was one)
                    tqdm.write(f"Successfully converted {mp4_file_path}")
                except Exception as exc:
                    tqdm.write(f"File {mp4_file_path} generated an exception: {exc}")
                finally:
                    pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VPT mp4 files to mkv format with constant frame rate.")
    parser.add_argument(
        "vpt_folder_path",
        type=Path,
        help="Path to the VPT data folder containing mp4 files. We expect paired mp4 and jsonl files for VPT dataset.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=10, help="Maximum number of worker processes to use (default: 10)"
    )
    parser.add_argument("--test", action="store_true", help="Run test_target_fps with a random file from the folder")
    parser.add_argument(
        "--shard-index",
        type=int,
        help="Index of the current shard (0-based). Must be used together with --shard-count.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        help="Total number of shards for distributed processing. Must be used together with --shard-index.",
    )

    args = parser.parse_args()

    # Validate sharding arguments
    if (args.shard_index is None) != (args.shard_count is None):
        print("Error: --shard-index and --shard-count must be used together or not at all.")
        exit(1)

    if args.shard_count is not None and args.shard_count <= 0:
        print("Error: --shard-count must be a positive integer.")
        exit(1)

    if args.shard_index is not None and (args.shard_index < 0 or args.shard_index >= args.shard_count):
        print(f"Error: --shard-index must be between 0 and {args.shard_count - 1}.")
        exit(1)

    # Expand user path and ensure it's a directory
    vpt_folder_path = args.vpt_folder_path.expanduser()
    if not vpt_folder_path.exists():
        print(f"Error: VPT folder path does not exist: {vpt_folder_path}")
        exit(1)
    if not vpt_folder_path.is_dir():
        print(f"Error: VPT folder path is not a directory: {vpt_folder_path}")
        exit(1)

    # Handle test mode
    if args.test:
        # Find all mp4 files in the folder
        mp4_files = [f for f in vpt_folder_path.iterdir() if f.suffix == ".mp4" and f.is_file()]

        if not mp4_files:
            print(f"Error: No mp4 files found in {vpt_folder_path}")
            exit(1)

        # Select a random file
        random_file = random.choice(mp4_files)
        print(f"Running test_target_fps with random file: {random_file}")

        # Run the test function
        test_target_fps(random_file)
    else:
        main(vpt_folder_path, args.max_workers, args.shard_index, args.shard_count)
