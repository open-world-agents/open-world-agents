#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcap-owa-support==0.5.5",
#   "owa-core==0.5.5",
#   "owa-msgs==0.5.5",
#   "owa-env-desktop==0.5.5",
#   "tqdm",
#   "rich",
# ]
# [tool.uv]
# exclude-newer = "2025-08-01T12:00:00Z"
# ///
from pathlib import Path
from tqdm import tqdm
from rich import print
from concurrent.futures import ProcessPoolExecutor, as_completed

from owa.core.io.video import VideoReader, VideoWriter

VPT_FOLDER_PATH = Path(
    "/mnt/raid12/datasets/owa/mcaps/vpt"
).expanduser()  # NOTE: Change this to your VPT data folder path. We expect paired mp4 and jsonl files for VPT dataset.


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


def main(max_workers: int = None):
    if max_workers is None:
        max_workers = 50
    print(f"Using {max_workers} worker processes.")

    print(f"Reading {VPT_FOLDER_PATH=} for mp4 files.")

    mp4_target_list = [f for f in VPT_FOLDER_PATH.iterdir() if f.suffix == ".mp4" and f.is_file()]
    print(f"We will convert {len(mp4_target_list)=} mp4 files.")

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
                    print(f"Successfully converted {mp4_file_path}")
                except Exception as exc:
                    print(f"File {mp4_file_path} generated an exception: {exc}")
                finally:
                    pbar.update(1)


if __name__ == "__main__":
    main()
