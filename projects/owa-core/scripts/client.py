#!/usr/bin/env python3
"""
Benchmark extract_frame_api over multiple videos with random PTS sampling.

Measures p95/p99 latencies and throughput (requests/sec) for each concurrency level
within a fixed benchmark duration.
"""

import argparse
import base64
import random
import statistics
import threading
import time
from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
import requests


def extract_frame_api(
    video_path: Union[str, Path],
    pts: float,
    server_url: str = "http://127.0.0.1:8000",
) -> np.ndarray:
    """
    Send a frame-extraction request and return the RGB frame.

    Args:
        video_path: Path to the video file.
        pts: Timestamp in seconds.
        server_url: Base URL of the decoding server.

    Returns:
        RGB frame as a numpy array of shape (H, W, 3).

    Raises:
        requests.RequestException: On network or HTTP errors.
        ValueError: On missing or undecodable frame data.
    """
    resp = requests.post(
        f"{server_url}/predict",
        json={"video_path": str(video_path), "pts": pts},
    )
    resp.raise_for_status()
    data = resp.json()

    if "frame" not in data:
        raise ValueError("Missing 'frame' in response JSON")

    img_bytes = base64.b64decode(data["frame"])
    img_bgr = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image bytes")

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def get_video_durations(video_paths: List[Path]) -> Dict[Path, float]:
    """
    Compute the duration (in seconds) of each video.

    Args:
        video_paths: List of video file paths.

    Returns:
        Mapping from each Path to its duration in seconds.

    Raises:
        ValueError: If a video cannot be opened or has invalid FPS.
    """
    durations: Dict[Path, float] = {}
    for path in video_paths:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps <= 0:
            raise ValueError(f"Invalid FPS ({fps}) for video {path}")
        durations[path] = frames / fps

    return durations


def run_benchmark(
    video_paths: List[Path],
    durations: Dict[Path, float],
    server_url: str,
    concurrency: int,
    duration_seconds: float,
) -> Dict[str, float]:
    """
    Run the benchmark at a given concurrency level for a fixed duration.

    Args:
        video_paths: List of video files.
        durations: Precomputed durations for each video.
        server_url: Decoding server URL.
        concurrency: Number of parallel worker threads.
        duration_seconds: How long to run (per concurrency).

    Returns:
        A dict with keys: 'requests', 'throughput', 'p95', 'p99'.
    """
    latencies: List[float] = []
    lock = threading.Lock()
    end_time = time.perf_counter() + duration_seconds

    def worker() -> None:
        while time.perf_counter() < end_time:
            video = random.choice(video_paths)
            pts = random.random() * durations[video]

            start = time.perf_counter()
            extract_frame_api(video, pts, server_url)
            elapsed = time.perf_counter() - start

            with lock:
                latencies.append(elapsed)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = len(latencies)
    if total == 0:
        raise RuntimeError("No successful requests completed during benchmark")

    throughput = total / duration_seconds
    p95 = statistics.quantiles(latencies, n=100)[94]
    p99 = statistics.quantiles(latencies, n=100)[98]

    return {"requests": total, "throughput": throughput, "p95": p95, "p99": p99}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Benchmark extract_frame_api with random PTS sampling")
    parser.add_argument(
        "--video-list",
        type=Path,
        nargs="+",
        required=True,
        help="List of video file paths to benchmark.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Decoding server base URL.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64],
        help="Concurrency levels to test, e.g. --concurrency 1 2 4 8",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=5.0,
        help="Benchmark duration (in seconds) per concurrency level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = args.video_list
    print("Computing video durations...")
    durations = get_video_durations(videos)

    print(f"Running benchmark for {args.duration_seconds}s each:")
    for conc in args.concurrency:
        metrics = run_benchmark(
            video_paths=videos,
            durations=durations,
            server_url=args.server_url,
            concurrency=conc,
            duration_seconds=args.duration_seconds,
        )
        print(
            f"[conc={conc:<2}] reqs={metrics['requests']:<4} | "
            f"thrpt={metrics['throughput']:.1f} req/s | "
            f"p95={metrics['p95'] * 1000:6.1f} ms | "
            f"p99={metrics['p99'] * 1000:6.1f} ms"
        )


if __name__ == "__main__":
    main()
