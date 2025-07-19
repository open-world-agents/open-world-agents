#!/usr/bin/env python3
"""
Benchmark extract_frame_api over multiple videos with random PTS sampling.
Measures latencies, throughput, and bitrate for each concurrency level.
"""

import argparse
import base64
import random
import threading
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Union

import cv2
import numpy as np
import requests


class RequestResult(NamedTuple):
    """Result of a single API request."""

    latency: float
    response_size: int


def extract_frame_api(
    video_path: Union[str, Path],
    pts: float,
    server_url: str = "http://127.0.0.1:8000",
) -> RequestResult:
    """
    Send a frame-extraction request and return timing/size metrics.

    Args:
        video_path: Path to the video file.
        pts: Timestamp in seconds.
        server_url: Base URL of the decoding server.

    Returns:
        RequestResult containing latency and response size.

    Raises:
        requests.RequestException: On network or HTTP errors.
        ValueError: On missing or undecodable frame data.
    """
    start_time = time.perf_counter()
    resp = requests.post(
        f"{server_url}/predict",
        json={"video_path": str(video_path), "pts": pts},
    )
    resp.raise_for_status()

    response_size = len(resp.content)
    data = resp.json()

    if "frame" not in data:
        raise ValueError("Missing 'frame' in response JSON")

    # Validate frame data without storing the actual frame
    img_bytes = base64.b64decode(data["frame"])
    img_bgr = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image bytes")

    latency = time.perf_counter() - start_time
    return RequestResult(latency, response_size)


def get_video_durations(video_paths: List[Path]) -> Dict[Path, float]:
    """Compute the duration (in seconds) of each video."""
    durations = {}
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


class BenchmarkMetrics(NamedTuple):
    """Benchmark results for a concurrency level."""

    requests: int
    throughput: float
    bitrate_mbps: float
    p95_ms: float
    p99_ms: float


def run_benchmark(
    video_paths: List[Path],
    durations: Dict[Path, float],
    server_url: str,
    concurrency: int,
    duration_seconds: float,
) -> BenchmarkMetrics:
    """Run benchmark at given concurrency level for fixed duration."""
    results: List[RequestResult] = []
    lock = threading.Lock()
    end_time = time.perf_counter() + duration_seconds

    def worker() -> None:
        while time.perf_counter() < end_time:
            video = random.choice(video_paths)
            pts = random.random() * durations[video]

            try:
                result = extract_frame_api(video, pts, server_url)

                # Only count results that completed before end_time
                if time.perf_counter() < end_time:
                    with lock:
                        results.append(result)
            except Exception:
                # Skip failed requests
                pass

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if not results:
        raise RuntimeError("No successful requests completed during benchmark")

    # Calculate metrics
    latencies = [r.latency for r in results]
    total_bytes = sum(r.response_size for r in results)

    throughput = len(results) / duration_seconds
    bitrate_mbps = (total_bytes * 8) / (duration_seconds * 1_000_000)  # Mbps
    p95_ms = float(np.percentile(latencies, 95) * 1000)
    p99_ms = float(np.percentile(latencies, 99) * 1000)

    return BenchmarkMetrics(
        requests=len(results),
        throughput=throughput,
        bitrate_mbps=bitrate_mbps,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark extract_frame_api with random PTS sampling")
    parser.add_argument(
        "--video-list",
        type=Path,
        nargs="+",
        required=True,
        help="List of video file paths to benchmark",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Decoding server base URL",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64],
        help="Concurrency levels to test",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=5.0,
        help="Benchmark duration per concurrency level",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark with specified parameters."""
    args = parse_args()

    print("Computing video durations...")
    durations = get_video_durations(args.video_list)

    print(f"Running benchmark for {args.duration_seconds}s each:")
    print("Concurrency | Requests | Throughput | Bitrate  | P95 Latency | P99 Latency")
    print("-" * 75)

    for concurrency in args.concurrency:
        metrics = run_benchmark(
            video_paths=args.video_list,
            durations=durations,
            server_url=args.server_url,
            concurrency=concurrency,
            duration_seconds=args.duration_seconds,
        )
        print(
            f"{concurrency:>11} | {metrics.requests:>8} | "
            f"{metrics.throughput:>7.1f} r/s | {metrics.bitrate_mbps:>6.1f} Mbps | "
            f"{metrics.p95_ms:>8.1f} ms | {metrics.p99_ms:>8.1f} ms"
        )


if __name__ == "__main__":
    main()
