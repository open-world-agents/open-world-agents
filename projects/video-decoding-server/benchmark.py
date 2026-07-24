#!/usr/bin/env python3
"""
Benchmark extract_frame_api over multiple videos with random PTS sampling.
Measures latencies, throughput, and bitrate for each concurrency level.

Supports both threading and multiprocessing modes. Multiprocessing mode overcomes
Python's GIL limitations to achieve high throughput (>20 Gbps) similar to perf_analyzer.
"""

import argparse
import multiprocessing as mp
import random
import threading
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple, Union

import cv2
import numpy as np
import tritonclient.http as httpclient


class RequestResult(NamedTuple):
    """Result of a single API request."""

    latency: float
    response_size: int


def extract_frames_api(
    requests: Union[Tuple[Union[str, Path], float], List[Tuple[Union[str, Path], float]]],
    client: httpclient.InferenceServerClient,
) -> RequestResult:
    """
    Send frame-extraction request(s) and return timing/size metrics.

    Args:
        requests: Single (video_path, pts) tuple or list of tuples
        client: Triton client instance

    Returns:
        RequestResult containing latency and total response size.
    """
    # Normalize to list
    is_single = not isinstance(requests, list)
    if is_single:
        requests = [requests]

    if not requests:
        return RequestResult(0.0, 0)

    start_time = time.perf_counter()
    batch_size = len(requests)

    # Prepare batch inputs
    video_paths = [str(req[0]) for req in requests]
    time_secs = [req[1] for req in requests]

    inputs = [
        httpclient.InferInput("video_path", [batch_size, 1], "BYTES"),
        httpclient.InferInput("time_sec", [batch_size, 1], "FP32"),
    ]

    # Convert to numpy arrays
    # Convert to numpy arrays and reshape
    video_path_data = np.array([path.encode() for path in video_paths], dtype=np.object_).reshape(batch_size, 1)
    time_sec_data = np.array(time_secs, dtype=np.float32).reshape(batch_size, 1)

    inputs[0].set_data_from_numpy(video_path_data)
    inputs[1].set_data_from_numpy(time_sec_data)

    outputs = [httpclient.InferRequestedOutput("frame")]
    response = client.infer("video_decoder", inputs=inputs, outputs=outputs)

    # Calculate total response size
    batch_frames = response.as_numpy("frame")
    total_response_size = batch_frames.nbytes if batch_frames is not None else 0

    latency = time.perf_counter() - start_time
    return RequestResult(latency, total_response_size)


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


def multiprocess_worker(
    video_paths: List[Path],
    durations: Dict[Path, float],
    server_url: str,
    duration_seconds: float,
    result_queue: mp.Queue,
    worker_id: int,
) -> None:
    """
    Worker function for multiprocessing benchmark.

    Args:
        video_paths: List of video file paths
        durations: Dictionary mapping video paths to their durations
        server_url: Triton server URL
        duration_seconds: How long to run the benchmark
        result_queue: Queue to store results
        worker_id: Unique identifier for this worker process
    """
    # Each process needs its own client instance
    client = httpclient.InferenceServerClient(url=server_url)
    end_time = time.perf_counter() + duration_seconds
    local_results = []

    try:
        while time.perf_counter() < end_time:
            video = random.choice(video_paths)
            pts = random.random() * durations[video]

            try:
                result = extract_frames_api((video, pts), client)

                # Only count results that completed before end_time
                if time.perf_counter() < end_time:
                    local_results.append(result)
            except Exception:
                # Skip failed requests
                pass

        # Put all results from this worker into the queue
        result_queue.put(local_results)

    except Exception as e:
        # Put error information in queue for debugging
        result_queue.put(f"Worker {worker_id} error: {e}")


def threading_worker(
    video_paths: List[Path],
    durations: Dict[Path, float],
    server_url: str,
    end_time: float,
    results: List[RequestResult],
    lock: threading.Lock,
) -> None:
    """
    Worker function for threading benchmark (legacy mode).
    """
    # NOTE: none of httpclient method is thread-safe. Separate client is needed to be instantiated per thread.
    client = httpclient.InferenceServerClient(url=server_url)
    while time.perf_counter() < end_time:
        video = random.choice(video_paths)
        pts = random.random() * durations[video]

        try:
            result = extract_frames_api((video, pts), client)

            # Only count results that completed before end_time
            if time.perf_counter() < end_time:
                with lock:
                    results.append(result)
        except Exception:
            # Skip failed requests
            pass


def run_benchmark(
    video_paths: List[Path],
    durations: Dict[Path, float],
    server_url: str,
    concurrency: int,
    duration_seconds: float,
    use_multiprocessing: bool = True,
) -> BenchmarkMetrics:
    """
    Run benchmark at given concurrency level for fixed duration.

    Args:
        video_paths: List of video file paths
        durations: Dictionary mapping video paths to their durations
        server_url: Triton server URL
        concurrency: Number of concurrent workers
        duration_seconds: How long to run the benchmark
        use_multiprocessing: If True, use multiprocessing for higher throughput.
                           If False, use threading (legacy mode).

    Returns:
        BenchmarkMetrics with performance results
    """
    if use_multiprocessing:
        return _run_multiprocess_benchmark(video_paths, durations, server_url, concurrency, duration_seconds)
    else:
        return _run_threading_benchmark(video_paths, durations, server_url, concurrency, duration_seconds)


def _run_multiprocess_benchmark(
    video_paths: List[Path],
    durations: Dict[Path, float],
    server_url: str,
    concurrency: int,
    duration_seconds: float,
) -> BenchmarkMetrics:
    """Run benchmark using multiprocessing for high throughput."""
    # Use a manager to create a shared queue
    with mp.Manager() as manager:
        result_queue = manager.Queue()

        # Create and start worker processes
        processes = []
        for worker_id in range(concurrency):
            p = mp.Process(
                target=multiprocess_worker,
                args=(video_paths, durations, server_url, duration_seconds, result_queue, worker_id),
            )
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Collect results from all workers
        all_results = []
        while not result_queue.empty():
            worker_results = result_queue.get()
            if isinstance(worker_results, list):
                all_results.extend(worker_results)
            else:
                # This is an error message
                print(f"Warning: {worker_results}")

        if not all_results:
            raise RuntimeError("No successful requests completed during benchmark")

        return _calculate_metrics(all_results, duration_seconds)


def _run_threading_benchmark(
    video_paths: List[Path],
    durations: Dict[Path, float],
    server_url: str,
    concurrency: int,
    duration_seconds: float,
) -> BenchmarkMetrics:
    """Run benchmark using threading (legacy mode)."""
    results: List[RequestResult] = []
    lock = threading.Lock()
    end_time = time.perf_counter() + duration_seconds

    threads = [
        threading.Thread(target=threading_worker, args=(video_paths, durations, server_url, end_time, results, lock))
        for _ in range(concurrency)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if not results:
        raise RuntimeError("No successful requests completed during benchmark")

    return _calculate_metrics(results, duration_seconds)


def _calculate_metrics(results: List[RequestResult], duration_seconds: float) -> BenchmarkMetrics:
    """Calculate benchmark metrics from results."""
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
    parser = argparse.ArgumentParser(
        description="High-performance benchmark for extract_frame_api with random PTS sampling"
    )
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
        default="127.0.0.1:8000",
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
    parser.add_argument(
        "--use-threading",
        action="store_true",
        help="Use threading instead of multiprocessing (legacy mode with lower throughput)",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=None,
        help="Maximum number of processes to use (defaults to CPU count)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark with specified parameters."""
    args = parse_args()

    # Determine execution mode
    use_multiprocessing = not args.use_threading
    max_cpu_count = mp.cpu_count()

    if args.max_processes is not None:
        max_processes = min(args.max_processes, max_cpu_count)
    else:
        max_processes = max_cpu_count

    # Validate concurrency levels for multiprocessing
    if use_multiprocessing:
        validated_concurrency = []
        for concurrency in args.concurrency:
            if concurrency > max_processes:
                print(
                    f"Warning: Concurrency {concurrency} exceeds available processes ({max_processes}). "
                    f"Consider using --max-processes or --use-threading for higher concurrency."
                )
                validated_concurrency.append(max_processes)
            else:
                validated_concurrency.append(concurrency)
        args.concurrency = validated_concurrency

    print("Computing video durations...")
    durations = get_video_durations(args.video_list)

    mode_str = "multiprocessing" if use_multiprocessing else "threading"
    print(f"Running benchmark using {mode_str} for {args.duration_seconds}s each:")
    if use_multiprocessing:
        print(f"Available CPU cores: {max_cpu_count}, Max processes: {max_processes}")

    print("Concurrency | Requests | Throughput | Bitrate  | P95 Latency | P99 Latency")
    print("-" * 75)

    for concurrency in args.concurrency:
        try:
            metrics = run_benchmark(
                video_paths=args.video_list,
                durations=durations,
                server_url=args.server_url,
                concurrency=concurrency,
                duration_seconds=args.duration_seconds,
                use_multiprocessing=use_multiprocessing,
            )
            print(
                f"{concurrency:>11} | {metrics.requests:>8} | "
                f"{metrics.throughput:>7.1f} r/s | {metrics.bitrate_mbps:>6.1f} Mbps | "
                f"{metrics.p95_ms:>8.1f} ms | {metrics.p99_ms:>8.1f} ms"
            )
        except Exception as e:
            print(f"{concurrency:>11} | ERROR: {e}")


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method("spawn", force=True)
    main()
