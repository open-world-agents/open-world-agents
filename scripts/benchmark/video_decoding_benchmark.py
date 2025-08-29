import argparse
import time
from typing import Callable

import numpy as np

from owa.core.io.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder


def benchmark(fn: Callable, max_time: float = 3.0) -> None:
    """Benchmark a function, reporting mean/std/percentile statistics."""
    results = []
    start_time = time.time()
    while time.time() - start_time < max_time:
        now = time.time()
        fn()
        results.append(time.time() - now)
    print(
        f"Measured {len(results)} samples: Mean ± Std: {np.mean(results):.4f} ± {np.std(results):.4f} s | 95th: {np.percentile(results, 95):.4f} s"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark video decoding")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument(
        "--backend", "-b", choices=["torchcodec", "pyav"], default="torchcodec", help="Decoding backend"
    )
    args = parser.parse_args()

    VideoDecoder = TorchCodecVideoDecoder if args.backend == "torchcodec" else PyAVVideoDecoder
    decoder = VideoDecoder(args.video_path)

    print(f"Using backend: {args.backend}")
    print(f"Video: {args.video_path}")
    print(f"Metadata: {decoder.metadata}")
    print()

    # Simple Indexing API
    print(f"{decoder[0].shape=}, {decoder[0].dtype=}, {decoder[0].device=}")  # uint8 tensor of shape [C, H, W]
    # print(decoder[0:-1:20].shape)  # uint8 stacked tensor of shape [N, C, H, W]

    # # Indexing, with PTS and duration info:
    # print(decoder.get_frames_at(indices=[2, 100]))
    # # FrameBatch:
    # #   data (shape): torch.Size([2, 3, 270, 480])
    # #   pts_seconds: tensor([0.0667, 3.3367], dtype=torch.float64)
    # #   duration_seconds: tensor([0.0334, 0.0334], dtype=torch.float64)

    # # Time-based indexing with PTS and duration info
    # print(decoder.get_frames_played_at(seconds=[0.5, 10.4]))
    # # FrameBatch:
    # #   data (shape): torch.Size([2, 3, 270, 480])
    # #   pts_seconds: tensor([ 0.4671, 10.3770], dtype=torch.float64)
    # #   duration_seconds: tensor([0.0334, 0.0334], dtype=torch.float64)

    # Benchmark different access patterns
    print("\nBenchmarking different access patterns:")

    # Test patterns
    T1 = [[0.5, 0.6, 0.7, 0.8, 0.9], [10.4, 10.5, 10.6, 10.7, 10.8]]  # Two dense clusters
    T2 = [[0.5, 0.6, 0.7, 0.8, 0.9]]  # Single dense cluster
    T3 = [[0.5, 10.4, 20.8]]  # Sparse access
    T4 = [[0.5], [0.6], [0.7], [0.8], [0.9]]  # Multiple single accesses
    T5 = [[20.8], [20.7], [20.6], [20.5], [20.4]]  # Reverse order single accesses

    def benchmark_pattern(T, pattern_name):
        def f(T):
            for t in T:
                decoder = VideoDecoder(args.video_path)  # NOTE: comment this or not
                if args.backend == "pyav":
                    decoder.get_frames_played_at(seconds=t, strategy="sequential_per_keyframe_block")
                else:
                    decoder.get_frames_played_at(seconds=t)

        print(f"{pattern_name}")
        benchmark(lambda: f(T))
        print("")

    benchmark_pattern(T1, "Two dense clusters")
    benchmark_pattern(T2, "Single dense cluster")
    benchmark_pattern(T3, "Sparse access")
    benchmark_pattern(T4, "Multiple single accesses")
    benchmark_pattern(T5, "Reverse order accesses")


if __name__ == "__main__":
    main()
