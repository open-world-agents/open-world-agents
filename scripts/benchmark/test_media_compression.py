#!/usr/bin/env python3
"""
Video Compression Benchmark

This script compares the file sizes and compression ratios of different video storage formats:
1. Raw frames (uncompressed numpy arrays)
2. JPEG/PNG compressed frames
3. H.265 video compression

Usage:
    python scripts/benchmark/test_media_compression.py <video_path>
    python scripts/benchmark/test_media_compression.py --help

Example:
    python scripts/benchmark/test_media_compression.py tmp/example.mkv
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from owa.core.io import VideoReader, VideoWriter
except ImportError:
    print("Error: owa-core not available. Please install owa-core package.")
    exit(1)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def calculate_raw_frame_size(frame: np.ndarray) -> int:
    """Calculate the size of a raw frame in bytes."""
    return frame.nbytes


def compress_frame_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Compress frame using JPEG compression."""
    # Convert RGB to BGR for OpenCV
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Failed to encode frame as JPEG")
    return encoded.tobytes()


def compress_frame_png(frame: np.ndarray) -> bytes:
    """Compress frame using PNG compression."""
    # Convert RGB to BGR for OpenCV
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", bgr_frame)
    if not success:
        raise ValueError("Failed to encode frame as PNG")
    return encoded.tobytes()


class CompressionBenchmark:
    """Benchmark different compression methods for video data."""

    def __init__(self, video_path: str, max_frames: int = 100):
        """
        Initialize the benchmark.

        Args:
            video_path: Path to the input video file
            max_frames: Maximum number of frames to process (for performance)
        """
        self.video_path = Path(video_path)
        self.max_frames = max_frames
        self.frames: List[np.ndarray] = []
        self.video_info: Dict = {}

    def load_frames(self) -> None:
        """Load frames from the video file."""
        print(f"Loading frames from {self.video_path}...")

        with VideoReader(self.video_path) as reader:
            # Get video info
            container = reader.container
            video_stream = container.streams.video[0]
            self.video_info = {
                "width": video_stream.width,
                "height": video_stream.height,
                "fps": float(video_stream.average_rate),
                "duration": float(container.duration / 1_000_000) if container.duration else None,
                "total_frames": video_stream.frames if video_stream.frames else None,
                "codec": video_stream.codec.name,
                "pixel_format": video_stream.pix_fmt,
            }

            # Load frames
            frame_count = 0
            for frame in tqdm(reader.read_frames(), desc="Loading frames"):
                if frame_count >= self.max_frames:
                    break

                # Convert to RGB numpy array
                rgb_array = frame.to_ndarray(format="rgb24")
                self.frames.append(rgb_array)
                frame_count += 1

        print(f"Loaded {len(self.frames)} frames")
        print(f"Video info: {self.video_info['width']}x{self.video_info['height']} @ {self.video_info['fps']:.2f} FPS")

    def benchmark_raw_storage(self) -> Tuple[int, float]:
        """Benchmark raw frame storage (uncompressed)."""
        print("\n=== Raw Storage Benchmark ===")

        start_time = time.time()
        total_size = 0

        for frame in tqdm(self.frames, desc="Calculating raw sizes"):
            total_size += calculate_raw_frame_size(frame)

        elapsed_time = time.time() - start_time

        print(f"Raw storage total size: {format_file_size(total_size)}")
        print(f"Average per frame: {format_file_size(total_size // len(self.frames))}")
        print(f"Processing time: {elapsed_time:.2f} seconds")

        return total_size, elapsed_time

    def benchmark_jpeg_compression(self, quality: int = 85) -> Tuple[int, float]:
        """Benchmark JPEG compression."""
        print(f"\n=== JPEG Compression Benchmark (Quality: {quality}) ===")

        start_time = time.time()
        total_size = 0

        for frame in tqdm(self.frames, desc="Compressing with JPEG"):
            compressed_data = compress_frame_jpeg(frame, quality)
            total_size += len(compressed_data)

        elapsed_time = time.time() - start_time

        print(f"JPEG compressed total size: {format_file_size(total_size)}")
        print(f"Average per frame: {format_file_size(total_size // len(self.frames))}")
        print(f"Processing time: {elapsed_time:.2f} seconds")

        return total_size, elapsed_time

    def benchmark_png_compression(self) -> Tuple[int, float]:
        """Benchmark PNG compression."""
        print("\n=== PNG Compression Benchmark ===")

        start_time = time.time()
        total_size = 0

        for frame in tqdm(self.frames, desc="Compressing with PNG"):
            compressed_data = compress_frame_png(frame)
            total_size += len(compressed_data)

        elapsed_time = time.time() - start_time

        print(f"PNG compressed total size: {format_file_size(total_size)}")
        print(f"Average per frame: {format_file_size(total_size // len(self.frames))}")
        print(f"Processing time: {elapsed_time:.2f} seconds")

        return total_size, elapsed_time

    def benchmark_h265_compression(self, crf: int = 23) -> Tuple[int, float]:
        """Benchmark H.265 video compression using VideoWriter."""
        print(f"\n=== H.265 Video Compression Benchmark (CRF: {crf}) ===")

        start_time = time.time()

        # Create temporary file for H.265 video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Write frames to H.265 video
            with VideoWriter(
                temp_path, fps=self.video_info.get("fps", 30.0), codec="libx265", options={"crf": str(crf)}
            ) as writer:
                for frame in tqdm(self.frames, desc="Encoding H.265 video"):
                    writer.write_frame(frame)

            # Get file size
            total_size = temp_path.stat().st_size
            elapsed_time = time.time() - start_time

            print(f"H.265 video file size: {format_file_size(total_size)}")
            print(f"Average per frame: {format_file_size(total_size // len(self.frames))}")
            print(f"Processing time: {elapsed_time:.2f} seconds")

            return total_size, elapsed_time

        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

    def get_original_video_size(self) -> int:
        """Get the size of the original video file."""
        return self.video_path.stat().st_size

    def run_full_benchmark(self) -> Dict:
        """Run all compression benchmarks and return results."""
        print(f"Starting compression benchmark for {self.video_path}")
        print(f"Processing {len(self.frames)} frames")

        # Load frames first
        if not self.frames:
            self.load_frames()

        # Get original video size
        original_size = self.get_original_video_size()

        # Raw storage (baseline for the frames we're processing)
        raw_size, raw_time = self.benchmark_raw_storage()

        # Calculate what the raw size would be for the entire video
        total_frames = self.video_info.get("total_frames")
        if total_frames is None:
            # Estimate total frames from duration and fps
            duration = self.video_info.get("duration", 0)
            fps = self.video_info.get("fps", 30)
            total_frames = int(duration * fps) if duration else len(self.frames)

        # Scale raw size to match entire video
        frames_processed = len(self.frames)
        estimated_full_raw_size = (raw_size / frames_processed) * total_frames

        results = {
            "raw_sample": {
                "size": raw_size,
                "time": raw_time,
                "compression_ratio": 1.0,  # Baseline for sample
                "frames": frames_processed,
            },
            "raw_full_video_estimate": {
                "size": estimated_full_raw_size,
                "time": 0.0,
                "compression_ratio": 1.0,  # Baseline for full video
                "frames": total_frames,
            },
            "original_video": {
                "size": original_size,
                "time": 0.0,
                "compression_ratio": estimated_full_raw_size / original_size,
                "note": "Original H.265 video file",
            },
        }

        # JPEG compression (multiple qualities) - compared to sample raw
        for quality in [50, 75, 85, 95]:
            jpeg_size, jpeg_time = self.benchmark_jpeg_compression(quality)
            results[f"jpeg_q{quality}"] = {
                "size": jpeg_size,
                "time": jpeg_time,
                "compression_ratio": raw_size / jpeg_size,
                "frames": frames_processed,
            }

        # PNG compression - compared to sample raw
        png_size, png_time = self.benchmark_png_compression()
        results["png"] = {
            "size": png_size,
            "time": png_time,
            "compression_ratio": raw_size / png_size,
            "frames": frames_processed,
        }

        # H.265 compression (multiple CRF values) - compared to sample raw
        for crf in [18, 23, 28]:
            try:
                h265_size, h265_time = self.benchmark_h265_compression(crf)
                results[f"h265_crf{crf}"] = {
                    "size": h265_size,
                    "time": h265_time,
                    "compression_ratio": raw_size / h265_size,
                    "frames": frames_processed,
                }
            except Exception as e:
                print(f"Warning: H.265 CRF {crf} benchmark failed: {e}")
                results[f"h265_crf{crf}"] = {"size": 0, "time": 0.0, "compression_ratio": 0.0, "error": str(e)}

        return results


def print_results_table(results: Dict) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("COMPRESSION BENCHMARK RESULTS")
    print("=" * 80)

    # Header
    print(f"{'Format':<20} {'Size':<12} {'Ratio':<8} {'Time (s)':<10} {'Notes':<20}")
    print("-" * 80)

    # Sort results by compression ratio (descending), excluding special entries
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if k not in ["original_video", "raw_full_video_estimate"]],
        key=lambda x: x[1].get("compression_ratio", 0),
        reverse=True,
    )

    # Show full video comparison first
    if "raw_full_video_estimate" in results and "original_video" in results:
        raw_full = results["raw_full_video_estimate"]
        orig = results["original_video"]
        print(f"{'FULL VIDEO COMPARISON':<40}")
        frames_note = f"({raw_full['frames']} frames)"
        print(
            f"{'Raw (estimated)':<20} {format_file_size(raw_full['size']):<12} {'1.0x':<8} {'-':<10} {frames_note:<20}"
        )
        ratio_str = f"{orig['compression_ratio']:.1f}x"
        print(
            f"{'Original Video':<20} {format_file_size(orig['size']):<12} {ratio_str:<8} {'-':<10} {'(H.265 compressed)':<20}"
        )
        print("-" * 80)
        sample_frames = results.get("raw_sample", {}).get("frames", 0)
        print(f"{'SAMPLE COMPRESSION (' + str(sample_frames) + ' frames)':<40}")
        print("-" * 80)

    # Print sorted results
    for format_name, data in sorted_results:
        if "error" in data:
            print(f"{format_name:<20} {'ERROR':<12} {'-':<8} {'-':<10} {data['error'][:20]:<20}")
        else:
            size_str = format_file_size(data["size"])
            ratio_str = f"{data['compression_ratio']:.1f}x"
            time_str = f"{data['time']:.2f}"
            print(f"{format_name:<20} {size_str:<12} {ratio_str:<8} {time_str:<10} {'':<20}")

    print("=" * 80)

    # Summary statistics
    valid_results = {
        k: v for k, v in results.items() if "error" not in v and k not in ["original_video", "raw_full_video_estimate"]
    }
    if valid_results:
        best_compression = max(valid_results.items(), key=lambda x: x[1]["compression_ratio"])
        fastest_encoding = min(valid_results.items(), key=lambda x: x[1]["time"])

        print("\nSUMMARY:")
        print(f"Best compression ratio: {best_compression[0]} ({best_compression[1]['compression_ratio']:.1f}x)")
        print(f"Fastest encoding: {fastest_encoding[0]} ({fastest_encoding[1]['time']:.2f}s)")

        # Calculate space savings compared to sample raw
        raw_size = results.get("raw_sample", {}).get("size", 0)
        if raw_size > 0:
            print(f"\nSpace savings compared to raw sample ({results.get('raw_sample', {}).get('frames', 0)} frames):")
            for name, data in valid_results.items():
                if name != "raw_sample":
                    savings_percent = (1 - data["size"] / raw_size) * 100
                    print(f"  {name}: {savings_percent:.1f}% smaller")

        # Show full video comparison
        if "raw_full_video_estimate" in results and "original_video" in results:
            full_raw_size = results["raw_full_video_estimate"]["size"]
            orig_size = results["original_video"]["size"]
            full_savings = (1 - orig_size / full_raw_size) * 100
            print(f"\nFull video: Original H.265 is {full_savings:.1f}% smaller than estimated raw")


def save_results_to_json(results: Dict, output_path: str) -> None:
    """Save benchmark results to JSON file."""
    output_file = Path(output_path)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert results to JSON-serializable format
    json_results = {}
    for method, data in results.items():
        json_results[method] = {
            "size_bytes": data["size"],
            "size_human": format_file_size(data["size"]),
            "compression_ratio": data["compression_ratio"],
            "time_seconds": data["time"],
        }

        # Add optional fields
        if "frames" in data:
            json_results[method]["frames"] = data["frames"]
        if "note" in data:
            json_results[method]["note"] = data["note"]
        if "error" in data:
            json_results[method]["error"] = data["error"]

    # Save to file
    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)


def main():
    """Main function to run the benchmark."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark video compression methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark/test_media_compression.py tmp/example.mkv
  python scripts/benchmark/test_media_compression.py tmp/example.mkv --max-frames 50
  python scripts/benchmark/test_media_compression.py tmp/example.mkv --max-frames 200
        """,
    )

    parser.add_argument("video_path", help="Path to the input video file")

    parser.add_argument(
        "--max-frames", type=int, default=100, help="Maximum number of frames to process (default: 100)"
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="tmp/compression_results.json",
        help="Path to save JSON results (default: tmp/compression_results.json)",
    )

    args = parser.parse_args()

    # Validate input file
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    # Run benchmark
    try:
        benchmark = CompressionBenchmark(args.video_path, max_frames=args.max_frames)
        benchmark.load_frames()

        print("\nVideo Information:")
        for key, value in benchmark.video_info.items():
            print(f"  {key}: {value}")

        results = benchmark.run_full_benchmark()
        print_results_table(results)

        # Save results to JSON
        save_results_to_json(results, args.output_json)
        print(f"\nResults saved to: {args.output_json}")

        return 0

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
