#!/usr/bin/env python3
"""
MCAP Topic Aggregator

This script uses `owl mcap info` command to extract message counts per topic from all MCAP files.
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from collections import defaultdict, Counter
import json
import argparse
from typing import Dict, List, Any
import logging


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def find_mcap_files(dataset_path: str) -> List[Path]:
    """Find all MCAP files in the dataset directory."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    mcap_files = list(dataset_path.rglob("*.mcap"))
    logging.info(f"Found {len(mcap_files)} MCAP files")
    return mcap_files


def parse_owl_mcap_info(output: str) -> Dict[str, Any]:
    """Parse the output of 'owl mcap info' command to extract topic message counts."""
    result = {
        "topics": {},
        "total_messages": 0,
        "duration": "",
    }

    # Join all lines and then split again to handle line breaks in the middle of entries
    clean_output = " ".join(output.strip().split())

    # Extract total messages
    match = re.search(r"messages:\s+(\d+)", clean_output)
    if match:
        result["total_messages"] = int(match.group(1))

    # Extract duration
    match = re.search(r"duration:\s+([^\s]+)", clean_output)
    if match:
        result["duration"] = match.group(1)

    # Extract channels section
    channels_match = re.search(r"channels:(.*?)channels:\s*\d+", clean_output, re.DOTALL)
    if channels_match:
        channels_text = channels_match.group(1)

        # Find all channel entries
        # Format: (1) topic_name 12345 msgs (123.45 Hz) : message_type
        channel_pattern = r"\((\d+)\)\s+(\S+)\s+(\d+)\s+msgs\s+\([^)]+\)\s*:\s*([^\(]+?)(?=\(\d+\)|$)"

        for match in re.finditer(channel_pattern, channels_text):
            topic_name = match.group(2)
            message_count = int(match.group(3))
            message_type = match.group(4).strip()

            result["topics"][topic_name] = {"message_count": message_count, "message_type": message_type}

    return result


def parse_mcap_file(file_path: Path) -> Dict[str, Any]:
    """Parse a single MCAP file using owl mcap info command."""
    try:
        # Run owl mcap info command
        cmd = ["owl", "mcap", "info", str(file_path)]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, env=dict(os.environ, PATH=os.environ.get("PATH", ""))
        )

        if result.returncode != 0:
            return {
                "file_path": str(file_path),
                "error": f"Command failed: {result.stderr.strip()}",
                "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
            }

        # Parse the output
        info = parse_owl_mcap_info(result.stdout)

        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size_bytes": file_path.stat().st_size,
            "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            "total_messages": info["total_messages"],
            "duration": info["duration"],
            "topics": info["topics"],
            "topic_count": len(info["topics"]),
        }

    except subprocess.TimeoutExpired:
        return {
            "file_path": str(file_path),
            "error": "Command timeout",
            "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        }
    except Exception as e:
        logging.error(f"Error parsing {file_path}: {e}")
        return {
            "file_path": str(file_path),
            "error": str(e),
            "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results from all MCAP files."""
    successful_results = [r for r in results if "error" not in r]
    failed_results = [r for r in results if "error" in r]

    if not successful_results:
        return {
            "total_files": len(results),
            "successful_files": 0,
            "failed_files": len(failed_results),
            "errors": failed_results,
        }

    # Aggregate statistics
    total_messages = sum(r["total_messages"] for r in successful_results)
    total_size_bytes = sum(r["file_size_bytes"] for r in successful_results)

    # Topic statistics - aggregate message counts per topic across all files
    topic_message_counts = Counter()
    topic_file_counts = Counter()  # How many files contain each topic

    for result in successful_results:
        for topic_name, topic_info in result["topics"].items():
            topic_message_counts[topic_name] += topic_info["message_count"]
            topic_file_counts[topic_name] += 1

    # File statistics
    file_sizes = [r["file_size_mb"] for r in successful_results]

    return {
        "summary": {
            "total_files": len(results),
            "successful_files": len(successful_results),
            "failed_files": len(failed_results),
            "total_messages": total_messages,
            "total_size_bytes": total_size_bytes,
            "total_size_gb": round(total_size_bytes / (1024**3), 2),
            "average_file_size_mb": round(sum(file_sizes) / len(file_sizes), 2) if file_sizes else 0,
        },
        "topic_statistics": {
            "unique_topics": len(topic_message_counts),
            "topic_message_counts": dict(topic_message_counts.most_common()),
            "topic_file_counts": dict(topic_file_counts.most_common()),
        },
        "file_details": successful_results,
        "errors": failed_results,
    }


def save_results(results: Dict[str, Any], output_file: str):
    """Save results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"Results saved to {output_file}")


def print_summary(results: Dict[str, Any]):
    """Print a summary of the aggregated results."""
    summary = results["summary"]
    topic_stats = results["topic_statistics"]

    print("\n" + "=" * 60)
    print("MCAP TOPIC AGGREGATION SUMMARY")
    print("=" * 60)

    print("\nFile Statistics:")
    print(f"  Total files processed: {summary['total_files']}")
    print(f"  Successful: {summary['successful_files']}")
    print(f"  Failed: {summary['failed_files']}")
    print(f"  Total size: {summary['total_size_gb']} GB")
    print(f"  Average file size: {summary['average_file_size_mb']} MB")

    print("\nMessage Statistics:")
    print(f"  Total messages: {summary['total_messages']:,}")

    print("\nTopic Statistics:")
    print(f"  Unique topics: {topic_stats['unique_topics']}")

    print("\nTopics by File Count:")
    for topic, count in list(topic_stats["topic_file_counts"].items()):
        print(f"  {topic}: {count} files")

    print("\nTopics by Message Count:")
    for topic, count in list(topic_stats["topic_message_counts"].items()):
        print(f"  {topic}: {count:,} messages")

    if results["errors"]:
        print(f"\nErrors encountered in {len(results['errors'])} files:")
        for error in results["errors"][:5]:  # Show first 5 errors
            print(f"  {Path(error['file_path']).name}: {error['error']}")
        if len(results["errors"]) > 5:
            print(f"  ... and {len(results['errors']) - 5} more errors")


def main():
    parser = argparse.ArgumentParser(description="Aggregate topic information from MCAP files")
    parser.add_argument(
        "--dataset-path",
        default="/mnt/raid12/datasets/owa_game_dataset",
        help="Path to the dataset directory containing MCAP files",
    )
    parser.add_argument("--output", default="mcap_topic_aggregation.json", help="Output JSON file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--limit", type=int, help="Limit the number of files to process (for testing)")

    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        # Find all MCAP files
        mcap_files = find_mcap_files(args.dataset_path)

        if args.limit:
            mcap_files = mcap_files[: args.limit]
            logging.info(f"Limited to first {args.limit} files")

        if not mcap_files:
            print("No MCAP files found!")
            return

        # Process each file
        results = []
        for i, file_path in enumerate(mcap_files, 1):
            logging.info(f"Processing {i}/{len(mcap_files)}: {file_path.name}")
            result = parse_mcap_file(file_path)
            results.append(result)

        # Aggregate results
        aggregated = aggregate_results(results)

        # Save results
        save_results(aggregated, args.output)

        # Print summary
        print_summary(aggregated)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
