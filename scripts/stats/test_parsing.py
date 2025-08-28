#!/usr/bin/env python3

import subprocess
import re
import os


def parse_owl_mcap_info(output: str):
    """Parse the output of 'owl mcap info' command to extract topic message counts."""
    result = {
        "topics": {},
        "total_messages": 0,
        "duration": "",
    }

    print("Raw output:")
    print(repr(output))
    print("\n" + "=" * 50 + "\n")

    # Join all lines and then split again to handle line breaks in the middle of entries
    clean_output = " ".join(output.strip().split())
    print("Clean output:")
    print(clean_output)
    print("\n" + "=" * 50 + "\n")

    # Extract total messages
    match = re.search(r"messages:\s+(\d+)", clean_output)
    if match:
        result["total_messages"] = int(match.group(1))
        print(f"Found total messages: {result['total_messages']}")

    # Extract duration
    match = re.search(r"duration:\s+([^\s]+)", clean_output)
    if match:
        result["duration"] = match.group(1)
        print(f"Found duration: {result['duration']}")

    # Extract channels section
    channels_match = re.search(r"channels:(.*?)channels:\s*\d+", clean_output, re.DOTALL)
    if channels_match:
        channels_text = channels_match.group(1)
        print(f"Channels text: {channels_text}")
        print("\n" + "=" * 30 + "\n")

        # Find all channel entries
        # Format: (1) topic_name 12345 msgs (123.45 Hz) : message_type
        channel_pattern = r"\((\d+)\)\s+(\S+)\s+(\d+)\s+msgs\s+\([^)]+\)\s*:\s*([^\(]+?)(?=\(\d+\)|$)"

        for match in re.finditer(channel_pattern, channels_text):
            topic_name = match.group(2)
            message_count = int(match.group(3))
            message_type = match.group(4).strip()

            print(f"Found topic: {topic_name}, count: {message_count}, type: {message_type}")

            result["topics"][topic_name] = {"message_count": message_count, "message_type": message_type}
    else:
        print("No channels section found")

    return result


# Test with actual command
cmd = [
    "owl",
    "mcap",
    "info",
    "/mnt/raid12/datasets/owa_game_dataset/milkclouds00@gmail.com/apex_legends/apex_0811_02.mcap",
]
result = subprocess.run(
    cmd, capture_output=True, text=True, timeout=30, env=dict(os.environ, PATH=os.environ.get("PATH", ""))
)

if result.returncode == 0:
    parsed = parse_owl_mcap_info(result.stdout)
    print("\nFinal result:")
    print(parsed)
else:
    print(f"Command failed: {result.stderr}")
