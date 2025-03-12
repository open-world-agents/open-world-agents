"""
A simple script to sanity check the screen capture's ability to capture updates in real time.

This script continuously updates the progress bar description with the current timestamp.
The purpose is to verify that screen capture tools correctly capture and refresh dynamic content.
"""

import datetime
import time

from tqdm import tqdm


def get_current_time() -> str:
    # return time.time_ns()
    return str(datetime.datetime.fromtimestamp(time.time_ns() / 1e9))


def main():
    pbar = tqdm()  # Create a progress bar

    while True:
        pbar.set_description(f"time: {get_current_time()}")  # Update progress bar description
        pbar.update()  # Increment progress bar counter
        time.sleep(0.001)  # Small delay to allow readable updates without excessive CPU usage


if __name__ == "__main__":
    main()
