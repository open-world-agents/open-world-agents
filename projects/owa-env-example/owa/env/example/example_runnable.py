"""
Example runnable components for the example environment plugin.

This module demonstrates how to create runnable components that can
perform background tasks and processing.

Components are discovered via entry points and loaded lazily.
No decorators needed - registration happens via plugin_spec.
"""

import time
from pathlib import Path
from threading import Event

from owa.core import Runnable


class ExampleRunnable(Runnable):
    """
    Example runnable that performs a simple background task.

    This runnable demonstrates the basic pattern for creating background
    processes that can run independently while respecting stop signals.
    """

    def on_configure(self, *, interval: float = 1.0, output_file: str = "example_output.txt"):
        """
        Configure the runnable.

        Args:
            interval: Time between operations in seconds
            output_file: File to write output to
        """
        self.interval = interval
        self.output_file = Path(output_file)
        self.counter = 0

    def loop(self, *, stop_event: Event):
        """
        Main runnable loop that performs periodic work.

        Args:
            stop_event: Event to signal when to stop
        """
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, "w") as f:
            f.write("Example Runnable Output\n")
            f.write("=" * 30 + "\n")

            while not stop_event.is_set():
                # Perform some work
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                message = f"[{timestamp}] Task #{self.counter} completed\n"

                f.write(message)
                f.flush()  # Ensure data is written immediately

                self.counter += 1

                # Wait for the specified interval or until stop is requested
                stop_event.wait(self.interval)


class ExampleCounterRunnable(Runnable):
    """
    Example counter runnable that counts up to a maximum value.

    This demonstrates a runnable that stops itself after completing its task.
    """

    def on_configure(self, *, max_count: int = 10, interval: float = 0.5):
        """
        Configure the counter.

        Args:
            max_count: Maximum number to count to
            interval: Time between counts in seconds
        """
        self.max_count = max_count
        self.interval = interval

    def loop(self, *, stop_event: Event):
        """
        Counter loop that counts up to max_count.

        Args:
            stop_event: Event to signal when to stop
        """
        for i in range(self.max_count):
            if stop_event.is_set():
                break

            print(f"[Example Counter] Count: {i + 1}/{self.max_count}")

            # Wait for the interval or until stop is requested
            if stop_event.wait(self.interval):
                break  # Stop was requested

        print("[Example Counter] Counting completed!")
