#!/usr/bin/env python3
"""
Simple mouse monitoring script - minimal version.
Compares raw mouse vs standard mouse with tqdm progress and selective printing.
"""

import time
from threading import Lock

from tqdm import tqdm

from owa.core import LISTENERS

# Stats tracking
stats_lock = Lock()
stats = {"raw_count": 0, "std_count": 0, "raw_move_count": 0, "std_move_count": 0, "start_time": time.time()}

# Print control
print_events = False
print_start = 0
PRINT_DURATION = 3.0

# Control flags
should_quit = False


def on_raw_mouse(event):
    """Raw mouse event handler."""
    global print_events, print_start
    current_time = time.time()

    with stats_lock:
        stats["raw_count"] += 1

    # Print during print periods (all events for debugging)
    if print_events and (current_time - print_start) < PRINT_DURATION:
        tqdm.write(f"RAW: dx={event.dx:4d} dy={event.dy:4d} flags=0x{event.button_flags:04x}")
    elif print_events:
        print_events = False
        tqdm.write("--- Print ended ---")


def on_std_mouse(event):
    """Standard mouse event handler."""
    with stats_lock:
        stats["std_count"] += 1

    # Print during print periods (only moves)
    if print_events and (time.time() - print_start) < PRINT_DURATION:
        if event.event_type == "move":
            tqdm.write(f"STD: x={event.x:4d} y={event.y:4d}")


def enable_print():
    """Enable printing for PRINT_DURATION seconds."""
    global print_events, print_start
    print_events = True
    print_start = time.time()
    tqdm.write(f"--- Printing for {PRINT_DURATION}s ---")


def main():
    """Main function."""
    print("Simple Mouse Monitor")
    print("===================")
    print("Type 'p' + Enter to print events for 3 seconds")
    print("Type 'q' + Enter to quit")
    print()

    # Create listeners
    raw_listener = LISTENERS["desktop/raw_mouse"]()
    std_listener = LISTENERS["desktop/mouse"]()

    raw_listener.configure(callback=on_raw_mouse)
    std_listener.configure(callback=on_std_mouse)

    # Initialize progress bar
    pbar = None

    try:
        # Start listeners
        raw_listener.start()
        std_listener.start()
        print("✅ Listeners started. Move your mouse!")

        # Progress bar
        pbar = tqdm(desc="Raw:   0Hz | Std:   0Hz", unit="", bar_format="{desc}")

        # Input handling
        import threading

        def input_handler():
            global should_quit
            while not should_quit:
                try:
                    cmd = input().strip().lower()
                    if cmd == "p":
                        enable_print()
                    elif cmd == "q":
                        should_quit = True
                        break
                except (EOFError, KeyboardInterrupt):
                    should_quit = True
                    break

        input_thread = threading.Thread(target=input_handler, daemon=True)
        input_thread.start()

        # Main loop
        while not should_quit:
            time.sleep(0.5)  # Update every 500ms

            with stats_lock:
                elapsed = time.time() - stats["start_time"]
                raw_fps = stats["raw_count"] / elapsed if elapsed > 0 else 0
                std_fps = stats["std_count"] / elapsed if elapsed > 0 else 0

                pbar.set_description(
                    f"Raw: {raw_fps:5.1f}Hz | Std: {std_fps:5.1f}Hz | Total: R{stats['raw_count']} S{stats['std_count']}"
                )

    except KeyboardInterrupt:
        tqdm.write("\nStopping...")
    finally:
        if pbar is not None:
            pbar.close()
        raw_listener.stop()
        std_listener.stop()

        # Final stats
        with stats_lock:
            elapsed = time.time() - stats["start_time"]
            raw_fps = stats["raw_count"] / elapsed if elapsed > 0 else 0
            std_fps = stats["std_count"] / elapsed if elapsed > 0 else 0

        print(
            f"\nFinal: {elapsed:.1f}s | Raw: {raw_fps:.1f}Hz ({stats['raw_count']} events) | Std: {std_fps:.1f}Hz ({stats['std_count']} events)"
        )


if __name__ == "__main__":
    main()
