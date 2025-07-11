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

# Verbose mode - print all events
verbose_mode = False

# Summary mode - periodic dx/dy totals
summary_mode = False
summary_start_time = 0
summary_start_raw = {"dx": 0, "dy": 0}
summary_start_std = {"x": 0, "y": 0}
SUMMARY_INTERVAL = 2.0  # Print summary every 2 seconds

# Running totals
raw_total = {"dx": 0, "dy": 0}
std_total = {"x": 0, "y": 0}

# Control flags
should_quit = False


def on_raw_mouse(event):
    """Raw mouse event handler."""
    global verbose_mode, raw_total

    with stats_lock:
        stats["raw_count"] += 1

    # Update running totals
    raw_total["dx"] += event.dx
    raw_total["dy"] += event.dy

    # Verbose mode - print all events
    if verbose_mode:
        tqdm.write(f"RAW: {event}")


def on_std_mouse(event):
    """Standard mouse event handler."""
    global verbose_mode, std_total

    with stats_lock:
        stats["std_count"] += 1

    # Update running totals for move events
    if event.event_type == "move":
        std_total["x"] = event.x  # Absolute position, not delta
        std_total["y"] = event.y

        # Verbose mode - print all events
        if verbose_mode:
            tqdm.write(f"STD: x={event.x:4d} y={event.y:4d} type={event.event_type}")
    elif verbose_mode:
        # Print non-move events too
        tqdm.write(f"STD: {event.event_type} button={event.button} pressed={event.pressed}")


def toggle_verbose():
    """Toggle verbose mode - print all events."""
    global verbose_mode
    verbose_mode = not verbose_mode
    if verbose_mode:
        tqdm.write("--- Verbose mode ON (all events) ---")
    else:
        tqdm.write("--- Verbose mode OFF ---")


def toggle_summary():
    """Toggle summary mode - periodic dx/dy totals."""
    global summary_mode, summary_start_time, summary_start_raw, summary_start_std, raw_total, std_total

    summary_mode = not summary_mode
    if summary_mode:
        # Start summary mode
        summary_start_time = time.time()
        summary_start_raw["dx"] = raw_total["dx"]
        summary_start_raw["dy"] = raw_total["dy"]
        summary_start_std["x"] = std_total["x"]
        summary_start_std["y"] = std_total["y"]
        tqdm.write(f"--- Summary mode ON (every {SUMMARY_INTERVAL}s) ---")

        # Start summary timer thread
        import threading

        def summary_timer():
            # Keep track of last interval values for periodic reset
            last_raw = {"dx": summary_start_raw["dx"], "dy": summary_start_raw["dy"]}
            last_std = {"x": summary_start_std["x"], "y": summary_start_std["y"]}

            while summary_mode:
                time.sleep(SUMMARY_INTERVAL)
                if summary_mode:  # Check again in case it was turned off
                    # Calculate movement since last interval
                    current_raw_dx = raw_total["dx"] - last_raw["dx"]
                    current_raw_dy = raw_total["dy"] - last_raw["dy"]
                    current_std_dx = std_total["x"] - last_std["x"]
                    current_std_dy = std_total["y"] - last_std["y"]

                    elapsed = time.time() - summary_start_time
                    tqdm.write(
                        f"SUMMARY ({elapsed:.1f}s): RAW dx={current_raw_dx:6d} dy={current_raw_dy:6d} | STD dx={current_std_dx:6d} dy={current_std_dy:6d}"
                    )

                    # Update last values for next interval
                    last_raw["dx"] = raw_total["dx"]
                    last_raw["dy"] = raw_total["dy"]
                    last_std["x"] = std_total["x"]
                    last_std["y"] = std_total["y"]

        timer_thread = threading.Thread(target=summary_timer, daemon=True)
        timer_thread.start()
    else:
        tqdm.write("--- Summary mode OFF ---")


def main():
    """Main function."""
    print("Simple Mouse Monitor")
    print("===================")
    print("Type 'v' + Enter to toggle verbose mode (all events)")
    print("Type 's' + Enter to toggle summary mode (periodic dx/dy totals)")
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
                    if cmd == "v":
                        toggle_verbose()
                    elif cmd == "s":
                        toggle_summary()
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
