#!/usr/bin/env python3
"""
Simple mouse monitoring script - minimal version.
Compares raw mouse vs standard mouse with tqdm progress and selective printing.
"""

import time
from threading import Lock

from tqdm import tqdm

from owa.core import LISTENERS
from owa.msgs.desktop.mouse import MouseEvent, RawMouseEvent

# Stats tracking
stats_lock = Lock()
stats = {
    "raw_count": 0,
    "std_count": 0,
    "raw_move_count": 0,
    "std_move_count": 0,
    "start_time": time.time(),
    "raw_velocity": 0,
    "std_velocity": 0,
}

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

# Velocity tracking with time window
VELOCITY_WINDOW = 1  # Time window for velocity calculation (seconds)
raw_velocity_data = {"last_time": None, "last_update": None, "movement_buffer": []}
std_velocity_data = {
    "last_time": None,
    "last_x": 0,
    "last_y": 0,
    "last_update": None,
    "movement_buffer": [],
}

# Control flags
should_quit = False


def update_movement_buffer_and_velocity(movement_buffer, current_time, dx=None, dy=None):
    """
    Helper function to update movement buffer and calculate velocity.

    Args:
        movement_buffer: List of movement entries with 'time', 'dx', 'dy' keys
        current_time: Current timestamp
        dx: Delta X movement (optional, for adding new movement)
        dy: Delta Y movement (optional, for adding new movement)

    Returns:
        tuple: (updated_buffer, velocity, velocity_string)
    """
    # Add current movement to buffer if provided
    if dx is not None and dy is not None:
        movement_buffer.append({"time": current_time, "dx": dx, "dy": dy})

    # Remove old entries outside the velocity window
    cutoff_time = current_time - VELOCITY_WINDOW
    updated_buffer = [entry for entry in movement_buffer if entry["time"] > cutoff_time]

    # Calculate velocity over the window
    velocity = 0
    velocity_str = ""

    if len(updated_buffer) >= 2:
        # Sum all movements in the window
        total_dx = sum(entry["dx"] for entry in updated_buffer)
        total_dy = sum(entry["dy"] for entry in updated_buffer)

        # Calculate velocity over the window
        velocity = ((total_dx / VELOCITY_WINDOW) ** 2 + (total_dy / VELOCITY_WINDOW) ** 2) ** 0.5
        velocity_str = f" vel={velocity:.0f}px/s"

    return updated_buffer, velocity, velocity_str


def on_raw_mouse(event: RawMouseEvent):
    """Raw mouse event handler."""
    global verbose_mode, raw_total, raw_velocity_data

    with stats_lock:
        stats["raw_count"] += 1

    # Update running totals
    raw_total["dx"] += event.dx
    raw_total["dy"] += event.dy

    # Calculate velocity using movement buffer
    current_time = time.time()

    # Update movement buffer and calculate velocity
    raw_velocity_data["movement_buffer"], velocity, velocity_str = update_movement_buffer_and_velocity(
        raw_velocity_data["movement_buffer"], current_time, event.dx, event.dy
    )

    # Update stats with calculated velocity
    if velocity > 0:
        with stats_lock:
            stats["raw_velocity"] = velocity

    raw_velocity_data["last_time"] = current_time
    raw_velocity_data["last_update"] = current_time

    # Verbose mode - print all events
    if verbose_mode:
        tqdm.write(f"RAW: {event}{velocity_str}")


def on_std_mouse(event: MouseEvent):
    """Standard mouse event handler."""
    global verbose_mode, std_total, std_velocity_data

    with stats_lock:
        stats["std_count"] += 1

    # Update running totals for move events
    if event.event_type == "move":
        std_total["x"] = event.x  # Absolute position, not delta
        std_total["y"] = event.y

        # Calculate velocity for move events using movement buffer
        current_time = time.time()

        # Convert position to movement and update buffer
        if std_velocity_data["last_time"] is not None:
            dx = event.x - std_velocity_data["last_x"]
            dy = event.y - std_velocity_data["last_y"]

            # Update movement buffer and calculate velocity
            std_velocity_data["movement_buffer"], velocity, velocity_str = update_movement_buffer_and_velocity(
                std_velocity_data["movement_buffer"], current_time, dx, dy
            )

            # Update stats with calculated velocity
            if velocity > 0:
                with stats_lock:
                    stats["std_velocity"] = velocity
        else:
            # First movement, just clean the buffer
            std_velocity_data["movement_buffer"], _, velocity_str = update_movement_buffer_and_velocity(
                std_velocity_data["movement_buffer"], current_time
            )

        std_velocity_data["last_time"] = current_time
        std_velocity_data["last_x"] = event.x
        std_velocity_data["last_y"] = event.y
        std_velocity_data["last_update"] = current_time

        # Verbose mode - print all events
        if verbose_mode:
            tqdm.write(f"STD: x={event.x:4d} y={event.y:4d} type={event.event_type}{velocity_str}")
    elif verbose_mode:
        # Print non-move events too
        tqdm.write(f"STD: {event.event_type} button={event.button} pressed={event.pressed}")


def apply_velocity_decay():
    """Apply velocity decay by cleaning old entries from movement buffers."""
    current_time = time.time()

    # Clean old entries from raw buffer and recalculate velocity
    raw_velocity_data["movement_buffer"], raw_velocity, _ = update_movement_buffer_and_velocity(
        raw_velocity_data["movement_buffer"], current_time
    )

    with stats_lock:
        stats["raw_velocity"] = raw_velocity

    # Clean old entries from std buffer and recalculate velocity
    std_velocity_data["movement_buffer"], std_velocity, _ = update_movement_buffer_and_velocity(
        std_velocity_data["movement_buffer"], current_time
    )

    with stats_lock:
        stats["std_velocity"] = std_velocity


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
            time.sleep(0.1)  # Update every 100ms for smoother velocity decay

            # Apply velocity decay
            apply_velocity_decay()

            with stats_lock:
                elapsed = time.time() - stats["start_time"]
                raw_fps = stats["raw_count"] / elapsed if elapsed > 0 else 0
                std_fps = stats["std_count"] / elapsed if elapsed > 0 else 0

                # Calculate velocity ratio
                velocity_ratio = 0
                if stats["raw_velocity"] > 0:
                    velocity_ratio = stats["std_velocity"] / stats["raw_velocity"]

                pbar.set_description(
                    f"Raw: {raw_fps:5.1f}Hz ({stats['raw_velocity']:.0f}px/s) | Std: {std_fps:5.1f}Hz ({stats['std_velocity']:.0f}px/s) | Ratio: {velocity_ratio:.2f} | Total: R{stats['raw_count']} S{stats['std_count']}"
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
