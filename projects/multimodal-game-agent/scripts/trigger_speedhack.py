import os
import tempfile
import time

# Get the path to the temp folder
temp_dir = tempfile.gettempdir()
file_path = os.path.join(temp_dir, "speedhack_trigger.txt")


def enable_speedhack():
    """Writes 'on' to the signal file to enable speedhack."""
    with open(file_path, "w") as f:
        f.write("on")


def disable_speedhack():
    """Writes 'off' to the signal file to disable speedhack."""
    with open(file_path, "w") as f:
        f.write("off")


if __name__ == "__main__":
    print(f"Speedhack signal file: {file_path}")

    # Example usage
    for i in range(5):
        print(f"Iteration {i + 1}")
        enable_speedhack()  # Uncomment to enable
        print("Speedhack ENABLED (wrote 'on')")
        time.sleep(0.3)
        disable_speedhack()  # Uncomment to disable
        print("Speedhack DISABLED (wrote 'off')")
        time.sleep(0.3)
