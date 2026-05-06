import matplotlib.pyplot as plt  # Import matplotlib for plotting
import numpy as np

from mcap_owa.highlevel import OWAMcapReader
from owa.core.time import TimeUnits


def main():
    diffs = []
    pts_values = []

    with OWAMcapReader("tmp/clock.mcap") as reader:
        print(reader.topics)
        print(reader.start_time, reader.end_time)
        for mcap_msg in reader.iter_messages(topics=["screen"]):
            start_time = mcap_msg.timestamp  # 1
            # start_time = mcap_msg.decoded.utc_ns  # 2
            break

        for mcap_msg in reader.iter_messages(topics=["screen"]):
            print(f"Topic: {mcap_msg.topic}, Timestamp: {mcap_msg.timestamp}, Message: {mcap_msg.decoded}")
            diffs.append((mcap_msg.timestamp - start_time) - mcap_msg.decoded.pts)  # 1
            # diffs.append((mcap_msg.decoded.utc_ns - start_time) - mcap_msg.decoded.pts) # 2
            pts_values.append(mcap_msg.decoded.pts)

    # Convert to numpy arrays
    diffs = np.array(diffs) / TimeUnits.MSECOND  # Convert to milliseconds
    pts_values = np.array(pts_values) / TimeUnits.MSECOND  # Convert to milliseconds

    # Print statistics
    print(f"Mean: {np.mean(diffs):.2f}ms")
    print(f"Median: {np.median(diffs):.2f}ms")
    print(f"Q1: {np.percentile(diffs, 25):.2f}ms")
    print(f"Q3: {np.percentile(diffs, 75):.2f}ms")
    print(f"Max: {np.max(diffs):.2f}ms")
    print(f"Min: {np.min(diffs):.2f}ms")

    # Plot diff vs pts
    plt.figure(figsize=(10, 5))
    plt.scatter(pts_values, diffs, label="Diff vs PTS", alpha=0.6, s=10)
    plt.xlabel("PTS (ms)")
    plt.ylabel("Diff (ms)")
    plt.title("Difference (Timestamp - Start Time - PTS) vs PTS")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
