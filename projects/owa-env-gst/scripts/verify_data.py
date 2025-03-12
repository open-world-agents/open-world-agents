import matplotlib.pyplot as plt  # Import matplotlib for plotting
import numpy as np

from mcap_owa.highlevel import OWAMcapReader


class TimeUnits:
    NSECOND = 1
    USECOND = 10**3
    MSECOND = 10**6
    SECOND = 10**9


def main():
    diffs = []
    pts_values = []

    with OWAMcapReader("tmp/clock.mcap") as reader:
        print(reader.topics)
        print(reader.start_time, reader.end_time)
        for topic, timestamp, msg in reader.iter_decoded_messages(topics=["screen"]):
            start_time = timestamp  # 1
            # start_time = msg.utc_ns  # 2
            break

        for topic, timestamp, msg in reader.iter_decoded_messages(topics=["screen"]):
            print(f"Topic: {topic}, Timestamp: {timestamp}, Message: {msg}")
            diffs.append((timestamp - start_time) - msg.pts)  # 1
            # diffs.append((msg.utc_ns - start_time) - msg.pts) # 2
            pts_values.append(msg.pts)

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
