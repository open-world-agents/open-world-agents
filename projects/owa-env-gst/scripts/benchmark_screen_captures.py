#!/usr/bin/env python3

import os
import queue
import sys
import threading
import time

# Attempt optional imports for each capturing method
try:
    from PIL import ImageGrab

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from mss import mss

    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    import pyscreenshot as ImageGrab2

    PYSCREENSHOT_AVAILABLE = True
except ImportError:
    PYSCREENSHOT_AVAILABLE = False
    ImageGrab2 = None

try:
    from PyQt5.QtCore import QRect  # noqa
    from PyQt5.QtGui import QGuiApplication
    from PyQt5.QtWidgets import QApplication

    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

# Attempt OWA-based import
try:
    from owa.core.registry import LISTENERS, RUNNABLES

    OWA_AVAILABLE = True
except ImportError:
    OWA_AVAILABLE = False

# For CPU & memory usage
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# For GPU usage (NVIDIA only)
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


###############################################################################
# Helper: Resource Measurement
###############################################################################


class ResourceUsageMonitor:
    """
    Periodically samples CPU%, memory usage, and GPU usage (if available),
    storing them in a queue. On stop(), aggregates average usage.
    """

    def __init__(self, interval=0.1):
        """interval: sampling interval in seconds."""
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None

        self.gpu_handle = None
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                # By default, only look at GPU 0
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.gpu_handle = None

    def start(self):
        if not PSUTIL_AVAILABLE:
            print("WARNING: psutil not available; no CPU/memory usage will be measured.")
        if not PYNVML_AVAILABLE:
            print("WARNING: pynvml not available; GPU usage will not be measured.")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def _run(self):
        while not self.stop_event.is_set():
            cpu_percent = None
            mem_val = None
            gpu_percent = None

            if self.process:
                cpu_percent = self.process.cpu_percent(interval=None)
                mem_info = self.process.memory_info()
                mem_val = mem_info.rss / (1024 * 1024)  # Convert bytes to MB

            if self.gpu_handle:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_percent = util.gpu
                except Exception:
                    pass

            # Fallbacks if something is missing
            if cpu_percent is None:
                cpu_percent = 0.0
            if mem_val is None:
                mem_val = 0.0
            if gpu_percent is None:
                gpu_percent = 0.0

            self.samples.put((cpu_percent, mem_val, gpu_percent))
            time.sleep(self.interval)

        def get_averages(self):
            count = 0
            total_cpu = 0.0
            total_mem = 0.0
            total_gpu = 0.0

            while True:
                try:
                    cpu, mem, gpu = self.samples.get_nowait()
                    count += 1
                    total_cpu += cpu
                    total_mem += mem
                    total_gpu += gpu
                except queue.Empty:
                    break

            if count == 0:
                return (0.0, 0.0, 0.0)
            return (total_cpu / count, total_mem / count, total_gpu / count)

    def get_averages(self):
        count = 0
        total_cpu = 0.0
        total_mem = 0.0
        total_gpu = 0.0

        while True:
            try:
                cpu, mem, gpu = self.samples.get_nowait()
                count += 1
                total_cpu += cpu
                total_mem += mem
                total_gpu += gpu
            except queue.Empty:
                break

        if count == 0:
            return (0.0, 0.0, 0.0)
        return (total_cpu / count, total_mem / count, total_gpu / count)


###############################################################################
# Screen Capture Routines
###############################################################################

owa_args = dict(fps=240, window_name=None, monitor_idx=None)


def capture_owa_runnable():
    """
    Capture using the OWA "owa.env.gst" module in runnable mode.
    Warm-up for 2 seconds, then measure for 2 seconds.
    Returns: (number_of_frames, elapsed_time)
    """
    if not OWA_AVAILABLE:
        raise RuntimeError("OWA environment not available.")

    screen_capture = RUNNABLES["screen_capture"]()
    screen_capture.configure(**owa_args)
    screen_capture.start()

    # Warm-up: capture frames for 2 seconds
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        _ = screen_capture.grab()

    # Measurement: capture frames for 2 seconds
    start = time.time()
    count = 0
    while time.time() - start < 2.0:
        _ = screen_capture.grab()
        count += 1
    elapsed = time.time() - start

    screen_capture.stop()
    screen_capture.join()
    return (count, elapsed)


def capture_owa_listener():
    """
    Capture using the OWA "owa.env.gst" module in listener mode.
    Warm-up for 2 seconds, then measure for 2 seconds.
    Returns: (number_of_frames, elapsed_time)
    """
    if not OWA_AVAILABLE:
        raise RuntimeError("OWA environment not available.")

    count = 0

    def callback(x, proc):
        nonlocal count
        count += 1

    screen_capture = LISTENERS["screen"]()
    screen_capture.configure(**owa_args, callback=callback)
    screen_capture.start()

    # Warm-up period
    time.sleep(2)

    start_count = count
    time.sleep(2.0)
    frame_count = count - start_count

    screen_capture.stop()
    screen_capture.join()
    return (frame_count, 2.0)


def capture_pillow():
    """
    Capture using Pillow's ImageGrab.grab().
    Warm-up for 2 seconds, then measure for 2 seconds.
    Returns: (number_of_frames, elapsed_time)
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow/ImageGrab not available.")

    # Warm-up: capture frames for 2 seconds
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        _ = ImageGrab.grab()

    # Measurement: capture frames for 2 seconds
    start = time.time()
    count = 0
    while time.time() - start < 2.0:
        _ = ImageGrab.grab()
        count += 1
    elapsed = time.time() - start
    return (count, elapsed)


def capture_mss():
    """
    Capture using mss.
    Warm-up for 2 seconds, then measure for 2 seconds.
    Returns: (number_of_frames, elapsed_time)
    """
    if not MSS_AVAILABLE:
        raise RuntimeError("mss not available.")

    sct = mss()
    # Warm-up: capture frames for 2 seconds
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        _ = sct.grab(sct.monitors[0])

    # Measurement: capture frames for 2 seconds
    start = time.time()
    count = 0
    while time.time() - start < 2.0:
        _ = sct.grab(sct.monitors[0])
        count += 1
    elapsed = time.time() - start
    sct.close()
    return (count, elapsed)


def capture_pyscreenshot():
    """
    Capture using pyscreenshot.
    Warm-up for 2 seconds, then measure for 2 seconds.
    Returns: (number_of_frames, elapsed_time)
    """
    if not PYSCREENSHOT_AVAILABLE:
        raise RuntimeError("pyscreenshot not available.")

    # Warm-up: capture frames for 2 seconds
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        _ = ImageGrab2.grab()

    # Measurement: capture frames for 2 seconds
    start = time.time()
    count = 0
    while time.time() - start < 2.0:
        _ = ImageGrab2.grab()
        count += 1
    elapsed = time.time() - start
    return (count, elapsed)


def capture_pyqt5():
    """
    Capture using PyQt5's primary screen grab.
    Warm-up for 2 seconds, then measure for 2 seconds.
    Returns: (number_of_frames, elapsed_time)
    """
    if not PYQT5_AVAILABLE:
        raise RuntimeError("PyQt5 not available.")

    # If there's already an instance, reuse it. Otherwise, create one.
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    screen = QGuiApplication.primaryScreen()
    rect = screen.availableGeometry()

    # Warm-up: capture frames for 2 seconds
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        _ = screen.grabWindow(0, rect.x(), rect.y(), rect.width(), rect.height())

    # Measurement: capture frames for 2 seconds
    start = time.time()
    count = 0
    while time.time() - start < 2.0:
        _ = screen.grabWindow(0, rect.x(), rect.y(), rect.width(), rect.height())
        count += 1
    elapsed = time.time() - start

    return (count, elapsed)


###############################################################################
# Benchmarking Helper
###############################################################################


def run_benchmark(name, func):
    """
    Runs a single benchmark, measuring capture time and resource usage.
    Each capture function is assumed to return (frame_count, elapsed_time).
    """
    monitor = ResourceUsageMonitor(interval=0.1)
    monitor.start()
    try:
        result = func()
    except Exception as e:
        monitor.stop()
        print(f"Error in {name} capture: {e}")
        return

    monitor.stop()
    avg_cpu, avg_mem, avg_gpu = monitor.get_averages()
    frame_count, elapsed = result
    print(f"[{name}] {frame_count} frames captured")
    print(f"    Elapsed: {elapsed:.3f}s, {frame_count / elapsed:.2f} FPS")
    print(f"    Avg CPU (%): {avg_cpu:.2f}")
    print(f"    Avg Memory (MB): {avg_mem:.2f}")
    if PYNVML_AVAILABLE:
        print(f"    Avg GPU Usage (%): {avg_gpu:.2f}")
    print("")
    return result, avg_cpu, avg_mem, avg_gpu


def plot_results(results):
    # This function can be implemented to plot the benchmark results.
    # For now, it is a placeholder.
    print("Plotting is not yet implemented.")


def main():
    benchmarks = [
        ("OWA (runnable)", capture_owa_runnable),
        ("OWA (listener)", capture_owa_listener),
        ("Pillow", capture_pillow),
        ("mss", capture_mss),
        ("pyscreenshot", capture_pyscreenshot),
        ("PyQt5", capture_pyqt5),
    ]
    results = {}

    for name, func in benchmarks:
        print(f"==== Benchmarking {name} ====")
        result = run_benchmark(name, func)
        results[name] = result

    plot_results(results)


if __name__ == "__main__":
    main()
