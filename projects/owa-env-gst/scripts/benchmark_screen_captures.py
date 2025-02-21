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

# this import causes script not to exit
# try:
#     import pyscreenshot as ImageGrab2

#     PYSCREENSHOT_AVAILABLE = True
# except ImportError:
#     PYSCREENSHOT_AVAILABLE = False

try:
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QGuiApplication
    from PyQt5.QtWidgets import QApplication

    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

# Attempt OWA-based import
try:
    from owa.registry import RUNNABLES, activate_module

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
            except:
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
            mem_percent = None
            gpu_percent = None

            if self.process:
                cpu_percent = self.process.cpu_percent(interval=None)
                # The above call updates the internal stats; memory_info below is an instantaneous snapshot
                mem_info = self.process.memory_info()
                # Convert memory usage to MB
                mem_percent = mem_info.rss / (1024 * 1024)

            if self.gpu_handle:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_percent = util.gpu
                except:
                    pass

            # We actually sleep at the end for the correct interval
            # but we gather CPU usage "just in time"
            if cpu_percent is None:
                cpu_percent = 0.0
            if mem_percent is None:
                mem_percent = 0.0
            if gpu_percent is None:
                gpu_percent = 0.0

            self.samples.put((cpu_percent, mem_percent, gpu_percent))

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


###############################################################################
# Screen Capture Routines
###############################################################################


def capture_owa(frames=30):
    """
    Capture using the OWA "owa_env_gst" module for a certain number of frames.
    """
    if not OWA_AVAILABLE:
        raise RuntimeError("OWA environment not available.")

    activate_module("owa_env_gst")
    screen_capture = RUNNABLES["screen_capture"]()
    screen_capture.start()
    screen_capture.configure(fps=60)

    # Warm-up
    for _ in range(15):
        _ = screen_capture.grab()

    now = time.time()
    for _ in range(frames):
        frame_msg = screen_capture.grab()
    elapsed = time.time() - now

    screen_capture.stop()
    screen_capture.join()
    return elapsed


def capture_pillow(frames=30):
    """
    Capture using Pillow's ImageGrab.grab().
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow/ImageGrab not available.")
    # Warm-up
    for _ in range(5):
        _ = ImageGrab.grab()
    now = time.time()
    for _ in range(frames):
        _ = ImageGrab.grab()
    elapsed = time.time() - now
    return elapsed


def capture_mss(frames=30):
    """
    Capture using mss. On Windows or Linux, mss can capture the entire screen.
    """
    if not MSS_AVAILABLE:
        raise RuntimeError("mss not available.")
    sct = mss()
    # Warm-up
    for _ in range(5):
        _ = sct.grab(sct.monitors[0])
    now = time.time()
    for _ in range(frames):
        _ = sct.grab(sct.monitors[0])
    elapsed = time.time() - now
    sct.close()
    return elapsed


def capture_pyscreenshot(frames=30):
    """
    Capture using pyscreenshot.
    """
    if not PYSCREENSHOT_AVAILABLE:
        raise RuntimeError("pyscreenshot not available.")
    # Warm-up
    for _ in range(5):
        _ = ImageGrab2.grab()
    now = time.time()
    for _ in range(frames):
        _ = ImageGrab2.grab()
    elapsed = time.time() - now
    return elapsed


def capture_pyqt5(frames=30):
    """
    Capture using PyQt5's primary screen grab.
    We need a QApplication running to do this.
    """
    if not PYQT5_AVAILABLE:
        raise RuntimeError("PyQt5 not available.")
    # If there's already an instance, let's reuse it. Otherwise, create one.
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    screen = QGuiApplication.primaryScreen()
    rect = screen.availableGeometry()  # entire screen

    # Warm-up
    for _ in range(5):
        img = screen.grabWindow(0, rect.x(), rect.y(), rect.width(), rect.height())

    now = time.time()
    for _ in range(frames):
        img = screen.grabWindow(0, rect.x(), rect.y(), rect.width(), rect.height())  # noqa: F841
    elapsed = time.time() - now
    return elapsed


###############################################################################
# Benchmarking Helper
###############################################################################


def run_benchmark(name, func, frames=30):
    """
    Runs a single benchmark, measuring capture time and resource usage.
    """
    monitor = ResourceUsageMonitor(interval=0.1)
    monitor.start()
    try:
        elapsed = func(frames=frames)
    except Exception as e:
        monitor.stop()
        print(f"Error in {name} capture: {e}")
        return

    monitor.stop()
    avg_cpu, avg_mem, avg_gpu = monitor.get_averages()
    print(f"[{name}] {frames} frames captured")
    print(f"    Elapsed: {elapsed:.3f}s")
    print(f"    Avg CPU (%): {avg_cpu:.2f}")
    print(f"    Avg Memory (MB): {avg_mem:.2f}")
    if PYNVML_AVAILABLE:
        print(f"    Avg GPU Usage (%): {avg_gpu:.2f}")
    print("")
    return elapsed, avg_cpu, avg_mem, avg_gpu


def plot_results(results): ...


def main():
    # You can adjust the number of frames for your test:
    N_FRAMES = 60

    benchmarks = [
        ("OWA (owa_env_gst)", capture_owa),
        ("Pillow", capture_pillow),
        ("mss", capture_mss),
        # ("pyscreenshot", capture_pyscreenshot),
        ("PyQt5", capture_pyqt5),
    ]
    results = {}

    for name, func in benchmarks:
        print(f"==== Benchmarking {name} ====")
        result = run_benchmark(name, func, frames=N_FRAMES)
        results[name] = result

    plot_results(results)


if __name__ == "__main__":
    main()
