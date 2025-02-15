import multiprocessing
import threading
from abc import ABC, abstractmethod


class RunnableMixin(ABC):
    """
    Interface class for Runnable objects, which supports start/stop/join/cleanup operations.

    Example:
    ```python
    class MyRunnable(Runnable):
        def loop(self):
            file = open("test.txt", "w")
            while not self._stop_event.is_set():
                file.write("Hello, world!\n")
                self._stop_event.wait(1)
        def cleanup(self):
            file.close()
    """

    # What user calls
    @abstractmethod
    def start(self): ...
    @abstractmethod
    def stop(self): ...
    @abstractmethod
    def join(self): ...
    @abstractmethod
    def is_alive(self): ...

    # What user implements
    def configure(self):
        """Optional method for configuration."""

    @abstractmethod
    def loop(self):
        """Main loop. This method must be interruptable by calling stop()."""

    @abstractmethod
    def cleanup(self):
        """Clean up resources. This method is called after loop() exits."""


class RunnableThread(threading.Thread, RunnableMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def run(self):
        try:
            self.loop()
        finally:
            self.cleanup()

    def stop(self):
        self._stop_event.set()

    # What user implements
    def configure(self):
        """Optional method for configuration."""

    @abstractmethod
    def loop(self):
        """Main loop. This method must be interruptable by calling stop(), which sets the self._stop_event."""

    @abstractmethod
    def cleanup(self):
        """Clean up resources. This method is called after loop() exits."""


class RunnableProcess(multiprocessing.Process, RunnableMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = multiprocessing.Event()

    def run(self):
        try:
            self.loop()
        finally:
            self.cleanup()

    def stop(self):
        self._stop_event.set()

    # What user implements
    def configure(self):
        """Optional method for configuration."""

    @abstractmethod
    def loop(self):
        """Main loop. This method must be interruptable by calling stop(), which sets the self._stop_event."""

    @abstractmethod
    def cleanup(self):
        """Clean up resources. This method is called after loop() exits."""


Runnable = RunnableThread  # Default to RunnableThread
