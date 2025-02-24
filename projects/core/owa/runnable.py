import multiprocessing
import threading
from abc import ABC, abstractmethod
from typing import Self


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

    # What is implemented
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.join()

    def configure(self, *args, **kwargs) -> Self:
        self.on_configure(*args, **kwargs)
        return self

    # What user implements
    def on_configure(self):
        """Optional method for configuration. This method is called when self.configure() is called."""

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
    def on_configure(self):
        """Optional method for configuration. This method is called when self.configure() is called."""

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
    def on_configure(self):
        """Optional method for configuration. This method is called when self.configure() is called."""

    @abstractmethod
    def loop(self):
        """Main loop. This method must be interruptable by calling stop(), which sets the self._stop_event."""

    @abstractmethod
    def cleanup(self):
        """Clean up resources. This method is called after loop() exits."""


Runnable = RunnableThread  # Default to RunnableThread
