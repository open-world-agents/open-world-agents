import inspect
import multiprocessing as mp
import threading
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event as mpEvent
from typing import Self


class RunnableSessionContextManager:
    """
    Context manager for RunnableMixin objects that automatically handles the start/stop lifecycle.

    Provides a convenient way to use runnable objects with Python's 'with' statement,
    ensuring that resources are properly managed.
    """

    def __init__(self, runnable: "RunnableMixin"):
        self.runnable = runnable

    def __enter__(self):
        self.runnable.start()
        return self.runnable

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop and join the runnable object when exiting the context."""
        self.runnable.stop()
        self.runnable.join()

    def is_alive(self):
        return self.runnable.is_alive()


class RunnableMixin(ABC):
    """
    Interface class for Runnable objects, which supports start/stop/join operations.

    This class provides a common interface similar to Python's threading.Thread, but with
    additional functionality like the stop method. It's designed to be used as a mixin
    with actual implementation classes.

    Example:
        ```python
        class MyRunnable(Runnable):
            def loop(self, stop_event):
                with open("test.txt", "w") as file:
                    while not stop_event.is_set():
                        file.write("Hello, world! ")
                        stop_event.wait(1)

        runnable = MyRunnable().configure()
        with runnable.session as session:
            time.sleep(5)
        ```
    """

    # Interface methods that implementations must provide
    @abstractmethod
    def start(self): ...

    @abstractmethod
    def stop(self):
        """Signal the runnable object to stop."""

    @abstractmethod
    def join(self):
        """Wait for the runnable object to complete."""

    @abstractmethod
    def is_alive(self): ...

    # Common functionality
    _configured = False

    @property
    def session(self):
        """
        Get a context manager for this runnable.

        Returns:
            RunnableSessionContextManager: A context manager that handles the lifecycle.
        """
        return RunnableSessionContextManager(self)

    def configure(self, *args, **kwargs) -> Self:
        """
        Configure the runnable before starting.

        This method must be called before start() to prepare the runnable.

        Args:
            *args: Positional arguments to pass to on_configure.
            **kwargs: Keyword arguments to pass to on_configure.

        Returns:
            Self: The configured runnable instance.
        """
        self.on_configure(*args, **kwargs)
        self._configured = True
        return self

    # Methods for subclasses to implement
    def on_configure(self, *args, **kwargs):
        """
        Optional method for configuration.

        Override this method to perform initialization before the runnable starts.
        This method is called when self.configure() is called.

        Args:
            *args: Positional arguments passed from configure.
            **kwargs: Keyword arguments passed from configure.
        """
        pass

    @abstractmethod
    def loop(self):
        """
        Main execution loop. Must be implemented by subclasses.

        This method contains the main logic that runs while the runnable is active.

        Requirements:
            - It MUST respect the self.stop to allow clean termination.
        """


class RunnableThread(threading.Thread, RunnableMixin):
    """
    A Thread implementation of the RunnableMixin interface.

    This class provides a thread-based implementation of a runnable object,
    suitable for I/O-bound tasks or operations that should run in a separate thread.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a new RunnableThread. Whole arguments are passed to threading.Thread.

        To configure the runnable, write your own on_configure method instead.
        """
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def run(self):
        """
        Thread execution method. Do not call this directly; use start() instead.

        This method ensures the runnable is configured before executing the loop.
        """
        if not getattr(self, "_configured", False):
            raise RuntimeError(
                "RunnableThread is not configured. Call configure() before start(). Or you may have overriden the configure method, not on_configure."
            )

        kwargs = {}
        if "stop_event" in inspect.signature(self.loop).parameters:
            kwargs["stop_event"] = self._stop_event
        self.loop(**kwargs)

    def stop(self):
        """Signal the thread to stop by setting the stop event."""
        self._stop_event.set()

    @abstractmethod
    def loop(self, stop_event: threading.Event):
        """
        Main thread execution loop. Must be implemented by subclasses.

        Args:
            stop_event (threading.Event): An event that will be set when the thread should stop.
                                        Check this event regularly and exit when it's set.
                                        If this argument is not present, the loop will be called without it.
        """
        pass


class RunnableProcess(mp.Process, RunnableMixin):
    """
    A Process implementation of the RunnableMixin interface.

    This class provides a process-based implementation of a runnable object,
    suitable for CPU-bound tasks or operations that benefit from separate process isolation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a new RunnableProcess. Whole arguments are passed to threading.Thread.

        To configure the runnable, write your own on_configure method instead.
        """
        super().__init__(*args, **kwargs)
        self._stop_event = mp.Event()

    def run(self):
        """
        Process execution method. Do not call this directly; use start() instead.

        This method ensures the runnable is configured before executing the loop.
        """
        if not getattr(self, "_configured", False):
            raise RuntimeError("RunnableProcess is not configured. Call configure() before start().")

        kwargs = {}
        if "stop_event" in inspect.signature(self.loop).parameters:
            kwargs["stop_event"] = self._stop_event
        self.loop(**kwargs)

    def stop(self):
        """Signal the process to stop by setting the stop event."""
        self._stop_event.set()

    @abstractmethod
    def loop(self, stop_event: mpEvent):
        """
        Main process execution loop. Must be implemented by subclasses.

        Args:
            stop_event (multiprocessing.Event): An event that will be set when the process should stop.
                                                Check this event regularly and exit when it's set.
                                                If this argument is not present, the loop will be called without it.
        """
        pass


# Default implementation is thread-based for better compatibility and easier use
Runnable = RunnableThread
