from owa import Runnable
from owa.registry import RUNNABLES


@RUNNABLES.register("example/runnable")
class ExampleRunnable(Runnable):
    """
    Runnable must implement the `loop` and `cleanup` methods.
    """

    def on_configure(self):
        """Optional method for configuration. This method is called when self.configure() is called."""

    def loop(self):
        """Main loop. This method must be interruptable by calling stop(), which sets the self._stop_event."""
        # Implement here!
        pass

    def cleanup(self):
        """Clean up resources. This method is called after loop() exits."""
        # Implement here!
        pass
