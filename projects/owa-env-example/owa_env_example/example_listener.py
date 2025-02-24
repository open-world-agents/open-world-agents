from owa import Listener
from owa.registry import LISTENERS


@LISTENERS.register("example/listener")
class ExampleListener(Listener):
    """
    This listener must implement the `loop` and `cleanup` methods.

    Within the listener, call `self.callback` to notify the environment of an event.
    The `callback` function is provided as an argument to `__init__` and stored as `self.callback`.
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
