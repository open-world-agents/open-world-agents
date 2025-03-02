from owa import Listener
from owa.registry import LISTENERS


@LISTENERS.register("example/listener")
class ExampleListener(Listener):
    def on_configure(self):
        # Implement here!
        pass

    def loop(self, *, stop_event, callback):
        # Implement here!
        pass
