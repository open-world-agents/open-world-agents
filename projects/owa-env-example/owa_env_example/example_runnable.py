from owa import Runnable
from owa.registry import RUNNABLES


@RUNNABLES.register("example/runnable")
class ExampleRunnable(Runnable):
    def on_configure(self):
        # Implement here!
        pass

    def loop(self, *, stop_event):
        # Implement here!
        pass
