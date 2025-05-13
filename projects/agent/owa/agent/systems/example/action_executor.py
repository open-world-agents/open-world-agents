from owa.core import Runnable


class ActionExecutor(Runnable):
    def on_configure(self, action_queue, clock):
        self._action_queue = action_queue
        self._clock = clock

    def loop(self, *, stop_event):
        # Execute actions logic
        pass
