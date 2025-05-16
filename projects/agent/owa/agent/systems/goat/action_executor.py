import queue

from owa.core import Runnable
from owa.core.registry import CALLABLES, activate_module


class ActionExecutor(Runnable):
    def on_configure(self, action_queue, clock):
        self._action_queue = action_queue
        self._clock = clock

        activate_module("owa.env.desktop")

    def loop(self, *, stop_event):
        while not stop_event.is_set():
            try:
                # NOTE: this sleep at least 1 seconds to avoid busy waiting
                action: str = self._action_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Execute the action
            if action.isdigit():
                CALLABLES["keyboard.type"](action)
            else:
                print(f"Unknown action: {action}")
