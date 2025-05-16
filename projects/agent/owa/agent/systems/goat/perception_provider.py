from collections import defaultdict
from contextlib import contextmanager

from loguru import logger

from owa.agent.core import Clock, Event
from owa.agent.core.perception import PerceptionQueue, PerceptionSpec, PerceptionSpecDict
from owa.core import Runnable
from owa.core.registry import LISTENERS, activate_module

from .perception_spec import PERCEPTION_SPEC_DICT

WINDOW_NAME = None
# TODO: adjust FPS per PerceptionSpecDict for more efficient processing
# Note that FPS=60 works well without problem since it certainly ensures "upper bound" of the perceived info.
FPS = 60


def configure():
    activate_module("owa.env.desktop")
    activate_module("owa.env.gst")


class PerceptionProvider(Runnable):
    def on_configure(
        self, perception_queue: PerceptionQueue, clock: Clock, spec: PerceptionSpecDict = PERCEPTION_SPEC_DICT
    ):
        self._perception_queue = perception_queue
        self._clock = clock
        self._callbacks = defaultdict(list)

        for channel, spec in spec.items():
            self.setup_spec(channel, spec)
        self.setup_listeners()

    def loop(self, *, stop_event):
        with self.setup_listeners() as resources:
            while not stop_event.is_set():
                self._clock.sleep(1)

    def setup_spec(self, channel: str, spec: PerceptionSpec):
        for topic in spec.topics:

            def callback(event):
                return self._perception_queue[topic].put_nowait(
                    Event(timestamp=self._clock.get_time_ns(), topic=topic, msg=event)
                )

            self._callbacks[channel].append(callback)

        # TODO?: more efficient processing utilizing spec.sample_configs

    def _handle_callbacks(self, channel: str, x):
        """Invoke all callbacks for a given channel with argument x."""
        for func in self._callbacks[channel]:
            func(x)

    @contextmanager
    def setup_listeners(self):
        configure()
        # Register listeners
        screen = LISTENERS["screen"]().configure(
            window_name=WINDOW_NAME, fps=FPS, callback=lambda x: self._handle_callbacks("screen", x)
        )
        keyboard = LISTENERS["keyboard"]().configure(callback=lambda x: self._handle_callbacks("keyboard", x))
        mouse = LISTENERS["mouse"]().configure(callback=lambda x: self._handle_callbacks("mouse", x))
        window = LISTENERS["window"]().configure(callback=lambda x: self._handle_callbacks("window", x))
        keyboard_state = LISTENERS["keyboard/state"]().configure(
            callback=lambda x: self._handle_callbacks("keyboard/state", x)
        )
        mouse_state = LISTENERS["mouse/state"]().configure(callback=lambda x: self._handle_callbacks("mouse/state", x))

        resources = [
            (screen, "screen"),
            (keyboard, "keyboard"),
            (mouse, "mouse"),
            (window, "window"),
            (keyboard_state, "keyboard/state"),
            (mouse_state, "mouse/state"),
        ]

        for resource, name in resources:
            resource.start()
            logger.debug(f"Started {name}")
        try:
            yield resources
        finally:
            for resource, name in reversed(resources):
                try:
                    resource.stop()
                    resource.join(timeout=5)
                    logger.debug(f"Stopped {name}")
                except Exception as e:
                    logger.error(f"Error stopping {name}: {e}")
