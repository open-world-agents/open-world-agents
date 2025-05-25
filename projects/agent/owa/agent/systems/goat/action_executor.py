import queue
import threading
from typing import Union

from loguru import logger

from owa.agent.core import Clock, Event
from owa.core import Runnable
from owa.core.registry import CALLABLES, activate_module
from owa.core.time import TimeUnits
from owa.env.desktop.msg import KeyboardEvent, MouseEvent


class ActionExecutor(Runnable):
    QUEUE_TIMEOUT = 1  # seconds

    def on_configure(self, action_queue: queue.Queue[list[Event]], clock: Clock, preempt: bool = False):
        self._action_queue = action_queue
        self._clock = clock
        self._preempt = preempt
        self._dequeued_task_count = 0

        activate_module("owa.env.desktop")

    def loop(self, *, stop_event):
        while not stop_event.is_set():
            events = self._get_events_from_queue()
            # do not skip for events=[], since no-op is also an action
            if events is None:
                continue

            self._dequeued_task_count += 1
            self._process_events(events)
            self._action_queue.task_done()

    def _get_events_from_queue(self) -> list[Event] | None:
        """Get events from queue with timeout handling."""
        try:
            return self._action_queue.get(timeout=self.QUEUE_TIMEOUT)
        except queue.Empty:
            return None

    def _process_events(self, events: list[Event]):
        """Process a list of events."""
        for event in events:
            action_msg = event.msg
            current_time = self._clock.get_time_ns()

            if event.timestamp <= current_time:
                self.fallback(action_msg)
            else:
                delay = (event.timestamp - current_time) / TimeUnits.SECOND
                if delay > 0:
                    self._schedule_action(action_msg, delay)
                else:
                    self.fallback(action_msg)

    def _schedule_action(self, action_msg, delay: float):
        """Schedule an action to execute after a delay."""
        threading.Thread(
            target=self._scheduled_execute_action,
            args=(action_msg, delay, self._dequeued_task_count),
            daemon=True,
        ).start()

    def _scheduled_execute_action(self, action, delay: float, task_count: int):
        """Sleep using clock then execute the action."""
        self._clock.sleep(delay)
        if self._preempt and task_count < self._dequeued_task_count:
            return
        self._execute_action(action)

    def fallback(self, action):
        """Fallback method to execute the action immediately."""
        logger.warning(f"Executing past action: {action}")
        self._execute_action(action)

    def _execute_action(self, action: Union[KeyboardEvent, MouseEvent]):
        """Execute an action based on its type."""
        try:
            # TODO: move Keyboard/Mouse execution logic to `owa.env.desktop`'s Callable
            if isinstance(action, KeyboardEvent):
                self._execute_keyboard_event(action)
            elif isinstance(action, MouseEvent):
                self._execute_mouse_event(action)
            else:
                logger.warning(f"Unknown action type: {type(action)}. Action: {action}")
        except Exception as e:
            logger.error(f"Error executing action: {e}")

    def _execute_keyboard_event(self, event: KeyboardEvent):
        """Execute a keyboard event."""
        try:
            if event.event_type == "press":
                logger.info(f"Executing keyboard press: {event.vk}")
                CALLABLES["keyboard.press"](event.vk)
            elif event.event_type == "release":
                logger.info(f"Executing keyboard release: {event.vk}")
                CALLABLES["keyboard.release"](event.vk)
            else:
                logger.warning(f"Unknown keyboard event type: {event.event_type}")
        except Exception as e:
            logger.error(f"Error executing keyboard event: {e}")

    def _execute_mouse_event(self, event: MouseEvent):
        """Execute a mouse event."""
        try:
            if event.event_type == "move":
                logger.info(f"Executing mouse move to ({event.x}, {event.y})")
                self._move_mouse_to_position(event.x, event.y)

            elif event.event_type == "click":
                self._handle_mouse_click(event)

            elif event.event_type == "scroll":
                self._handle_mouse_scroll(event)

            else:
                logger.warning(f"Unknown mouse event type: {event.event_type}")
        except Exception as e:
            logger.error(f"Error executing mouse event: {e}")

    def _move_mouse_to_position(self, x: int, y: int):
        """Move mouse to the specified position."""
        current_x, current_y = CALLABLES["mouse.position"]()
        CALLABLES["mouse.move"](x - current_x, y - current_y)

    def _handle_mouse_click(self, event: MouseEvent):
        """Handle mouse click events."""
        if event.button and event.pressed is not None:
            action = "press" if event.pressed else "release"
            logger.info(f"Executing mouse {action}: {event.button} at ({event.x}, {event.y})")
            self._move_mouse_to_position(event.x, event.y)
            CALLABLES[f"mouse.{action}"](event.button)
        else:
            logger.info(f"Executing mouse click at ({event.x}, {event.y})")
            self._move_mouse_to_position(event.x, event.y)
            CALLABLES["mouse.click"](event.button or "left", 1)

    def _handle_mouse_scroll(self, event: MouseEvent):
        """Handle mouse scroll events."""
        if event.dx is not None or event.dy is not None:
            logger.info(f"Executing mouse scroll: dx={event.dx}, dy={event.dy}")
            CALLABLES["mouse.scroll"](event.dx or 0, event.dy or 0)
        else:
            logger.warning("Scroll event missing dx or dy values")
