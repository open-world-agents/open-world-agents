import queue
import threading
import time

from loguru import logger

from owa.agent.core import Clock, Event
from owa.core import Runnable
from owa.core.registry import CALLABLES, activate_module
from owa.core.time import TimeUnits
from owa.env.desktop.msg import KeyboardEvent, MouseEvent


class ActionExecutor(Runnable):
    def on_configure(self, action_queue: queue.Queue[Event], clock: Clock):
        self._action_queue = action_queue
        self._clock = clock

        activate_module("owa.env.desktop")

    def loop(self, *, stop_event):
        while not stop_event.is_set():
            try:
                # NOTE: this sleep at least 1 seconds to avoid busy waiting
                event = self._action_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Schedule the action to execute at the specified timestamp
            action_msg = event.msg
            current_time = self._clock.get_time_ns()

            if event.timestamp > current_time:
                # Calculate delay needed
                delay = (event.timestamp - current_time) / TimeUnits.SECOND  # Convert to seconds
                if delay > 0:
                    threading.Timer(delay, self._execute_action, args=(action_msg,)).start()
                else:
                    # Execute immediately if the delay is non-positive
                    self.fallback(action_msg)
            else:
                # Execute immediately if the timestamp is in the past
                self.fallback(action_msg)

            # Mark the item as done in the queue
            self._action_queue.task_done()

    def fallback(self, action):
        """Fallback method to execute the action immediately."""
        logger.warning(f"Executing past action: {action}")
        self._execute_action(action)

    def _execute_action(self, action):
        try:
            if isinstance(action, KeyboardEvent):
                self._execute_keyboard_event(action)
            elif isinstance(action, MouseEvent):
                self._execute_mouse_event(action)
            else:
                logger.warning(f"Unknown action type: {type(action)}. Action: {action}")
        except Exception as e:
            logger.error(f"Error executing action: {e}")

    def _execute_keyboard_event(self, event: KeyboardEvent):
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
        try:
            if event.event_type == "move":
                logger.info(f"Executing mouse move to ({event.x}, {event.y})")
                current_x, current_y = CALLABLES["mouse.position"]()
                CALLABLES["mouse.move"](event.x - current_x, event.y - current_y)

            elif event.event_type == "click":
                if event.button and event.pressed is not None:
                    if event.pressed:
                        logger.info(f"Executing mouse press: {event.button} at ({event.x}, {event.y})")
                        # First move to the position
                        current_x, current_y = CALLABLES["mouse.position"]()
                        CALLABLES["mouse.move"](event.x - current_x, event.y - current_y)
                        # Then press
                        CALLABLES["mouse.press"](event.button)
                    else:
                        logger.info(f"Executing mouse release: {event.button} at ({event.x}, {event.y})")
                        # First move to the position
                        current_x, current_y = CALLABLES["mouse.position"]()
                        CALLABLES["mouse.move"](event.x - current_x, event.y - current_y)
                        # Then release
                        CALLABLES["mouse.release"](event.button)
                else:
                    logger.info(f"Executing mouse click at ({event.x}, {event.y})")
                    # Move to position then click
                    current_x, current_y = CALLABLES["mouse.position"]()
                    CALLABLES["mouse.move"](event.x - current_x, event.y - current_y)
                    CALLABLES["mouse.click"](event.button or "left", 1)

            elif event.event_type == "scroll":
                if event.dx is not None or event.dy is not None:
                    logger.info(f"Executing mouse scroll: dx={event.dx}, dy={event.dy}")
                    CALLABLES["mouse.scroll"](event.dx or 0, event.dy or 0)
                else:
                    logger.warning("Scroll event missing dx or dy values")

            else:
                logger.warning(f"Unknown mouse event type: {event.event_type}")
        except Exception as e:
            logger.error(f"Error executing mouse event: {e}")
