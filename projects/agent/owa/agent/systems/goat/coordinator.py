import queue
import threading

from loguru import logger

from owa.agent.core import Clock, Event, Rate, get_default_clock
from owa.agent.core.perception import Perception, PerceptionQueue, PerceptionSpecDict
from owa.agent.core.pipe import Pipe
from owa.core import Runnable

from .processors import lazy_load_images, perception_to_conversation
from .trigger_speedhack import disable_speedhack, enable_speedhack
from .utils import EventProcessor


def decision_to_action(decision: str, *, now: int, event_processor: EventProcessor) -> list[Event]:
    events = event_processor.detokenize(decision)
    recon_now = event_processor.detokenize(event_processor._tokenize_timestamp(now))[0].timestamp
    for event in events:
        event.timestamp = (event.timestamp - recon_now) % event_processor.timestamp_range + now
    return events


class RealTimeAgentCoordinator(Runnable):
    def on_configure(
        self,
        perception_queue: PerceptionQueue,
        thought_queue: queue.Queue,
        action_queue: queue.Queue,
        decision_queue: queue.Queue,
        event_processor: EventProcessor,
        perception_spec_dict: PerceptionSpecDict,
        rate: float,
        clock: Clock,
        world_pause: bool = False,
    ):
        self._perception_queue = perception_queue
        self._thought_queue = thought_queue
        self._action_queue = action_queue
        self._decision_queue = decision_queue
        self._clock = clock
        self._rate = Rate(rate, clock=clock)
        self._perception_spec_dict = perception_spec_dict
        self._event_processor = event_processor
        self._world_pause = world_pause

    def loop(self, *, stop_event: threading.Event):
        # To prevent cold-start, we need to wait for the first perception
        self._clock.sleep(2 + self._perception_spec_dict.duration)

        perception_history = Perception()  # Stores the history of perceptions
        tick = 0
        while not stop_event.is_set():
            # 1. Perceive. From: PerceptionProvider
            current_perceptions = self._perception_queue.iter_queue()
            logger.trace(f"[PERCEIVE] Current perceptions: {current_perceptions!r}")

            now = self._clock.get_time_ns()
            perception_history, conversation = perception_to_conversation(
                perception_history,
                current_perceptions,
                now=now,
                is_training=False,
                spec=self._perception_spec_dict,
                event_processor=self._event_processor,
            )
            logger.trace(f"[PERCEIVE] Updated perception history: {perception_history!r}")
            logger.trace(f"now: {now}, tick: {tick}")
            logger.trace(f"[CONVERSATION] New conversation: {conversation!r}")

            # 2. Think. To: ModelWorker
            if conversation is not None:
                if self._world_pause:
                    self._clock.pause()
                    enable_speedhack()

                logger.debug("[THINK] Starting thought generation for conversation")
                pending_thought = (Pipe(conversation) | lazy_load_images).execute()
                logger.debug(f"[THINK] Enqueueing thought: {pending_thought!r}")
                # TODO: more safe, atomic way to handle queue
                try:
                    self._thought_queue.get_nowait()
                    logger.debug("[THINK] Thought queue is full, removing the oldest thought")
                except queue.Empty:
                    ...
                self._thought_queue.put_nowait(pending_thought)  # Enqueue the generated thought

            # 3. Act. From: ModelWorker, To: ActionExecutor
            try:
                if not self._world_pause:
                    decision = self._decision_queue.get_nowait()
                else:
                    decision = self._decision_queue.get()
                    self._clock.resume()
                    disable_speedhack()
                logger.info(f"[DECISION] Received decision: {decision!r}")
                actions = decision_to_action(decision, now=now, event_processor=self._event_processor)
                logger.debug(f"[ACTION] Parsed actions: {actions!r}")
                for action in actions:
                    logger.trace(f"[ACTION] Enqueueing action: {action!r}")
                    self._action_queue.put_nowait(action)  # Enqueue the action
            except queue.Empty:
                logger.debug("[DECISION] No decision available this tick")
                ...  # No decision available, continue

            # Wait until the next clock tick
            self._rate.sleep()
            tick += 1
