import threading
from queue import Queue

from agent_system.core.pipe import Pipe
from agent_system.core.utils import (
    decision_to_action,
    dequeue_decision,
    dequeue_perceptions,
    enqueue_action,
    enqueue_thought,
)
from agent_system.pipeline.processors import apply_processor, lazy_load_images, perception_to_conversation

from owa.core import Runnable


class RealTimeAgentCoordinator(Runnable):
    def on_configure(self, perception_queue: Queue, decision_queue: Queue, clock):
        self._perception_queue = perception_queue
        self._decision_queue = decision_queue
        self._clock = clock

    def loop(self, *, stop_event: threading.Event):
        perception_history = []  # Stores the history of perceptions
        while not stop_event.is_set():
            # 1. Perceive. From: PerceptionProvider
            current_perceptions = dequeue_perceptions(self._perception_queue)
            perception_history, conversation = perception_to_conversation(
                perception_history, current_perceptions, now=self._clock.get_time()
            )

            # 2. Think. To: ModelWorker
            pending_thought = (Pipe(conversation) | lazy_load_images | apply_processor).execute()
            enqueue_thought(pending_thought)  # Enqueue the generated thought

            # 3. Act. From: ModelWorker, To: ActionExecutor
            decision = dequeue_decision(self._decision_queue)
            action = decision_to_action(decision)
            enqueue_action(action)  # Enqueue the action

            # Wait until the next clock tick
            self._clock.sleep_until_next_tick()
