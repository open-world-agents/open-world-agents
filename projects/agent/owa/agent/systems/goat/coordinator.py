import queue
import threading

from owa.agent.core import Clock, Rate, get_default_clock
from owa.agent.core.perception import Perception, PerceptionQueue, PerceptionSpecDict
from owa.agent.core.pipe import Pipe
from owa.agent.core.utils import iter_queue
from owa.core import Runnable

from .perception_spec import PERCEPTION_SPEC_DICT
from .processors import lazy_load_images, perception_to_conversation


def decision_to_action(decision):
    # Placeholder function to convert decision to action
    # Implement your logic here
    return None


class RealTimeAgentCoordinator(Runnable):
    def on_configure(
        self,
        perception_queue: PerceptionQueue,
        thought_queue: queue.Queue,
        action_queue: queue.Queue,
        decision_queue: queue.Queue,
        rate: float,
        clock: Clock | None = None,
        perception_spec_dict: PerceptionSpecDict = PERCEPTION_SPEC_DICT,
    ):
        self._perception_queue = perception_queue
        self._thought_queue = thought_queue
        self._action_queue = action_queue
        self._decision_queue = decision_queue
        self._clock = clock or get_default_clock()
        self._rate = Rate(rate, clock=clock)
        self._perception_spec_dict = perception_spec_dict

    def loop(self, *, stop_event: threading.Event):
        perception_history = Perception()  # Stores the history of perceptions
        while not stop_event.is_set():
            # 1. Perceive. From: PerceptionProvider
            current_perceptions = self._perception_queue.iter_queue()
            perception_history, conversation = perception_to_conversation(
                perception_history,
                current_perceptions,
                now=self._clock.get_time_ns(),
                spec=self._perception_spec_dict,
            )

            # 2. Think. To: ModelWorker
            if conversation is not None:
                pending_thought = (Pipe(conversation) | lazy_load_images).execute()
                self._thought_queue.put_nowait(pending_thought)  # Enqueue the generated thought

            # 3. Act. From: ModelWorker, To: ActionExecutor
            try:
                decision = self._decision_queue.get_nowait()
                action = decision_to_action(decision)
                self._action_queue.put_nowait(action)  # Enqueue the action
            except queue.Empty:
                ...  # No decision available, continue

            # Wait until the next clock tick
            print(f"Perception history: {str(perception_history)[:30]}...")
            print(f"Conversation: {conversation}")
            print(f"Pending thought: {pending_thought if 'pending_thought' in locals() else None}")
            print(f"Decision: {decision if 'decision' in locals() else None}")
            self._rate.sleep()
