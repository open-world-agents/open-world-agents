import threading
import time
from contextlib import contextmanager
from queue import Queue

from loguru import logger

from mcap_owa.highlevel import OWAMcapReader
from owa.core import Runnable


class Event: ...


class Clock: ...


# Shared class across training/inference time
def perception_to_conversation(perception_history, thought_history, current_perception, now):
    """**Note**: For events later than now, it's considered as future events("label")."""


def lazy_load_images(inputs): ...


def apply_processor(processor, inputs): ...


from typing import Callable


class Pipe:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._pipe = []

    def __or__(self, other: "Callable | Pipe") -> "Pipe":
        if isinstance(other, Pipe):
            self._pipe.extend(other._pipe)
        elif callable(other):
            self._pipe.append(other)
        else:
            raise TypeError(f"Unsupported type for pipe: {type(other)}")
        return self

    def execute(self):
        if not self._pipe:
            raise ValueError("No functions in the pipe to execute.")
        result = self._pipe[0](*self._args, **self._kwargs)
        for func in self._pipe[1:]:
            result = func(result)
        return result


def decision_to_action(decision): ...


enqueue_thought = lambda x: None  # Placeholder for the actual implementation
enqueue_action = lambda: None  # Placeholder for the actual implementation


def dequeue_perceptions(perception_queue):
    """Get all available perceptions from the queue without blocking."""
    perceptions = []
    while not perception_queue.empty():
        try:
            perception = perception_queue.get_nowait()
            perceptions.append(perception)
        except Queue.Empty:
            break
    return perceptions


def dequeue_decision(decision_queue):
    """Get the latest decision from the queue, or None if no decision is available."""
    try:
        return decision_queue.get_nowait()
    except Queue.Empty:
        return None


class RealTimeAgentCoordinator(Runnable):
    def on_configure(self, perception_queue: Queue, decision_queue: Queue, clock: Clock):
        self._perception_queue = perception_queue  # from AgentExecutor to Agent
        self._decision_queue = decision_queue  # from ModelWorker to Agent
        self._clock = clock

    def loop(self, *, stop_event: threading.Event):
        perception_history = []
        thought_history = []
        while not stop_event.is_set():
            # On each tick, the agent will do the following:
            # 1. Perceive: Get the latest events from the perception queue
            # 2. Think: Process the events and prepare the next action
            # 3. Act: Execute the action and enqueue the result to the action queue
            # 4. Sleep until the next tick
            # **Note**: All of these steps must be done within a single tick.
            # Some common patterns are:
            # - Single-thread think: Heavy ML model inference is done in a separate thread.
            #   In this case, block new think enqueue until the inference is done.

            # 1. Perceive
            current_perceptions = dequeue_perceptions(self._perception_queue)

            # 2. Think
            pending_thought = (
                Pipe(perception_history, thought_history, current_perceptions, now=self._clock.get_time())
                | perception_to_conversation
                | lazy_load_images
                | apply_processor
            )
            enqueue_thought(pending_thought)

            # 3. Act
            decision = dequeue_decision(self._decision_queue)
            action = decision_to_action(decision)
            enqueue_action(action)

            self._clock.sleep_until_next_tick()


# -----------------------------------------------------------------------------
# AgentCoordinator (main orchestrator)
#      |
#      |-- PerceptionProvider  # Handles perception acquisition, fills perception_queue
#      |
#      |-- ModelWorker         # Processes perceptions/history, produces decisions (ML)
#      |
#      |-- ActionExecutor      # Executes actions based on decisions
# -----------------------------------------------------------------------------

# Data/control flow:
#
#   +---------------------+
#   | PerceptionProvider  |
#   +---------------------+
#             |
#      fills perception_queue
#             |
#   +---------------------+      +---------------------+      +------------------+
#   |  AgentCoordinator   | ---> |    ModelWorker      | ---> |  ActionExecutor  |
#   +---------------------+      +---------------------+      +------------------+
#          |  reads perceptions      | process           | outputs action
#          |  from queue            | to decision queue | to actuator/system
#          v
#        orchestrates the loop:
#           - drains perceptions
#           - maintains histories
#           - submits thought requests
#           - collects decisions
#           - enqueues actions


@contextmanager
def setup_resources(model_id: str):
    perception_queue = Queue()
    thought_queue = Queue()
    decision_queue = Queue()
    action_queue = Queue()
    clock = Clock()

    # Overall process: perception -> thought -> decision -> action
    # PerceptionProvider -- (perception) --> RealTimeAgentCoordinator -- (thought)
    # --> ModelWorker -- (decision) --> ActionExecutor
    perception_provider = PerceptionProvider().configure(perception_queue=perception_queue, clock=clock)
    agent_coordinator = RealTimeAgentCoordinator().configure(
        perception_queue=perception_queue, decision_queue=decision_queue, clock=clock
    )
    model_worker = ModelWorker().configure(
        thought_queue=thought_queue, decision_queue=decision_queue, clock=clock, model_id=model_id
    )
    action_executor = ActionExecutor().configure(action_queue=action_queue, clock=clock)

    resources = [
        [perception_provider, "perception_provider"],
        (agent_coordinator, "agent_coordinator"),
        (model_worker, "model_worker"),
        (action_executor, "action_executor"),
    ]
    for resource, name in resources:
        resource.start()
    try:
        yield resources
    finally:
        for resource, name in reversed(resources):
            try:
                resource.stop()
                resource.join(timeout=5)
                logger.info(f"Stopped {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")


def main(*args):
    """Run the game agent with the specified model."""
    with setup_resources(*args) as resources:
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Operation stopped by user.")


class OWAMcapPerceptionReader:
    def __init__(self, file_path): ...
    def sample(self, now): ...


def create_dataset():
    for file_path in ["file1.mcap", "file2.mcap"]:
        with OWAMcapReader(file_path) as reader:  # noqa: F841
            valid_intervals = [(0, 1000), (2000, 3000)]  # Example intervals
            # In real implementation, these intervals are derived by various logics.
            # e.g. representing valid interval with special key, ...
        for now in iter_timestamps(valid_intervals):
            try:
                current_perception = OWAMcapPerceptionReader(file_path).sample(now)
                pending_thought = Pipe([], [], current_perception, now=now) | perception_to_conversation
                yield file_path, now, pending_thought
            except Exception as e:  # noqa: F841
                ...


def iter_timestamps(valid_intervals):
    for start, end in valid_intervals:
        for timestamp in range(start, end, 100):  # Example step
            yield timestamp


class MyDataset:
    def __init__(self, data):
        """Initialize with a list of file paths and timestamps."""
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        # 1. Get input from the queue
        file_path, timestamp = self._data[idx]
        current_perception = OWAMcapPerceptionReader(file_path).sample(now=timestamp)

        pending_thought = (
            Pipe([], [], current_perception, now=timestamp) | perception_to_conversation | lazy_load_images
        )
        # mllm_preprocess is done at data collator in batch.

        return pending_thought.execute()


def collate_fn(batch, processor=None):
    """Process a batch of data samples."""
    if processor is None:
        return batch

    # Apply any batch-level processor operations
    processed_batch = []
    for item in batch:
        processed_item = apply_processor(processor, item)
        processed_batch.append(processed_item)

    return processed_batch
