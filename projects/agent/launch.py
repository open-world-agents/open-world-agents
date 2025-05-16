import time
from contextlib import contextmanager
from queue import Queue

import typer
from loguru import logger

from owa.agent.core import get_default_clock
from owa.agent.core.perception import PerceptionQueue
from owa.agent.systems.example import ActionExecutor, ModelWorker, PerceptionProvider, RealTimeAgentCoordinator

# TODO: init from yaml, with configurable provider/coordinator/worker/action_executor/...


@contextmanager
def setup_resources():
    perception_queue = PerceptionQueue()
    thought_queue = Queue()
    decision_queue = Queue()
    action_queue = Queue()
    clock = get_default_clock()

    perception_provider = PerceptionProvider().configure(perception_queue=perception_queue, clock=clock)
    agent_coordinator = RealTimeAgentCoordinator().configure(
        perception_queue=perception_queue,
        thought_queue=thought_queue,
        decision_queue=decision_queue,
        action_queue=action_queue,
        rate=1.0,
    )
    model_worker = ModelWorker().configure(
        thought_queue=thought_queue, decision_queue=decision_queue, clock=clock, model_id="test_model"
    )
    action_executor = ActionExecutor().configure(action_queue=action_queue, clock=clock)

    resources = [
        (perception_provider, "perception_provider"),
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


def main():
    with setup_resources() as resources:  # noqa: F841
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Operation stopped by user.")


if __name__ == "__main__":
    typer.run(main)
