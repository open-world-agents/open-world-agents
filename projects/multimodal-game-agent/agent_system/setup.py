import time
from contextlib import contextmanager
from queue import Queue

from agent_system.agent.action_executor import ActionExecutor
from agent_system.agent.coordinator import RealTimeAgentCoordinator
from agent_system.agent.worker import ModelWorker
from agent_system.core.event import Clock
from agent_system.perception.provider import PerceptionProvider
from loguru import logger


@contextmanager
def setup_resources(model_id: str):
    perception_queue = Queue()
    thought_queue = Queue()
    decision_queue = Queue()
    action_queue = Queue()
    clock = Clock()

    perception_provider = PerceptionProvider().configure(perception_queue=perception_queue, clock=clock)
    agent_coordinator = RealTimeAgentCoordinator().configure(
        perception_queue=perception_queue, decision_queue=decision_queue, clock=clock
    )
    model_worker = ModelWorker().configure(
        thought_queue=thought_queue, decision_queue=decision_queue, clock=clock, model_id=model_id
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


def main(*args):
    with setup_resources(*args) as resources:
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Operation stopped by user.")
