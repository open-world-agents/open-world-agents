import sys
import time
from contextlib import contextmanager
from queue import Queue

import typer
from loguru import logger

from owa.agent.core import Clock
from owa.agent.core.perception import PerceptionQueue
from owa.agent.systems.goat import (
    PERCEPTION_SPEC_DICT,
    ActionExecutor,
    EventProcessor,
    ModelWorker,
    PerceptionProvider,
    RealTimeAgentCoordinator,
)

# TODO: init from yaml, with configurable provider/coordinator/worker/action_executor/...
# TODO: add EventProcessor init from config


def configure_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        filter={"owa.env.gst": False},
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add("launch.log", level="TRACE", encoding="utf-8")


@contextmanager
def setup_resources():
    configure_logging()

    perception_queue = PerceptionQueue()
    thought_queue = Queue(maxsize=1)  # no task parallelism in model worker
    decision_queue = Queue()
    action_queue = Queue()
    event_processor = EventProcessor()
    clock = Clock(scale=1.0)

    perception_provider = PerceptionProvider().configure(perception_queue=perception_queue, clock=clock)
    agent_coordinator = RealTimeAgentCoordinator().configure(
        perception_queue=perception_queue,
        thought_queue=thought_queue,
        decision_queue=decision_queue,
        action_queue=action_queue,
        perception_spec_dict=PERCEPTION_SPEC_DICT,
        event_processor=event_processor,
        rate=20.0,
        clock=clock,
        world_pause=False,  # Set to True to pause the world. If set to True, adjust rate to 4.0/2.0
    )
    model_worker = ModelWorker().configure(
        thought_queue=thought_queue,
        decision_queue=decision_queue,
        clock=clock,
        model_id=r"C:\Users\MilkClouds\Downloads\SmolVLM2-256M",
    )
    action_executor = ActionExecutor().configure(action_queue=action_queue, clock=clock, preempt=True)

    resources = [
        (perception_provider, "perception_provider"),
        (agent_coordinator, "agent_coordinator"),
        (model_worker, "model_worker"),
        (action_executor, "action_executor"),
    ]
    for resource, name in resources:
        resource.start()
        logger.info(f"Started {name}")
    try:
        yield resources
    finally:
        for resource, name in reversed(resources):
            try:
                resource.stop()
                resource.join(timeout=5)
                assert not resource.is_alive(), f"{name} is still alive after stop"
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
