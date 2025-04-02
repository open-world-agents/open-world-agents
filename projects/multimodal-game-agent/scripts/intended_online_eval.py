"""
This script demonstrates the intended usage and interaction between Agent and Evaluator components
for online evaluation of environment(such as games) agents. Both components run independently and communicate via HTTP,
following a finite state machine design pattern.

Key Features:
- Independent Agent and Evaluator processes
- HTTP-based communication (FastAPI)
- Simple API for researchers to implement their own agents
- Task-based evaluation with automated scoring
- Minimal asynchronous programming requirements for researchers
"""

import logging
import time
from enum import Enum

import cv2
import typer
from rich.logging import RichHandler
from typing_extensions import Annotated

from owa.core.registry import CALLABLES, RUNNABLES, activate_module
from owa.env.desktop.constants import VK
from owa.env.gst.msg import FrameStamped
from owa_game_agent.agent import Agent, AgentAPIClient
from owa_game_agent.commons import EvaluationResult, Task
from owa_game_agent.constants import DEFAULTS, NETWORK, TIMEOUTS
from owa_game_agent.evaluator import Evaluator, EvaluatorAPIClient, EvaluatorState
from rich import print

logging.basicConfig(
    level=logging.DEBUG,
    format="(%(asctime)s) [%(name)s]:\n %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)],
)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- Agent-Evaluator Implementation Examples --- #


class MySuperHexagonAgent(Agent):
    """Example implementation of an agent"""

    def __init__(self):
        super().__init__()

        # Example: Super Hexagon implementation would use screen observations
        # and generate keyboard inputs based on those observations
        activate_module(
            "owa.env.desktop"
        )  # https://open-world-agents.github.io/open-world-agents/env/plugins/desktop_env/
        activate_module(
            "owa.env.gst"
        )  # https://open-world-agents.github.io/open-world-agents/env/plugins/gstreamer_env/

    def _play_env(self, task: Task) -> bool:
        """
        Implement the environment playing logic here for a single step.
        This is what researchers would customize with their models and logic.

        Args:
            task (Task): The task configuration.

        Returns:
            bool: True if the task should continue, False if the task is complete
        """

        # Check if window is active
        window_active = CALLABLES["window.is_active"](task.window_name)
        # if not window_active:
        #     logger.debug(f"Window {task.window_name} is not active")
        #     return True  # Continue the task

        # Example: Get screen state and make decisions
        # This would use your ML model to generate actions

        # Example keyboard input (pressing right arrow)
        CALLABLES["keyboard.press"](VK.RIGHT)  # Right arrow
        time.sleep(DEFAULTS.KEYBOARD_PRESS_DELAY)
        CALLABLES["keyboard.release"](VK.RIGHT)
        logger.debug(f"key {VK.RIGHT} pressed")

        # Check if the success condition for the task has been met
        def check_success_condition(task: Task) -> bool:
            """
            Check if the success condition for the task has been met.
            """
            # This would implement environment-specific success detection
            # For example, detecting a "victory" screen or a specific score
            # using task.success_criteria
            return False

        if check_success_condition(task):
            logger.debug(f"{self._play_env.__name__} finished: Success condition met")
            return False  # Do not continue the task. Evaluation will be made by the evaluator.

        return True  # Continue the task


class MyEvaluator(Evaluator):
    """Example implementation of an Evaluator"""

    def _score_task(self, task: Task, task_elapsed_time: float, note: str = "") -> EvaluationResult:
        """
        Simple scoring based on task task_elapsed_time and success criteria.

        Args:
            task (Task): The task configuration that was completed
            task_elapsed_time (float): How long the task took

        Returns:
            EvaluationResult: The scoring results
        """

        # Reference Task for configurations
        success_criteria = task.success_criteria

        # Here you would check if the agent actually achieved the success criteria
        # This could involve checking game state, screenshot analysis, etc.
        # In a real evaluator, you might capture screenshots or other artifacts for scoring
        CALLABLES["screen.capture"]

        # For this example, we'll just assume success if they finished before timeout
        success = task_elapsed_time < task.timeout

        return EvaluationResult(
            task_id=task.task_id,
            metrics={
                "time": task_elapsed_time,
                "success": success,
            },
            notes=note,
        )

    def _setup_environment(self, task: Task):
        """
        Setup the environment for a task. Also handles restarting the environment.

        Args:
            task (Task): Configuration for the task to setup
        """
        logger.debug(f"Setting up environment for {task.env_name}")
        # In a real implementation, this would launch games, configure windows, etc.
        # Also handles restarting the environment.


# --- Example Tasks --- #


def get_example_task() -> Task:
    """
    Get an example task for testing.

    Returns:
        Task: An example task configuration.
    """
    return Task(
        env_name="Super Hexagon",
        window_name="Super Hexagon",
        task_description="Survive as long as possible",
        timeout=1,
        success_criteria={"time_survived": 1},
    )
    # Task(
    #     env_name="ZType",
    #     window_name="ZType",
    #     task_description="Complete the first level with at least 90% accuracy",
    #     timeout=120,
    #     success_criteria={"accuracy": 0.9, "level_completed": True},
    # ),


# --- Run Functions --- #


def run_agent():
    """
    Run the agent server. Blocking.

    Args:
        model_id (str): The model ID to use for the agent.
    """
    agent = MySuperHexagonAgent()
    print("Starting agent server")
    agent.run(host=NETWORK.DEFAULT_HOST, port=NETWORK.AGENT_PORT)


def run_evaluator():
    """
    Run the evaluator server. Blocking.
    """
    evaluator = MyEvaluator()
    print("Starting evaluator server")
    evaluator.run(host=NETWORK.DEFAULT_HOST, port=NETWORK.EVALUATOR_PORT)


def run_evaluation_client(agent_url: str, evaluator_url: str):
    """
    Run an example evaluation. Assumes the agent and evaluator servers are already running.

    Args:
        agent_url (str): The URL of the agent server.
        evaluator_url (str): The URL of the evaluator server.
    """
    # Create API clients
    evaluator_client = EvaluatorAPIClient(evaluator_url)
    agent_client = AgentAPIClient(agent_url)

    # Register evaluator with agent
    print("Registering evaluator with agent...")
    if not agent_client.register_evaluator(evaluator_url):
        print("Failed to register evaluator with agent. Exiting.")
        return

    # Register agent with evaluator
    print("Registering agent with evaluator...")
    if not evaluator_client.register_agent(agent_url):
        print("Failed to register agent with evaluator. Exiting.")
        return

    # Get example task
    example_task = get_example_task()

    # Start evaluation
    for _ in range(3):
        # wait for evaluator to be ready
        while evaluator_client.get_status()["state"] != EvaluatorState.READY.name:
            print(f"Evaluator is not ready: {evaluator_client.get_status()}")
            time.sleep(TIMEOUTS.EVALUATION_POLL_INTERVAL)
        print(f"Evaluator is ready: {evaluator_client.get_status()}")

        print(f"Starting evaluation with task: {example_task.env_name}...")
        if not evaluator_client.start_evaluation(example_task):
            print("Failed to start evaluation. Exiting.")
            return

        # Wait for evaluation to complete
        print("Waiting for evaluation to complete...")
        while True:
            try:
                # Poll for results
                results = evaluator_client.get_results()
                if results:
                    print(f"Evaluation completed with results: {results}")
                    break
                time.sleep(TIMEOUTS.EVALUATION_POLL_INTERVAL)
            except KeyboardInterrupt:
                print("Interrupted by user. Exiting.")
                break


def run_evaluation_client_with_server():
    """
    Run an example evaluation without blocking

    This function starts both the agent and evaluator servers in background threads,
    waits for them to be ready, and then runs an example evaluation against them.
    """
    # Create agent and evaluator instances
    agent = MySuperHexagonAgent()
    evaluator = MyEvaluator()

    # Start agent server in background
    print("Starting agent server...")
    agent_url = agent.run_background(host=NETWORK.DEFAULT_HOST, port=NETWORK.AGENT_PORT)
    if not agent_url:
        print("Failed to start agent server. Exiting.")
        return

    # Start evaluator server in background
    print("Starting evaluator server...")
    evaluator_url = evaluator.run_background(host=NETWORK.DEFAULT_HOST, port=NETWORK.EVALUATOR_PORT)
    if not evaluator_url:
        print("Failed to start evaluator server. Exiting.")
        return

    # Short delay to ensure servers are ready
    time.sleep(TIMEOUTS.SERVER_STARTUP_RETRY_INTERVAL)

    # Run evaluation
    print("Running evaluation...")
    run_evaluation_client(agent_url, evaluator_url)

    # Wait for user input to exit
    input("Press Enter to exit...")


class Mode(Enum):
    AGENT = "agent"
    EVALUATOR = "evaluator"
    RUN_CLIENT = "run_client"
    RUN_CLIENT_WITH_SERVER = "run_client_with_server"


def main(
    mode: Annotated[
        Mode,
        typer.Option(
            help="Mode to run: 'agent', 'evaluator', 'run_client', or 'run_client_with_server'",
        ),
    ] = Mode.RUN_CLIENT_WITH_SERVER.value,
):
    """
    Main entry point.

    Args:
        mode (Mode): The mode to run in.
        model_id (str): The model ID to use for the agent.
    """
    if mode == Mode.AGENT:
        run_agent()
    elif mode == Mode.EVALUATOR:
        run_evaluator()
    elif mode == Mode.RUN_CLIENT:
        run_evaluation_client(NETWORK._AGENT_URL, NETWORK._EVALUATOR_URL)
    elif mode == Mode.RUN_CLIENT_WITH_SERVER:
        run_evaluation_client_with_server()
    else:
        raise ValueError(f"Unknown mode: {mode=}")


if __name__ == "__main__":
    typer.run(main)
