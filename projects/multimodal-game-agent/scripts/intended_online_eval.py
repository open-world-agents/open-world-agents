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

import base64
import json
import logging
import time
from enum import Enum

import cv2
import typer
from openai import OpenAI
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
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- Agent-Evaluator Implementation Examples --- #


def get_current_frame_base64(window_name: str) -> str:
    screen_capture = RUNNABLES["screen_capture"]().configure(fps=60, window_name=window_name)
    with screen_capture.session:
        # get a single frame
        frame: FrameStamped = screen_capture.grab()

    # just for debugging, display the frame
    # cv2.imshow("frame", frame.frame_arr)
    # cv2.waitKey(100)  # wait for 100ms
    # cv2.destroyAllWindows()

    # change the frame to base64
    frame_base64 = cv2.imencode(".png", frame.frame_arr)[1].tobytes()
    frame_base64 = base64.b64encode(frame_base64).decode("utf-8")
    return frame_base64


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
        self.openai_client = OpenAI()

    def _play_env(self, task: Task) -> bool:
        """
        Implement the environment playing logic here for a single step.

        Must be implemented by subclasses.
        This is what researchers would customize with their models and logic.

        Args:
            task (Task): The task configuration.

        Returns:
            bool: True if the task should continue, False if the task should not continue
        """

        # make the window active
        CALLABLES["window.make_active"](task.window_name)

        # Check if the task should continue
        def check_continue(task: Task) -> bool:
            """
            Check if the task should continue.
            """
            # In this version, evaluator will monitor and check the task. agent does not need to report termination.
            return True

        if check_continue(task):
            # logger.debug(f"{self._play_env.__name__}(): task should continue")

            # Normally, you would use your model to generate actions

            # Currently, this agent just presses the right arrow key
            # Example keyboard input (pressing right arrow)
            CALLABLES["keyboard.press"](VK.RIGHT)  # Right arrow
            time.sleep(DEFAULTS.KEYBOARD_PRESS_DELAY)
            CALLABLES["keyboard.release"](VK.RIGHT)
            logger.debug(f"key {VK.RIGHT} pressed")

            return True  # Continue the task

        else:
            # logger.error(f"{self._play_env.__name__}(): task should not continue. This should not happen.")
            return False  # Do not continue the task. Evaluation will be made by the evaluator.


class MySuperHexagonEvaluator(Evaluator):
    """Example implementation of an Evaluator"""

    def __init__(self):
        super().__init__()
        self.openai_client = OpenAI()

    def _check_env_continue(self, task: Task) -> bool:
        """
        Check the environment for a task, to see if the task should continue.

        Returns:
            bool: True if the task should continue, False otherwise.

        Might be overrided by subclasses to check the environment for a task.
        """
        # In this version, evaluator will monitor and check the task. agent does not need to report termination.

        # we can also utilize task information
        # prompt = f"""Did you survive for at least {task.success_criteria["time_survived"]} seconds? Say only yes or no."""
        prompt = "This is a screenshot of a game called Super Hexagon. Can you see the text `PRESS SPACE TO RETRY` in the bottom? Say only yes or no."
        frame_base64 = get_current_frame_base64(task.window_name)

        # use gpt-4o to check if the task should continue
        response = self.openai_client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{frame_base64}",
                            "detail": "auto",
                        },
                    ],
                }
            ],
            temperature=0.0,
        )

        logger.debug(f"response: {response.output_text}")

        if "yes" in response.output_text.lower():
            return False  # if we can see the retry button, the task should not continue
        else:
            return True  # if we cannot see the retry button, the task should continue

    def _score_task(self, task: Task, task_elapsed_time: float, note: str = "") -> EvaluationResult:
        """
        Calculate score for a completed task.

        Args:
            task (Task): The task configuration that was completed
            task_elapsed_time (float): How long the task took

        Returns:
            EvaluationResult: The scoring results
        """

        # Here you would check if the agent actually achieved the success criteria
        # This could involve checking game state, screenshot analysis, etc.

        prompt = "This is a screenshot of a game called Super Hexagon. What is the score? It should be written next to `LAST`, and should be in the format of `%d:%d` up to 2 decimals. If you cannot find the score, the score should be `-1`."
        frame_base64 = get_current_frame_base64(task.window_name)

        # use gpt-4o to evaluate the score
        response = self.openai_client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{frame_base64}",
                            "detail": "auto",
                        },
                    ],
                }
            ],
            temperature=0.0,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "score",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "number"},
                        },
                        "required": ["score"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )

        logger.debug(f"response: {response.output_text}")

        try:
            score = json.loads(response.output_text)["score"]
        except json.JSONDecodeError:
            score = -1

        if score < 0:
            success = False
        else:
            if score > task.success_criteria["time_survived"]:
                success = True
            else:
                success = False

        return EvaluationResult(
            task_id=task.task_id,
            metrics={
                "task_elapsed_time": task_elapsed_time,
                "success": success,
                "score": score,
            },
            notes=note,
        )

    def _setup_environment(self, task: Task):
        """
        Setup the environment for a task. Also should handle restarting the environment.

        Args:
            task (Task): Configuration for the task to setup
        """
        logger.debug(f"Setting up environment for {task.env_name}")
        # In a real implementation, this would launch games, configure windows, etc.
        # Also handles restarting the environment.

        # make the window active
        CALLABLES["window.make_active"](task.window_name)

        # for super hexagon, we need to press space
        CALLABLES["keyboard.press"](VK.SPACE)
        time.sleep(DEFAULTS.KEYBOARD_PRESS_DELAY)
        CALLABLES["keyboard.release"](VK.SPACE)


class YourSuperHexagonAgent(Agent):
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
        self.openai_client = OpenAI()

    def _play_env(self, task: Task) -> bool:
        """
        Implement the environment playing logic here for a single step.

        Must be implemented by subclasses.
        This is what researchers would customize with their models and logic.

        Args:
            task (Task): The task configuration.

        Returns:
            bool: True if the task should continue, False if the task should not continue
        """

        # make the window active
        CALLABLES["window.make_active"](task.window_name)

        # Check if the task should continue
        def check_continue(task: Task) -> bool:
            """
            Check if the task should continue.
            """
            # This would implement environment-specific conditions
            # For example, detecting a "game over" screen or a specific score

            # we can also utilize task information
            # prompt = f"""Did you survive for at least {task.success_criteria["time_survived"]} seconds? Say only yes or no."""
            prompt = "This is a screenshot of a game called Super Hexagon. Can you see the text `PRESS SPACE TO RETRY` in the bottom? Say only yes or no."
            frame_base64 = get_current_frame_base64(task.window_name)

            # use gpt-4o to check if the task should continue
            response = self.openai_client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt,
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{frame_base64}",
                                "detail": "auto",
                            },
                        ],
                    }
                ],
                temperature=0.0,
            )

            logger.debug(f"response: {response.output_text}")

            if "yes" in response.output_text.lower():
                return False  # if we can see the retry button, the task should not continue
            else:
                return True  # if we cannot see the retry button, the task should continue

        if check_continue(task):
            logger.debug(f"{self._play_env.__name__}(): task should continue")

            # Normally, you would use your model to generate actions

            # Currently, this agent just presses the right arrow key
            # Example keyboard input (pressing right arrow)
            CALLABLES["keyboard.press"](VK.RIGHT)  # Right arrow
            time.sleep(DEFAULTS.KEYBOARD_PRESS_DELAY)
            CALLABLES["keyboard.release"](VK.RIGHT)
            logger.debug(f"key {VK.RIGHT} pressed")

            return True  # Continue the task

        else:
            logger.debug(f"{self._play_env.__name__}(): task should not continue")
            return False  # Do not continue the task. Evaluation will be made by the evaluator.


class YourSuperHexagonEvaluator(Evaluator):
    """Example implementation of an Evaluator"""

    def __init__(self):
        super().__init__()
        self.openai_client = OpenAI()

    def _score_task(self, task: Task, task_elapsed_time: float, note: str = "") -> EvaluationResult:
        """
        Calculate score for a completed task.

        Args:
            task (Task): The task configuration that was completed
            task_elapsed_time (float): How long the task took

        Returns:
            EvaluationResult: The scoring results
        """

        # Here you would check if the agent actually achieved the success criteria
        # This could involve checking game state, screenshot analysis, etc.

        prompt = "This is a screenshot of a game called Super Hexagon. What is the score? It should be written next to `LAST`, and should be in the format of `%d:%d` up to 2 decimals. If you cannot find the score, the score should be `-1`."
        frame_base64 = get_current_frame_base64(task.window_name)

        # use gpt-4o to evaluate the score
        response = self.openai_client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{frame_base64}",
                            "detail": "auto",
                        },
                    ],
                }
            ],
            temperature=0.0,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "score",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "number"},
                        },
                        "required": ["score"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )

        logger.debug(f"response: {response.output_text}")

        try:
            score = json.loads(response.output_text)["score"]
        except json.JSONDecodeError:
            score = -1

        if score < 0:
            success = False
        else:
            if score > task.success_criteria["time_survived"]:
                success = True
            else:
                success = False

        return EvaluationResult(
            task_id=task.task_id,
            metrics={
                "task_elapsed_time": task_elapsed_time,
                "success": success,
                "score": score,
            },
            notes=note,
        )

    def _setup_environment(self, task: Task):
        """
        Setup the environment for a task. Also should handle restarting the environment.

        Args:
            task (Task): Configuration for the task to setup
        """
        logger.debug(f"Setting up environment for {task.env_name}")
        # In a real implementation, this would launch games, configure windows, etc.
        # Also handles restarting the environment.

        # make the window active
        CALLABLES["window.make_active"](task.window_name)

        # for super hexagon, we need to press space
        CALLABLES["keyboard.press"](VK.SPACE)
        time.sleep(DEFAULTS.KEYBOARD_PRESS_DELAY)
        CALLABLES["keyboard.release"](VK.SPACE)


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
        task_description="Survive as long as possible. Maximum time is 60 seconds. A survival time over 10 seconds is considered a success.",
        timeout=60,
        success_criteria={"time_survived": 10},
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
    evaluator = MySuperHexagonEvaluator()
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
    evaluator = MySuperHexagonEvaluator()

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
    from dotenv import load_dotenv

    load_dotenv()
    typer.run(main)
