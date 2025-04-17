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

import _thread
import base64
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from enum import Enum

import cv2
import numpy as np
import torch
import typer
from openai import OpenAI
from rich import print
from rich.logging import RichHandler
from transformers import AutoModelForImageTextToText, AutoProcessor
from typing_extensions import Annotated

from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES, activate_module
from owa.env.desktop.constants import VK
from owa.env.desktop.msg import KeyboardEvent
from owa.env.gst.msg import FrameStamped
from owa.env.gst.screen.listeners import MetricManager
from owa_game_agent.agent import Agent, AgentAPIClient
from owa_game_agent.commons import EvaluationResult, Task
from owa_game_agent.constants import NETWORK, TIMES
from owa_game_agent.data import OWATrainingSample
from owa_game_agent.data.datasets.smolvlm2 import sample_to_smolvlm_input
from owa_game_agent.data.sample_processor import SampleProcessor
from owa_game_agent.evaluator import Evaluator, EvaluatorAPIClient, EvaluatorState

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

env_lock = threading.Lock()  # mutex lock for controlling the environment

# --- Agent-Evaluator Implementation Examples --- #

FPS = 5


def get_current_frame_base64(window_name: str) -> tuple[str, np.ndarray]:
    """NOTE: this function is slow. Only use for evaluator, not agent."""
    screen_capture = RUNNABLES["screen_capture"]().configure(fps=FPS, window_name=window_name)
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


"""
(Recommended Implementation)
MySuperHexagonAgent and MySuperHexagonEvaluator: 
Agent does not decide task finish (returning only True for _play_env()), Evaluator decides task finish with _check_env_continue().
_check_env_continue() is synchronous, so if it is blocked, the subsequent _check_env_continue() will be blocked too.
_check_env_continue_timer() is asynchronous, so it will not block the subsequent _check_env_continue_timer().
Agent behavior is inferenced by a trained model.
"""


class MySuperHexagonAgent(Agent):
    """Example implementation of an agent"""

    def __init__(self, model_id: str):
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
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            # load_in_4bit=True,
            # _attn_implementation="flash_attention_2",
        ).to("cuda")

        def on_keyboard_event(keyboard_event: KeyboardEvent):
            if keyboard_event.vk == VK.F10:
                logger.debug("Stopping agent with F10 key")
                self.stop_event.set()
                _thread.interrupt_main()

        keyboard_listener = LISTENERS["keyboard"]().configure(callback=on_keyboard_event)
        keyboard_listener.start()

    def _init_task_hook(self, task: Task) -> None:
        """A customizable hook for task initialization. Called by _run_task(). Use with caution."""
        self.recent_frames = []
        self.num_frames = task.num_frames

        def on_screen_event(frame: FrameStamped, metric: MetricManager):
            self.recent_frames.append(frame)
            if len(self.recent_frames) > self.num_frames:
                self.recent_frames.pop(0)

        self.screen_listener = LISTENERS["screen"]().configure(
            callback=on_screen_event, fps=FPS, window_name="Super Hexagon"
        )
        self.screen_listener.start()

    def _finish_task_hook(self, task: Task) -> None:
        """A customizable hook for task finishing. Called by _run_task(). Use with caution."""
        self.screen_listener.stop()
        self.screen_listener.join(TIMES.THREAD_JOIN_TIMEOUT)
        self.recent_frames.clear()
        self.num_frames = None
        logger.debug("_finish_task_hook() finished")

    def __get_recent_frames(self) -> list[FrameStamped]:
        while len(self.recent_frames) < self.num_frames:
            time.sleep(TIMES.BUSY_WAIT_PREVENT_AGENT)  # wait for the frames to be collected
        return self.recent_frames[-self.num_frames :]

    # TODO : generate_response() and execute() seems to be something that should be included in library
    def __generate_response(self, sample: OWATrainingSample) -> str:
        """Process the sample and generate a response using the VLM."""
        sample_processor = SampleProcessor()
        tokenized_sample = sample_processor.tokenize(sample)
        vlm_input = sample_to_smolvlm_input(tokenized_sample)

        example = {"messages": vlm_input.messages, "images": vlm_input.images}
        examples = [example]
        texts = []
        images = []

        for ex in examples:
            assistant_prompt = ex["messages"].pop(-1)  # noqa: F841
            texts.append(
                self.processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True) + " "
            )
            images.append(ex["images"])

        # profile: 38.7%
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(
            self.model.device, dtype=self.model.dtype
        )

        # profile: 57.9%
        outputs = self.model.generate(**batch, logits_processor=[], do_sample=False, max_new_tokens=64)

        output = outputs[0]
        generated = self.processor.decode(output, skip_special_tokens=True)
        return generated[generated.find("Assistant: ") + len("Assistant: ") :]

    def __execute(self, generated: str, anchor_time: float, processing_time: float, task: Task) -> None:
        """Execute the generated response as scheduled keyboard actions."""
        tokens = re.findall(r"<(.*?)>", generated)
        timestamp_list = []
        events = []

        # Parse tokens: build a list of timestamps and pair them with keyboard actions.
        for token in tokens:
            if token.startswith("TIMESTAMP"):
                timestamp = int(token.split("_")[1])
                # Each timestamp represents an absolute time: anchor_time + (timestamp * TIMESTAMP_INTERVAL)
                timestamp_list.append(anchor_time + timestamp * TIMES.TIMESTAMP_INTERVAL)
            elif token.startswith("KEYBOARD"):
                if not timestamp_list:
                    logger.warning("Found KEYBOARD without TIMESTAMP")
                    return
                ts = timestamp_list.pop(0)
                vk, state = map(int, token.split("_")[1:])
                events.append((ts, vk, state, token))
            else:
                logger.warning(f"Invalid token: {token}")
                return

        if not events:
            return

        # The first event's intended time
        base_timestamp = events[0][0]  # e.g. anchor_time + 12 * 0.05 = anchor_time + 0.6

        # Compute delay needed for the first event:
        # Ideally, we want to execute the first event at base_timestamp.
        # But if processing already took processing_time, then the remaining delay is:
        first_delay = (base_timestamp - anchor_time) - processing_time
        if first_delay < 0:
            first_delay = 0

        # Schedule the first event to run after first_delay seconds from now.
        first_event_execution_time = time.time() + first_delay

        # Execute events while preserving relative differences.
        for original_time, vk, state, token in events:
            # Calculate the time difference (delta) relative to the first event.
            delta = original_time - base_timestamp
            # The scheduled time for this event is:
            scheduled_time = first_event_execution_time + delta
            to_sleep = max(0, scheduled_time - time.time())
            logger.info(f"Sleeping for {to_sleep:.2f}s, processing time {processing_time:.2f}s, token {token}")
            time.sleep(to_sleep)

            # Execute the key action.
            with env_lock:
                # make the window active
                CALLABLES["window.make_active"](task.window_name)

                if state:
                    CALLABLES["keyboard.press"](vk)
                    logger.warning(f"key {vk} pressed {task.task_id=}")
                else:
                    CALLABLES["keyboard.release"](vk)
                    logger.warning(f"key {vk} released {task.task_id=}")

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

        # Check if the task should continue
        def check_continue(task: Task) -> bool:
            """
            Check if the task should continue.
            """
            # In this version, evaluator will monitor and check the task. agent does not need to report termination.
            return True

        if check_continue(task):
            # Get recent frames. For now we just use a blocking call to get the frames.
            start_frame_collect = time.time()
            frames = self.__get_recent_frames()
            frame_collect = time.time() - start_frame_collect
            logger.debug(f"__get_recent_frames() took {frame_collect}sec")

            # Get current keyboard state
            state_keyboard = CALLABLES["keyboard.get_state"]().buttons - {1, 2, 4}
            state_mouse = CALLABLES["mouse.get_state"]()
            state_screen = []
            for frame in frames:
                state_screen.append((frame.timestamp_ns, frame.frame_arr))  # NOTE: is it ok to use frame.timestamp_ns?

            # Store in sample
            sample = OWATrainingSample(
                state_keyboard=state_keyboard,
                state_mouse=state_mouse,
                state_screen=state_screen,
                action_keyboard=[],
                action_mouse=None,
            )

            # Process sample exactly as in 05_online_evaluation.py
            now = time.time()
            generated = self.__generate_response(sample)
            taken = time.time() - now
            logger.debug(f"`__generate_response()`: {generated=}, {taken=}sec")

            # Extract the action from the generated response
            self.__execute(generated, anchor_time=now, processing_time=taken, task=task)

            return True  # Continue the task

        else:
            # logger.error(f"{self._play_env.__name__}(): task should not continue. This should not happen.")
            return False  # Do not continue the task. Evaluation will be made by the evaluator.


class MySuperHexagonEvaluator(Evaluator):
    """Example implementation of an Evaluator"""

    def __init__(self, report_jsonl_file_path: str | None = None):
        super().__init__(report_jsonl_file_path=report_jsonl_file_path)

        def on_keyboard_event(keyboard_event: KeyboardEvent):
            if keyboard_event.vk == VK.F10:
                logger.debug("Stopping evaluator with F10 key")
                self.stop_event.set()
                _thread.interrupt_main()

        keyboard_listener = LISTENERS["keyboard"]().configure(callback=on_keyboard_event)
        keyboard_listener.start()

        self.openai_client = OpenAI()

    def _check_env_continue(self, task: Task) -> bool:
        """
        Check the environment for a task, to see if the task should continue. Blocks subsequent calls of this function.

        Returns:
            bool: True if the task should continue, False otherwise.

        Might be overrided by subclasses to check the environment for a task.
        """
        return True  # for this example, we do not use _check_env_continue(), since we want to check the environment in a non-blocking way

    def _check_env_continue_timer(self, task: Task) -> bool:
        """
        Check the environment for a task, to see if the task should continue. Does not block subsequent calls of this function.

        Returns:
            bool: True if the task should continue, False otherwise.

        Might be overrided by subclasses to check the environment for a task.
        """
        # In this version, evaluator will monitor and check the task. agent does not need to report termination.

        # check if it is called in the correct interval
        logger.debug(f"`_check_env_continue_timer()` called at: {time.time()=}, {task.check_env_interval_seconds=}")

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

        logger.debug(f"GPT response: {response.output_text}")

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

        logger.debug(f"GPT response: {response.output_text}")

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

        with env_lock:
            # make the window active
            CALLABLES["window.make_active"](task.window_name)

            CALLABLES["keyboard.release_all_keys"]()  # release all keys

            # for super hexagon, we need to press space
            CALLABLES["keyboard.press"](VK.SPACE)
            time.sleep(TIMES.KEYBOARD_PRESS_DELAY)
            CALLABLES["keyboard.release"](VK.SPACE)


"""
(Less Recommended Implementation)
YourSuperHexagonAgent and YourSuperHexagonEvaluator: 
Agent decides task finish by the return value of _play_env(), Evaluator only scores after task finish.
Agent only presses the right arrow key.
"""


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

            logger.debug(f"GPT response: {response.output_text}")

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
            time.sleep(TIMES.KEYBOARD_PRESS_DELAY)
            CALLABLES["keyboard.release"](VK.RIGHT)
            # logger.debug(f"key {VK.RIGHT} pressed")

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

        logger.debug(f"GPT response: {response.output_text}")

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
        time.sleep(TIMES.KEYBOARD_PRESS_DELAY)
        CALLABLES["keyboard.release"](VK.SPACE)


# --- Example Tasks --- #


def generate_example_task() -> Task:
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
        check_env_interval_seconds=2.0,
        success_criteria={"time_survived": 10},
    )


def generate_example_task_2() -> Task:
    """
    Get an example task for testing.

    Returns:
        Task: An example task configuration.
    """
    return Task(
        env_name="Hexagon Super",
        window_name="Hexagon Super",
        task_description="Survive as long as possible. Maximum time is 60 seconds. A survival time over 10 seconds is considered a success.",
        timeout=60,
        check_env_interval_seconds=1.0,
        success_criteria={"time_survived": 10},
    )


# --- Run Functions --- #


def run_agent(model_id: str, host: str = NETWORK.DEFAULT_HOST, port: int = NETWORK.AGENT_PORT):
    """
    Run the agent server. Blocking.

    Args:
        model_id (str): The model ID to use for the agent.
    """
    agent = MySuperHexagonAgent(model_id=model_id)
    print("Starting agent server")
    agent.run(host=host, port=port)


def run_evaluator(host: str = NETWORK.DEFAULT_HOST, port: int = NETWORK.EVALUATOR_PORT):
    """
    Run the evaluator server. Blocking.
    """
    evaluator = MySuperHexagonEvaluator()
    print("Starting evaluator server")
    evaluator.run(host=host, port=port)


def run_evaluation_client(agent_url: str, evaluator_url: str, task_generator: Callable[[], Task]):
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

    # Start evaluation. We run 3 tasks consecutively for demonstration purposes.
    for _ in range(3):
        task = task_generator()

        # wait for evaluator to be ready
        while evaluator_client.get_status()["state"] != EvaluatorState.READY.name:
            print(f"Evaluator is not ready: {evaluator_client.get_status()}")
            time.sleep(TIMES.EVALUATION_POLL_INTERVAL)
        print(f"Evaluator is ready: {evaluator_client.get_status()}")

        if not evaluator_client.start_evaluation(task):
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
                time.sleep(TIMES.EVALUATION_POLL_INTERVAL)
            except KeyboardInterrupt:
                print("Interrupted by user. Exiting.")
                break


def run_evaluation_client_with_server(
    model_id: str,
    task_generator: Callable[[], Task],
    agent_host: str = NETWORK.DEFAULT_HOST,
    agent_port: int = NETWORK.AGENT_PORT,
    evaluator_host: str = NETWORK.DEFAULT_HOST,
    evaluator_port: int = NETWORK.EVALUATOR_PORT,
):
    """
    Run an example evaluation without blocking

    This function starts both the agent and evaluator servers in background threads,
    waits for them to be ready, and then runs an example evaluation against them.
    """
    # Create agent and evaluator instances
    agent = MySuperHexagonAgent(model_id=model_id)
    evaluator = MySuperHexagonEvaluator(report_jsonl_file_path="report.jsonl")

    # Start agent server in background
    print("Starting agent server...")
    agent_url = agent.run_background(host=agent_host, port=agent_port)
    if not agent_url:
        print("Failed to start agent server. Exiting.")
        return

    # Start evaluator server in background
    print("Starting evaluator server...")
    evaluator_url = evaluator.run_background(host=evaluator_host, port=evaluator_port)
    if not evaluator_url:
        print("Failed to start evaluator server. Exiting.")
        return

    # Short delay to ensure servers are ready
    time.sleep(TIMES.SERVER_STARTUP_RETRY_INTERVAL)

    # Run evaluation
    print("Running evaluation...")
    run_evaluation_client(agent_url, evaluator_url, task_generator)

    # Wait for user input to exit
    input("Press Enter to exit...")


def run_evaluation_client_with_server_parallel(model_id: str):
    """
    Run an example evaluation without blocking

    This function runs run_evaluation_client_with_server() in parallel for multiple tasks.
    """
    task_generators = [generate_example_task, generate_example_task_2]

    threads = []
    port = 8181
    for task_generator in task_generators:
        thread = threading.Thread(
            target=run_evaluation_client_with_server,
            args=(
                model_id,
                task_generator,
                NETWORK.DEFAULT_HOST,
                port,
                NETWORK.DEFAULT_HOST,
                port + 1,
            ),
        )
        thread.start()
        threads.append(thread)
        port += 2

    for thread in threads:
        thread.join()


class Mode(Enum):
    AGENT = "agent"
    EVALUATOR = "evaluator"
    RUN_CLIENT = "run_client"
    RUN_CLIENT_WITH_SERVER = "run_client_with_server"
    RUN_PARALLEL = "run_parallel"


def main(
    mode: Annotated[
        Mode,
        typer.Option(
            help="Mode to run: 'agent', 'evaluator', 'run_client', 'run_client_with_server', 'run_parallel'",
        ),
    ] = Mode.RUN_CLIENT_WITH_SERVER.value,
    model_id: Annotated[
        str, typer.Option(help="Model ID or path to use for the agent")
    ] = "__ignore__/SmolVLM2-256M-Video-Instruct-1e-4-10ep-8bs-3IMG",
):
    """
    Main entry point.

    Args:
        mode (Mode): The mode to run in.
        model_id (str): The model ID or path to use for the agent.
    """
    if mode == Mode.AGENT:
        run_agent(model_id)
    elif mode == Mode.EVALUATOR:
        run_evaluator()
    elif mode == Mode.RUN_CLIENT:
        run_evaluation_client(NETWORK._AGENT_URL, NETWORK._EVALUATOR_URL)
    elif mode == Mode.RUN_CLIENT_WITH_SERVER:
        run_evaluation_client_with_server(model_id, generate_example_task)
    elif mode == Mode.RUN_PARALLEL:
        run_evaluation_client_with_server_parallel(model_id)
    else:
        raise ValueError(f"Unknown mode: {mode=}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    typer.run(main)
