import logging
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from owa_game_agent.agent import AgentAPIClient
from owa_game_agent.commons import EvaluationResult, Task, handle_response_errors, run_server_background
from owa_game_agent.constants import ENDPOINTS, NETWORK, TIMES

logger = logging.getLogger(__name__)


class EvaluatorState(Enum):
    """States for the Evaluator state machine"""

    INIT = auto()
    READY = auto()
    EVALUATING = auto()
    SCORING = auto()


class Evaluator(ABC):
    """
    Abstract base class for evaluators of agents.

    Handles setting up tasks, monitoring agent behavior, and scoring performance.
    The child class must implement the _score_task() and _setup_environment() methods.
    """

    def __init__(self, report_jsonl_file_path: Optional[Path] = None):
        """
        Initialize the evaluator.
        """
        self._state = EvaluatorState.INIT
        self.evaluator_url = None
        self.agent_url = None
        self.agent_api_client: AgentAPIClient = None
        self.current_task = None
        self.result = None
        self.result_lock = threading.Lock()  # Lock for self.result
        self.task_start_time = None
        self.task_thread = None
        self.stop_event = threading.Event()
        self.api_server = None
        self.report_jsonl_file_path = report_jsonl_file_path  # Optional report file for logging

    @property
    def state(self) -> EvaluatorState:
        """Getter for the evaluator state."""
        return self._state

    @state.setter
    def state(self, new_state: EvaluatorState):
        """Setter for the evaluator state with debug logging."""
        old_state = self._state
        self._state = new_state
        logger.debug(f"Evaluator state changed: {old_state.name} -> {new_state.name}")

    def _run_evaluation(self, task: Task) -> bool:
        """
        Run the evaluation process.

        Args:
            task (Task): The task configuration.

        Returns:
            bool: True if the evaluation was run successfully, False otherwise.
        """
        if self.state != EvaluatorState.READY:
            logger.error("Evaluator not in READY state. Cannot run evaluation.")
            return False

        # Check if agent is ready
        if not self.agent_api_client.check_ready():
            logger.error("Agent not ready. Aborting evaluation.")
            return False

        # Check if the task has already been run previously
        if self.current_task and self.current_task.task_id == task.task_id:
            logger.error("Detected reuse of previously run task. Please create a new task object.")
            return False

        # Setup environment for the task
        try:
            self._setup_environment(task)
        except Exception as e:
            logger.error(f"Error setting up environment: {e}")
            return False

        # Start the agent on this task
        if not self.agent_api_client.start_task(task):
            logger.error(f"Failed to start task on agent: {task.env_name}")
            return False

        if self.task_thread and self.task_thread.is_alive():
            logger.error("Task thread is still alive in READY state. This should not happen.")
            return False

        self.stop_event.clear()
        self.current_task = task
        self.task_start_time = time.time()
        self.result = None
        self.state = EvaluatorState.EVALUATING

        # Start monitoring thread
        self.task_thread = threading.Thread(target=self._monitor_task, args=(self.current_task,), daemon=True)
        self.task_thread.start()

        return True

    def _monitor_task(self, task: Task):
        """
        Monitor a running task and handle timeouts.

        Args:
            task (Task): The task configuration.
        """

        # start a timeout thread that will stop the task after a certain time
        def timeout_watcher():
            time.sleep(task.timeout)

            if (
                self.current_task == task
            ):  # NOTE: check if the task is the current task. Needed since we cannot kill the timer thread, and a new task might be running
                logger.debug(f"Task timed out. Stopping task. {task=}")
                self.stop_event.set()
            else:
                logger.debug(f"Task timed out, but current task has changed. Ignoring. {self.current_task=} {task=}")

        timeout_thread = threading.Thread(target=timeout_watcher, daemon=True)
        timeout_thread.start()

        # start a timer thread that will spawn `_check_env_continue_timer()` every interval
        def timer_check():
            while not self.stop_event.is_set():
                if self.current_task != task or self.state != EvaluatorState.EVALUATING:
                    return

                # Spawn a new thread for each call to _check_env_continue_timer to prevent blocking
                def run_check():
                    try:
                        continue_task = self._check_env_continue_timer(task)
                        if not continue_task:
                            logger.debug("Task stopped by `_check_env_continue_timer()`. Stopping task.")
                            self.stop_event.set()
                    except Exception as e:
                        logger.error(f"Error in _check_env_continue_timer: {e}")

                # Create and start a new thread for this check
                check_thread = threading.Thread(target=run_check, daemon=True)
                check_thread.start()

                time.sleep(task.check_env_interval_seconds)
            return  # finish the thread when stop_event is set

        # Start the timer thread
        timer_thread = threading.Thread(target=timer_check, daemon=True)
        timer_thread.start()

        start_time = time.time()
        while not self.stop_event.is_set():
            # Check if the task has changed or if state has already changed
            if self.current_task != task or self.state != EvaluatorState.EVALUATING:
                return
            continue_task = self._check_env_continue(task)
            if not continue_task:
                logger.debug("Task stopped by `check_env_continue()`. Stopping task.")
                self.stop_event.set()
            time.sleep(
                TIMES.BUSY_WAIT_PREVENT_EVALUATOR
            )  # NOTE: Sleep to avoid busy waiting. Removing this will cause high CPU usage and interfere other threads.

        # stop event is set, meaning the task should be stopped
        assert self.stop_event.is_set(), "stop_event should be set when exiting the loop in `_monitor_task()`"
        if self.current_task == task and self.state == EvaluatorState.EVALUATING:
            elapsed = time.time() - start_time
            logger.debug(f"Task stopped after {elapsed} seconds")
            # Stop the agent task
            self.agent_api_client.stop_task()

            self.process_finished_task(note=f"`_monitor_task()`: Task stopped after {elapsed} seconds")
        else:
            # NOTE: if we don't check current_task, previous task timer might stop the current task
            logger.debug(
                f"Task stopped, but current task has changed or not in EVALUATING state. Ignoring. {self.current_task=} {task=}"
            )

    def process_finished_task(self, note: str = ""):
        """
        Process a finished task.

        This function might be called via a call from an agent (`agent_finished()`),
        or a timeout by the evaluator (`_monitor_task()`).
        We use double checked locking in self.result for thread safety.
        If self.result is already set, we do not set it again.

        Args:
            note (str): A note about the task completion.
        """
        logger.debug(f"process_finished_task(): {note=}")

        # First check without lock
        if self.result is None:
            # Acquire lock and check again (double-checked locking)
            with self.result_lock:
                if self.result is None:
                    if self.state != EvaluatorState.EVALUATING:
                        logger.warning(
                            f"`process_finished_task()`: Evaluator not in {EvaluatorState.EVALUATING.name} state, rather {self.state.name=}. Returning."
                        )
                        return

                    self.state = EvaluatorState.SCORING
                    task_elapsed_time = time.time() - self.task_start_time
                    self.result = self._score_task(self.current_task, task_elapsed_time, note)
                    logger.info(f"Task completed. Result: {self.result.model_dump()}")

                    # Log result to JSONL file if report_file_path is set
                    if self.report_jsonl_file_path:
                        try:
                            # Create directory if it doesn't exist
                            Path(self.report_jsonl_file_path).parent.mkdir(parents=True, exist_ok=True)
                            # Append the result as JSON to the file
                            with open(self.report_jsonl_file_path, "a") as f:
                                f.write(self.result.model_dump_json() + "\n")
                            logger.info(f"Evaluation result logged to {self.report_jsonl_file_path}")
                        except Exception as e:
                            logger.error(f"Failed to log evaluation result to file: {e}")
                    else:
                        logger.info("No report file path set. Skipping logging of evaluation result.")

                    self.state = EvaluatorState.READY
                else:
                    logger.debug("Task already completed. Skipping scoring.")
        else:
            logger.debug("Task already completed. Skipping scoring.")

    def _check_env_continue(self, task: Task) -> bool:
        """
        Check the environment for a task, to see if the task should continue. Blocks subsequent calls of this function.

        Returns:
            bool: True if the task should continue, False otherwise.

        Might be overrided by subclasses to check the environment for a task.
        """
        return True

    def _check_env_continue_timer(self, task: Task) -> bool:
        """
        Check the environment for a task, to see if the task should continue. Does not block subsequent calls of this function.

        Returns:
            bool: True if the task should continue, False otherwise.

        Might be overrided by subclasses to check the environment for a task.
        """
        return True

    @abstractmethod
    def _score_task(self, task: Task, task_elapsed_time: float, note: str = "") -> EvaluationResult:
        """
        Calculate score for a completed task.

        Must be implemented by subclasses.

        Args:
            task (Task): The task configuration that was completed
            task_elapsed_time (float): How long the task took

        Returns:
            EvaluationResult: The scoring results
        """
        ...

    @abstractmethod
    def _setup_environment(self, task: Task):
        """
        Setup the environment for a task. Also should handle restarting the environment.

        Must be implemented by subclasses.

        Args:
            task (Task): Configuration for the task to setup
        """
        ...

    def run(self, host: str = NETWORK.DEFAULT_HOST, port: int = NETWORK.EVALUATOR_PORT):
        """
        Run the evaluator server.

        Args:
            host (str): The host to run the server on.
            port (int): The port to run the server on.
        """
        self.api_server = EvaluatorAPIServer(self)
        self.evaluator_url = f"http://{host}:{port}"
        uvicorn.run(self.api_server.app, host=host, port=port)

    def run_background(self, host: str = NETWORK.DEFAULT_HOST, port: int = NETWORK.EVALUATOR_PORT) -> Optional[str]:
        """
        Run the evaluator server in a background thread and wait for it to be ready.

        Args:
            host (str): The host to run the server on.
            port (int): The port to run the server on.

        Returns:
            Optional[str]: The URL of the evaluator server if it started successfully, None otherwise.
        """
        return run_server_background(self.run, host, port, ENDPOINTS.EVALUATOR_STATUS)


class EvaluatorAPIServer:
    """API server for the Evaluator"""

    def __init__(self, evaluator: Evaluator):
        """Initialize the API server with a reference to the evaluator"""
        self.evaluator = evaluator
        self.app = FastAPI(title="Evaluator API")

        # Register API endpoints
        self.app.post(ENDPOINTS.EVALUATOR_REGISTER_AGENT)(self._register_agent)
        self.app.post(ENDPOINTS.EVALUATOR_EVALUATION_START)(self._start_evaluation)
        self.app.post(ENDPOINTS.EVALUATOR_AGENT_FINISHED)(self._agent_finished)
        self.app.get(ENDPOINTS.EVALUATOR_STATUS)(self._get_status)
        self.app.get(ENDPOINTS.EVALUATOR_EVALUATION_RESULTS)(self._get_results)

    class RegisterAgentRequest(BaseModel):
        agent_url: str

    def _register_agent(self, request: RegisterAgentRequest, response: Response):
        """Register an agent with the evaluator"""
        if self.evaluator.state != EvaluatorState.INIT:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {
                "success": False,
                "error": f"Evaluator not in INIT state. Current state: {self.evaluator.state.name}",
            }

        self.evaluator.agent_url = request.agent_url

        self.evaluator.agent_api_client = AgentAPIClient(self.evaluator.agent_url)
        self.evaluator.state = EvaluatorState.READY
        response.status_code = status.HTTP_200_OK
        return {"success": True, "message": f"Registered agent at {self.evaluator.agent_url}"}

    def _start_evaluation(self, task: Task, response: Response):
        """Start an evaluation with a task"""
        if self.evaluator.state != EvaluatorState.READY:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {
                "success": False,
                "error": f"Evaluator not waiting for agent. Current state: {self.evaluator.state.name}",
            }

        if not self.evaluator.agent_api_client:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"success": False, "error": "No agent registered"}

        if not self.evaluator._run_evaluation(task):
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"success": False, "error": "Failed to start evaluation"}
        else:
            response.status_code = status.HTTP_200_OK
            return {"success": True, "message": "Started evaluation"}

    def _agent_finished(self, response: Response):
        """Called when the agent finishes a task"""
        if self.evaluator.state == EvaluatorState.EVALUATING:
            self.evaluator.process_finished_task(note="`_agent_finished()`: Agent reported task completion")
            response.status_code = status.HTTP_200_OK
            return {"success": True, "message": "Task completion acknowledged"}
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {
                "success": False,
                "error": f"Unexpected state: {self.evaluator.state.name}. Expected: EVALUATING",
            }

    def _get_status(self, response: Response):
        """Get the current status of the evaluation"""
        response.status_code = status.HTTP_200_OK
        return {
            "state": self.evaluator.state.name,
            "current_task": self.evaluator.current_task.model_dump() if self.evaluator.current_task else None,
        }

    def _get_results(self, response: Response):
        """Get the results of the evaluation"""
        if self.evaluator.result:
            response.status_code = status.HTTP_200_OK
            return self.evaluator.result.model_dump()
        else:
            response.status_code = status.HTTP_204_NO_CONTENT
            return None


class EvaluatorAPIClient:
    """API client for interacting with the Evaluator server"""

    def __init__(self, evaluator_url: str):
        """
        Initialize the API client with the evaluator URL.

        Args:
            evaluator_url (str): The URL of the evaluator server.
        """
        self.evaluator_url = evaluator_url

    def register_agent(self, agent_url: str) -> bool:
        """
        Register an agent with the evaluator.

        Args:
            agent_url (str): The URL of the agent server.

        Returns:
            bool: True if the agent was registered successfully, False otherwise.
        """
        try:
            response = requests.post(
                f"{self.evaluator_url}{ENDPOINTS.EVALUATOR_REGISTER_AGENT}", json={"agent_url": agent_url}
            )
            handle_response_errors(response)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            return False

    def get_status(self) -> Optional[dict]:
        """
        Get the current status of the evaluator.

        Returns:
            Optional[dict]: A dictionary containing the evaluator's current state and task, or None if there was an error.
        """
        try:
            response = requests.get(f"{self.evaluator_url}{ENDPOINTS.EVALUATOR_STATUS}")
            handle_response_errors(response)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            return None

    def start_evaluation(self, task: Task) -> bool:
        """
        Start an evaluation with a task.

        Args:
            task (Task): The task configuration.

        Returns:
            bool: True if the evaluation was started successfully, False otherwise.
        """
        try:
            response = requests.post(
                f"{self.evaluator_url}{ENDPOINTS.EVALUATOR_EVALUATION_START}", json=task.model_dump()
            )
            handle_response_errors(response)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            return False

    def agent_finished(self) -> bool:
        """
        Notify the evaluator that the agent has finished its task.

        Returns:
            bool: True if the notification was sent successfully, False otherwise.
        """
        try:
            response = requests.post(f"{self.evaluator_url}{ENDPOINTS.EVALUATOR_AGENT_FINISHED}")
            handle_response_errors(response)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            return False

    def get_results(self) -> Optional[EvaluationResult]:
        """
        Get the results of the evaluation.

        Returns:
            Optional[EvaluationResult]: The evaluation results, or None if not available.
        """
        try:
            response = requests.get(f"{self.evaluator_url}{ENDPOINTS.EVALUATOR_EVALUATION_RESULTS}")
            handle_response_errors(response)

            # Check if the response has content (not 204 No Content)
            if response.status_code == 204 or not response.content:
                return None
            else:
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            return None
