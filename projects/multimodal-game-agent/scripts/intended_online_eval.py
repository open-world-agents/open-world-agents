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

import time
import threading
import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Optional
import uuid
from owa.env.desktop.constants import VK
from typing_extensions import Annotated

import typer
import requests
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Response, status
from pydantic import BaseModel, Field
from rich.logging import RichHandler

from owa.core.registry import CALLABLES, activate_module
from constants import NETWORK, ENDPOINTS, TIMEOUTS, DEFAULTS

# Configure Rich-based logging
logging.basicConfig(
    level=logging.DEBUG, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

# --- Common Components --- #


class AgentState(Enum):
    """States for the Agent state machine"""

    INIT = auto()
    READY = auto()
    RUNNING = auto()
    STOPPING = auto()


class EvaluatorState(Enum):
    """States for the Evaluator state machine"""

    INIT = auto()
    READY = auto()
    EVALUATING = auto()
    SCORING = auto()


class Task(BaseModel):
    """Configuration for a task in an environment"""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    env_name: str
    window_name: str
    task_description: str
    timeout: int
    success_criteria: dict[str, Any]


class EvaluationResult(BaseModel):
    """Results from an evaluation"""

    task_id: str
    metrics: dict[str, Any]
    notes: Optional[str] = None


def run_server_background(
    run_method,
    host: str = NETWORK.DEFAULT_HOST,
    port: int = NETWORK.AGENT_PORT,
    healthcheck_endpoint: str = ENDPOINTS.AGENT_STATUS,
    *args,
    **kwargs,
) -> Optional[str]:
    """
    Run a server in a background thread and wait for it to be ready.

    Args:
        run_method: The method to call to run the server.
        host: The host to run the server on.
        port: The port to run the server on.
        healthcheck_endpoint: The healthcheck endpoint to check for readiness.
        *args, **kwargs: Additional arguments to pass to the run method.

    Returns:
        Optional[str]: The URL of the server if it started successfully, None otherwise.
    """
    # Start the server in a background thread
    logger.info(f"Starting server in background on port {port}...")
    server_thread = threading.Thread(target=run_method, args=(host, port, *args), kwargs=kwargs, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    server_url = f"http://{host if host != NETWORK.DEFAULT_HOST else NETWORK.LOCALHOST}:{port}"
    max_retries = TIMEOUTS.SERVER_STARTUP_MAX_RETRIES
    for i in range(max_retries):
        try:
            response = requests.get(f"{server_url}{healthcheck_endpoint}")
            if response.status_code == 200:
                logger.debug(f"Server healthy after {i + 1} attempts")
                return server_url
        except requests.exceptions.RequestException:
            pass

        if i == max_retries - 1:
            logger.error("Failed to connect to server")
            return None

        time.sleep(TIMEOUTS.SERVER_STARTUP_RETRY_INTERVAL)


def handle_response_errors(response: Response, raise_error: bool = False):
    """Helper function to handle HTTP errors from responses"""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP {response.status_code}: {response.text}"
        logger.warning(error_msg)
        if raise_error:
            raise requests.exceptions.HTTPError(error_msg) from e


# --- Agent Components --- #


class Agent(ABC):
    """
    Abstract base class for implementing a agent.

    The child class must implement the _play_env method.
    """

    def __init__(self):
        """
        Initialize the agent.
        """
        self._state = AgentState.INIT
        self.current_task = None
        self.task_thread = None
        self.stop_event = threading.Event()
        self.api_server = None
        self.agent_url = None
        self.evaluator_url = None

    @property
    def state(self) -> AgentState:
        """Getter for the agent state."""
        return self._state

    @state.setter
    def state(self, new_state: AgentState):
        """Setter for the agent state with debug logging."""
        old_state = self._state
        self._state = new_state
        logger.debug(f"Agent state changed: {old_state.name} -> {new_state.name}")

    def _run_task(self, task: Task):
        """
        Run the task. Meant to be called in a background thread.

        Args:
            task (Task): The task configuration.
        """
        try:
            self.stop_event.clear()

            start_time = time.time()
            # Start the environment loop that handles the stop_event
            while not self.stop_event.is_set():
                # call user-implemented method _play_env()
                continue_task = self._play_env(task)

                # If _play_env returns False, it means the task is finished
                if not continue_task:
                    logger.debug(f"`_run_task()`: Agent task finished in {time.time() - start_time} seconds")
                    self._task_finished()
                    return

            # stop event has been set
            logger.debug(f"`_run_task()`: Agent task stopped in {time.time() - start_time} seconds")
            # TODO: signal to evaluator that task has been stopped. might be stop_event.clear(), or AgentState.READY

        except Exception as e:
            logger.error(f"Error in task execution: {e}")
        finally:
            self.state = AgentState.READY

    def _task_finished(self) -> bool:
        """
        Signal that the current task is finished. This method should be called by the agent
        implementation when it determines that a task has been successfully completed.
        """
        if self.state != AgentState.RUNNING:
            logger.error("Cannot finish task: no task is currently running")
            return False

        # Notify evaluator that we're done
        try:
            response = requests.post(f"{self.evaluator_url}{ENDPOINTS.EVALUATOR_AGENT_FINISHED}")
            handle_response_errors(response)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception when notifying evaluator: {e}")

        return True

    @abstractmethod
    def _play_env(self, task: Task) -> bool:
        """
        This method must be implemented by subclasses.
        It should implement the logic for playing the environment for a single step.

        Args:
            task (Task): Configuration for the task to perform

        Returns:
            bool: True if the task should continue, False if the task is complete
        """
        ...

    def run(self, host: str = NETWORK.DEFAULT_HOST, port: int = NETWORK.AGENT_PORT):
        """
        Run the agent server.

        Args:
            host (str): The host to run the server on.
            port (int): The port to run the server on.
        """
        self.api_server = AgentAPIServer(self)
        self.agent_url = f"http://{host}:{port}"
        uvicorn.run(self.api_server.app, host=host, port=port)

    def run_background(self, host: str = NETWORK.DEFAULT_HOST, port: int = NETWORK.AGENT_PORT) -> Optional[str]:
        """
        Run the agent server in a background thread and wait for it to be ready.

        Args:
            host (str): The host to run the server on.
            port (int): The port to run the server on.

        Returns:
            Optional[str]: The URL of the agent server if it started successfully, None otherwise.
        """
        return run_server_background(self.run, host, port, ENDPOINTS.AGENT_STATUS)


# --- API Server Components --- #


class AgentAPIServer:
    """API server for the Agent"""

    def __init__(self, agent: Agent):
        """Initialize the API server with a reference to the agent"""
        self.agent = agent
        self.app = FastAPI(title="Agent API")

        # Register API endpoints
        self.app.get(ENDPOINTS.AGENT_STATUS)(self._get_status)
        self.app.post(ENDPOINTS.AGENT_TASK_START)(self._task_start)
        self.app.post(ENDPOINTS.AGENT_TASK_STOP)(self._task_stop)
        self.app.post(ENDPOINTS.AGENT_KILL)(self._kill)
        self.app.post(ENDPOINTS.AGENT_REGISTER_EVALUATOR)(self._register_evaluator)

    def _get_status(self, response: Response):
        """Return the current state of the agent"""
        response.status_code = status.HTTP_200_OK
        return {
            "state": self.agent.state.name,
            "current_task": self.agent.current_task,
        }

    def _task_start(self, task: Task, background_tasks: BackgroundTasks, response: Response):
        """Start a new task"""
        if self.agent.state != AgentState.READY:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {
                "success": False,
                "error": f"Agent not ready. Current state: {self.agent.state.name}",
            }

        self.agent.current_task = task
        self.agent.state = AgentState.RUNNING
        self.agent.stop_event.clear()

        # Start the task in a background thread
        background_tasks.add_task(self.agent._run_task, task)

        response.status_code = status.HTTP_200_OK
        return {"success": True, "message": f"Started task for environment: {task.env_name}"}

    def _task_stop(self, response: Response):
        """Stop the current task"""
        if self.agent.state != AgentState.RUNNING:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"success": False, "error": "Agent is not currently running"}

        self.agent.state = AgentState.STOPPING
        self.agent.stop_event.set()

        # Wait for cleanup to actually complete
        max_wait_time = TIMEOUTS.TASK_CLEANUP_TIMEOUT
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            # AgentState should be READY after task is stopped
            if self.agent.state == AgentState.READY:
                logger.debug(f"Task cleanup completed successfully in {time.time() - start_time:.2f} seconds")
                response.status_code = status.HTTP_200_OK
                return {"success": True, "message": "Task stopped"}
            time.sleep(0.1)  # Small sleep to avoid busy waiting
        else:
            # If we exit the loop normally (timeout), log an error
            logger.error(f"Task cleanup may not have completed properly after {max_wait_time} seconds")
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"success": False, "error": "Task cleanup may not have completed properly"}

    def _kill(self, response: Response):
        """Force kill the agent process"""
        self.agent.stop_event.set()
        # TODO : implement kill logic. Kill is intended to be called by the evaluator, when the agent is not responding with _task_stop().
        response.status_code = status.HTTP_200_OK
        return {"success": True, "message": "Kill signal received"}

    class RegisterEvaluatorRequest(BaseModel):
        evaluator_url: str

    def _register_evaluator(self, request: RegisterEvaluatorRequest, response: Response):
        """Register an evaluator with the agent"""
        if self.agent.state != AgentState.INIT:
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

            return {
                "success": False,
                "error": f"Agent not in INIT state. Current state: {self.agent.state.name}",
            }

        self.agent.evaluator_url = request.evaluator_url
        self.agent.state = AgentState.READY
        response.status_code = status.HTTP_200_OK
        return {"success": True, "message": f"Registered evaluator at {request.evaluator_url}"}


class AgentAPIClient:
    """API client for interacting with the Agent server"""

    def __init__(self, agent_url: str):
        """
        Initialize the API client with the agent URL.

        Args:
            agent_url (str): The URL of the agent server.
        """
        self.agent_url = agent_url

    def check_ready(self) -> bool:
        """
        Check if the agent is ready to accept tasks.

        Returns:
            bool: True if the agent is ready, False otherwise.
        """
        try:
            response = requests.get(f"{self.agent_url}{ENDPOINTS.AGENT_STATUS}")
            handle_response_errors(response)
            return response.json().get("state") == "READY"
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            return False

    def get_status(self) -> Optional[dict]:
        """
        Get the current status of the agent.

        Returns:
            Optional[dict]: A dictionary containing the agent's current state and task, or None if there was an error.
        """
        try:
            response = requests.get(f"{self.agent_url}{ENDPOINTS.AGENT_STATUS}")
            handle_response_errors(response)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            return None

    def start_task(self, task: Task) -> bool:
        """
        Send a task to the agent.

        Args:
            task (Task): The task configuration.

        Returns:
            bool: True if the task was started successfully, False otherwise.
        """
        try:
            response = requests.post(f"{self.agent_url}{ENDPOINTS.AGENT_TASK_START}", json=task.model_dump())
            handle_response_errors(response)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            return False

    def stop_task(self) -> bool:
        """
        Request the agent to stop the current task.

        Returns:
            bool: True if the task was stopped successfully, False otherwise.
        """
        try:
            response = requests.post(f"{self.agent_url}{ENDPOINTS.AGENT_TASK_STOP}")
            handle_response_errors(response)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            return False

    def reset(self) -> bool:
        """
        Reset the agent state.

        Returns:
            bool: True if the agent was reset successfully, False otherwise.
        """
        try:
            response = requests.post(f"{self.agent_url}{ENDPOINTS.AGENT_RESET}")
            handle_response_errors(response)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            return False

    def kill(self) -> bool:
        """
        Force kill the agent process.

        Returns:
            bool: True if the agent was killed successfully, False otherwise.
        """
        try:
            response = requests.post(f"{self.agent_url}{ENDPOINTS.AGENT_KILL}")
            handle_response_errors(response)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            return False

    def register_evaluator(self, evaluator_url: str) -> bool:
        """
        Register an evaluator with the agent.

        Args:
            evaluator_url (str): The URL of the evaluator server.

        Returns:
            bool: True if the evaluator was registered successfully, False otherwise.
        """
        try:
            response = requests.post(
                f"{self.agent_url}{ENDPOINTS.AGENT_REGISTER_EVALUATOR}", json={"evaluator_url": evaluator_url}
            )
            handle_response_errors(response)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            return False


# --- Evaluator Components --- #


class Evaluator(ABC):
    """
    Abstract base class for evaluators of agents.

    Handles setting up tasks, monitoring agent behavior, and scoring performance.
    The child class must implement the _score_task() and _setup_environment() methods.
    """

    def __init__(self):
        """
        Initialize the evaluator.
        """
        self._state = EvaluatorState.INIT
        self.agent_url = None
        self.agent_api_client: AgentAPIClient = None
        self.task = None
        self.result = None
        self.result_lock = threading.Lock()  # Lock for self.result
        self.task_start_time = None
        self.task_thread = None
        self.stop_event = threading.Event()
        self.api_server = None

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
        self.task = task
        self.task_start_time = time.time()
        self.result = None
        self.state = EvaluatorState.EVALUATING

        # Start monitoring thread
        self.task_thread = threading.Thread(target=self._monitor_task, args=(self.task,), daemon=True)
        self.task_thread.start()

        return True

    def _monitor_task(self, task: Task):
        """
        Monitor a running task and handle timeouts.

        Args:
            task (Task): The task configuration.
        """

        # start a timer thread that will stop the task after a certain time
        def timeout_watcher():
            time.sleep(task.timeout)
            self.stop_event.set()

        timer_thread = threading.Thread(target=timeout_watcher, daemon=True)
        timer_thread.start()

        start_time = time.time()
        while not self.stop_event.is_set():
            # Check if the task has already been marked as finished
            if self.state != EvaluatorState.EVALUATING:
                return
            time.sleep(DEFAULTS.ENV_CHECK_INTERVAL)  # Sleep to avoid busy waiting

        if self.state == EvaluatorState.EVALUATING:
            elapsed = time.time() - start_time
            logger.debug(f"Task timed out after {elapsed} seconds: {task.env_name}")
            # Stop the agent task
            self.agent_api_client.stop_task()
            # Score the task

            self.process_finished_task(note=f"`_monitor_task()`: Task timed out after {elapsed} seconds")

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
                    self.result = self._score_task(self.task, task_elapsed_time, note)
                    logger.debug(f"Task completed. Result: {self.result.model_dump()}")
                    self.state = EvaluatorState.READY
                else:
                    logger.debug("Task already completed. Skipping scoring.")
        else:
            logger.debug("Task already completed. Skipping scoring.")

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
        Setup the environment for a task. Also handles restarting the environment.

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
            "current_task": self.evaluator.task.model_dump() if self.evaluator.task else None,
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


# --- Concrete Implementation Examples --- #


class MyAgent(Agent):
    """Example implementation of an agent"""

    def __init__(self):
        super().__init__()

        # Example: Super Hexagon implementation would use screen observations
        # and generate keyboard inputs based on those observations
        activate_module("owa.env.desktop")
        activate_module("owa.env.gst")

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
    agent = MyAgent()
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
    agent = MyAgent()
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
