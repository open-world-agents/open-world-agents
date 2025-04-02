import logging
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
import traceback
from typing import Optional

import requests
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Response, status
from pydantic import BaseModel
from rich.logging import RichHandler

from owa_game_agent.commons import Task, handle_response_errors, run_server_background
from owa_game_agent.constants import ENDPOINTS, NETWORK, TIMEOUTS

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """States for the Agent state machine"""

    INIT = auto()
    READY = auto()
    RUNNING = auto()
    STOPPING = auto()


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
        from owa_game_agent.evaluator import EvaluatorAPIClient

        self.evaluator_api_client: Optional[EvaluatorAPIClient] = None

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

        except Exception as e:
            logger.error(f"Error in task execution: {e} \n {traceback.format_exc()}")
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

        if self.evaluator_api_client:
            success = self.evaluator_api_client.agent_finished()
            if not success:
                logger.error("Failed to notify evaluator that agent task is finished")
        else:
            logger.warning("Cannot notify evaluator: evaluator_api_client is not initialized")

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
        # TODO : implement kill logic. Kill is intended to be called by the evaluator, when the agent is not responding with _task_stop().
        # To kill the agent, we should make _run_task() run as a process, because threads cannot be killed.
        response.status_code = status.HTTP_501_NOT_IMPLEMENTED
        return {"success": False, "message": "Kill is not implemented"}

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
        from owa_game_agent.evaluator import EvaluatorAPIClient

        self.agent.evaluator_api_client = EvaluatorAPIClient(request.evaluator_url)

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
