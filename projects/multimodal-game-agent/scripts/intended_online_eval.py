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
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Dict, Any, Optional
from typing_extensions import Annotated

import typer
import requests
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from owa.core.registry import CALLABLES, activate_module


# --- Common Components --- #


class AgentState(Enum):
    """States for the Agent state machine"""

    INIT = auto()
    READY = auto()
    RUNNING = auto()
    STOPPING = auto()
    FINISHED = auto()


class EvaluatorState(Enum):
    """States for the Evaluator state machine"""

    INIT = auto()
    WAITING_FOR_AGENT = auto()
    AGENT_READY = auto()
    EVALUATING = auto()
    SCORING = auto()


class TaskConfig(BaseModel):
    """Configuration for a task in an environment"""

    env_name: str
    window_name: str
    task_description: str
    max_duration_seconds: int
    success_criteria: Dict[str, Any]


class EvaluationResult(BaseModel):
    """Results from an evaluation"""

    task_id: str
    score: float
    metrics: Dict[str, Any]
    duration_seconds: float
    success: bool
    notes: Optional[str] = None


# --- Agent Components --- #


class Agent(ABC):
    """
    Abstract base class for implementing a agent.

    Researchers only need to implement the _play_env method to create their own agent.
    This provides a simple interface that handles the communication with the evaluator.
    """

    def __init__(self, model_id: str = None):
        """
        Initialize the agent with optional model.

        Args:
            model_id (str): The model ID.
        """
        self.model_id = model_id
        self.state = AgentState.INIT
        self.current_task = None
        self.task_thread = None
        self.stop_event = threading.Event()
        self.api_server = None

    def _run_task(self, task: TaskConfig):
        """
        Run the task in a background thread.

        Args:
            task (TaskConfig): The task configuration.
        """
        try:
            # Call the user-implemented method
            self._play_env(task, self.stop_event)

            # If we got here without being stopped, the task is finished
            if not self.stop_event.is_set():
                self.state = AgentState.FINISHED
                # Notify evaluator that we're done
                try:
                    requests.post("http://localhost:8001/evaluator/agent_finished")
                except requests.exceptions.RequestException as e:
                    print(f"Request exception: {e}")
        except Exception as e:
            print(f"Error in task execution: {e}")
        finally:
            if self.state == AgentState.RUNNING:
                self.state = AgentState.READY  # Change IDLE to READY to better support multiple evaluations

    def _reset_state(self):
        """
        Reset the agent state to READY after a task is finished.
        """
        self.state = AgentState.READY

    @abstractmethod
    def _play_env(self, task: TaskConfig, stop_event: threading.Event):
        """
        This method must be implemented by subclasses.

        Args:
            task (TaskConfig): Configuration for the task to perform
            stop_event (threading.Event): Event that will be set when the task should stop
        """
        pass

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Run the agent server.

        Args:
            host (str): The host to run the server on.
            port (int): The port to run the server on.
        """
        self.state = AgentState.READY
        self.api_server = AgentAPIServer(self)
        uvicorn.run(self.api_server.app, host=host, port=port)


# --- API Server Components --- #


class AgentAPIServer:
    """API server for the Agent"""

    def __init__(self, agent: Agent):
        """Initialize the API server with a reference to the agent"""
        self.agent = agent
        self.app = FastAPI(title="Agent API")

        # Register API endpoints
        self.app.get("/agent/status")(self._get_status)
        self.app.post("/agent/task/start")(self._start_task)
        self.app.post("/agent/task/stop")(self._stop_task)
        self.app.post("/agent/task/finished")(self._finished_task)
        self.app.post("/agent/kill")(self._kill)
        self.app.post("/agent/reset")(self._reset_agent)

    def _get_status(self):
        """Return the current state of the agent"""
        return {"state": self.agent.state.name}

    def _start_task(self, task: TaskConfig, background_tasks: BackgroundTasks):
        """Start a new task"""
        if self.agent.state != AgentState.READY:
            return {
                "success": False,
                "error": f"Agent not ready. Current state: {self.agent.state.name}",
            }

        self.agent.current_task = task
        self.agent.state = AgentState.RUNNING
        self.agent.stop_event.clear()

        # Start the task in a background thread
        background_tasks.add_task(self.agent._run_task, task)

        return {"success": True, "message": f"Started task for environment: {task.env_name}"}

    def _stop_task(self):
        """Stop the current task"""
        if self.agent.state != AgentState.RUNNING:
            return {"success": False, "error": "No task is currently running"}

        self.agent.state = AgentState.STOPPING
        self.agent.stop_event.set()

        # Wait for a bit to allow for cleanup
        time.sleep(0.5)
        self.agent.state = AgentState.READY

        return {"success": True, "message": "Task stopped"}

    def _finished_task(self):
        """Signal that the current task is finished"""
        if self.agent.state != AgentState.RUNNING:
            return {"success": False, "error": "No task is currently running"}

        self.agent.state = AgentState.FINISHED
        # Reset to READY state after a short delay to allow for cleanup
        threading.Timer(1.0, self.agent._reset_state).start()
        return {"success": True, "message": "Task marked as finished"}

    def _reset_agent(self):
        """Reset the agent state for a new evaluation run"""
        if self.agent.state == AgentState.RUNNING:
            self.agent.stop_event.set()
            time.sleep(0.5)

        self.agent.state = AgentState.READY
        self.agent.current_task = None
        self.agent.stop_event.clear()
        return {"success": True, "message": "Agent reset and ready for new tasks"}

    def _kill(self):
        """Force kill the agent process"""
        self.agent.stop_event.set()
        # In a real implementation, this would shut down the server
        return {"success": True, "message": "Kill signal received"}


class AgentAPIClient:
    """API client for interacting with the Agent server"""

    def __init__(self, agent_url: str = "http://localhost:8000"):
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
            response = requests.get(f"{self.agent_url}/agent/status")
            response.raise_for_status()
            return response.json().get("state") == "READY"
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False

    def start_task(self, task_config: TaskConfig) -> bool:
        """
        Send a task to the agent.

        Args:
            task_config (TaskConfig): The task configuration.

        Returns:
            bool: True if the task was started successfully, False otherwise.
        """
        try:
            response = requests.post(f"{self.agent_url}/agent/task/start", json=task_config.model_dump())
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False

    def stop_task(self) -> bool:
        """
        Request the agent to stop the current task.

        Returns:
            bool: True if the task was stopped successfully, False otherwise.
        """
        try:
            response = requests.post(f"{self.agent_url}/agent/task/stop")
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False

    def reset(self) -> bool:
        """
        Reset the agent state.

        Returns:
            bool: True if the agent was reset successfully, False otherwise.
        """
        try:
            response = requests.post(f"{self.agent_url}/agent/reset")
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False

    def kill(self) -> bool:
        """
        Force kill the agent process.

        Returns:
            bool: True if the agent was killed successfully, False otherwise.
        """
        try:
            response = requests.post(f"{self.agent_url}/agent/kill")
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False


# --- Evaluator Components --- #


class Evaluator(ABC):
    """
    Abstract base class for evaluators of agents.

    Handles setting up tasks, monitoring agent behavior, and scoring performance.
    """

    def __init__(self):
        """
        Initialize the evaluator.
        """
        self.state = EvaluatorState.INIT
        self.agent_url = None
        self.agent_api_client = None
        self.task = None
        self.result = None
        self.result_lock = threading.Lock()  # Lock for self.result
        self.task_start_time = None
        self.task_thread = None
        self.stop_event = threading.Event()
        self.api_server = None

    def _run_evaluation(self):
        """
        Run the evaluation process.
        """
        self.stop_event.clear()

        # Check if agent is ready
        if not self.agent_api_client.check_ready():
            print("Agent not ready. Aborting evaluation.")
            self.state = EvaluatorState.WAITING_FOR_AGENT
            return

        self.state = EvaluatorState.AGENT_READY

        # Start the task
        self._start_task()

    def _start_task(self):
        """
        Start the task.
        """
        if not self.task:
            print("No task to run.")
            self.state = EvaluatorState.WAITING_FOR_AGENT
            return

        # Setup environment for the task
        try:
            self._setup_environment(self.task)
        except Exception as e:
            print(f"Error setting up environment: {e}")
            self.state = EvaluatorState.WAITING_FOR_AGENT
            return

        # Start the agent on this task
        if not self.agent_api_client.start_task(self.task):
            print(f"Failed to start task on agent: {self.task.env_name}")
            self.state = EvaluatorState.WAITING_FOR_AGENT
            return

        self.task_start_time = time.time()
        self.state = EvaluatorState.EVALUATING

        # Start monitoring thread
        self.task_thread = threading.Thread(target=self._monitor_task, args=(self.task,))
        self.task_thread.daemon = True
        self.task_thread.start()

    def _monitor_task(self, task: TaskConfig):
        """
        Monitor a running task and handle timeouts.

        Args:
            task (TaskConfig): The task configuration.
        """
        start_time = time.time()
        max_duration = task.max_duration_seconds

        while time.time() - start_time < max_duration and not self.stop_event.is_set():
            # Check if the task has already been marked as finished by the agent
            if self.state != EvaluatorState.EVALUATING:
                return
            time.sleep(0.1)  # Sleep to avoid busy waiting

        # If we get here, either the timeout occurred or stop was requested
        if not self.stop_event.is_set() and self.state == EvaluatorState.EVALUATING:
            print(f"Task timed out after {max_duration} seconds: {task.env_name}")
            # Stop the agent task
            self.agent_api_client.stop_task()
            # Score the task
            self.state = EvaluatorState.SCORING
            self.process_finished_task(note=f"_monitor_task: Task timed out after {max_duration} seconds")

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
        print(f"_process_finished_task: {note}")
        if not self.task or not self.task_start_time:
            self.state = EvaluatorState.WAITING_FOR_AGENT
            return

        # First check without lock
        if self.result is None:
            # Acquire lock and check again (double-checked locking)
            with self.result_lock:
                if self.result is None:
                    duration = time.time() - self.task_start_time
                    self.result = self._score_task(self.task, duration, note)
                    print(f"Task completed. Result: {self.result.model_dump()}")
                else:
                    print("Task already completed. Skipping scoring.")
        else:
            print("Task already completed. Skipping scoring.")

        self.state = EvaluatorState.WAITING_FOR_AGENT

    @abstractmethod
    def _score_task(self, task: TaskConfig, duration: float, note: str = "") -> EvaluationResult:
        """
        Calculate score for a completed task.

        Must be implemented by subclasses.

        Args:
            task (TaskConfig): The task configuration that was completed
            duration (float): How long the task took to complete

        Returns:
            EvaluationResult: The scoring results
        """
        pass

    @abstractmethod
    def _setup_environment(self, task: TaskConfig):
        """
        Setup the environment for a task.

        Must be implemented by subclasses.

        Args:
            task (TaskConfig): Configuration for the task to setup
        """
        pass

    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """
        Run the evaluator server.

        Args:
            host (str): The host to run the server on.
            port (int): The port to run the server on.
        """
        self.api_server = EvaluatorAPIServer(self)
        uvicorn.run(self.api_server.app, host=host, port=port)


class EvaluatorAPIServer:
    """API server for the Evaluator"""

    def __init__(self, evaluator: Evaluator):
        """Initialize the API server with a reference to the evaluator"""
        self.evaluator = evaluator
        self.app = FastAPI(title="Evaluator API")

        # Register API endpoints
        self.app.post("/evaluator/register_agent")(self._register_agent)
        self.app.post("/evaluator/evaluation/start")(self._start_evaluation)
        self.app.post("/evaluator/agent_finished")(self._agent_finished)
        self.app.post("/evaluator/reset")(self._reset_evaluator)
        self.app.get("/evaluator/status")(self._get_status)
        self.app.get("/evaluator/evaluation/results")(self._get_results)

    def _register_agent(self, data: Dict[str, str]):
        """Register an agent with the evaluator"""
        if self.evaluator.state != EvaluatorState.INIT:
            return {
                "success": False,
                "error": f"Evaluator not idle. Current state: {self.evaluator.state.name}",
            }

        self.evaluator.agent_url = data.get("agent_url")
        if not self.evaluator.agent_url:
            return {"success": False, "error": "No agent URL provided"}

        self.evaluator.agent_api_client = AgentAPIClient(self.evaluator.agent_url)
        self.evaluator.state = EvaluatorState.WAITING_FOR_AGENT
        return {"success": True, "message": f"Registered agent at {self.evaluator.agent_url}"}

    def _start_evaluation(self, data: Dict[str, Any], background_tasks: BackgroundTasks):
        """Start an evaluation with a task"""
        if self.evaluator.state != EvaluatorState.WAITING_FOR_AGENT:
            return {
                "success": False,
                "error": f"Evaluator not waiting for agent. Current state: {self.evaluator.state.name}",
            }

        if not self.evaluator.agent_api_client:
            return {"success": False, "error": "No agent registered"}

        # Extract and validate the task
        task_data = data.get("task")
        if not task_data:
            return {"success": False, "error": "No task provided"}

        self.evaluator.task = TaskConfig(**task_data)

        # Start the evaluation in a background thread
        background_tasks.add_task(self.evaluator._run_evaluation)
        return {"success": True, "message": "Started evaluation"}

    def _agent_finished(self):
        """Called when the agent finishes a task"""
        if self.evaluator.state == EvaluatorState.EVALUATING:
            self.evaluator.state = EvaluatorState.SCORING
            self.evaluator.process_finished_task(note="Agent reported task completion")
            return {"success": True, "message": "Task completion acknowledged"}
        return {
            "success": False,
            "error": f"Unexpected state: {self.evaluator.state.name}. Expected: EVALUATING",
        }

    def _reset_evaluator(self):
        """Reset the evaluator for a new evaluation run"""
        self.evaluator.stop_event.set()
        if self.evaluator.task_thread and self.evaluator.task_thread.is_alive():
            self.evaluator.task_thread.join(timeout=2)  # Wait for thread to finish

        if self.evaluator.state == EvaluatorState.INIT:
            raise ValueError("Evaluator is not initiated. Call register_agent first")
        else:
            self.evaluator.state = EvaluatorState.WAITING_FOR_AGENT
        self.evaluator.task = None
        self.evaluator.result = None
        self.evaluator.task_start_time = None
        # reset agent_api_client
        self.evaluator.agent_api_client.reset()
        self.evaluator.stop_event.clear()
        return {"success": True, "message": "Evaluator reset"}

    def _get_status(self):
        """Get the current status of the evaluation"""
        return {
            "state": self.evaluator.state.name,
            "agent_url": self.evaluator.agent_url,
            "current_task": self.evaluator.task.model_dump() if self.evaluator.task else None,
        }

    def _get_results(self):
        """Get the results of the evaluation"""
        if self.evaluator.result:
            return self.evaluator.result.model_dump()
        return {"message": "No results available"}


class EvaluatorAPIClient:
    """API client for interacting with the Evaluator server"""

    def __init__(self, evaluator_url: str = "http://localhost:8001"):
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
            response = requests.post(f"{self.evaluator_url}/evaluator/register_agent", json={"agent_url": agent_url})
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False

    def start_evaluation(self, task: TaskConfig) -> bool:
        """
        Start an evaluation with a task.

        Args:
            task (TaskConfig): The task configuration.

        Returns:
            bool: True if the evaluation was started successfully, False otherwise.
        """
        try:
            response = requests.post(
                f"{self.evaluator_url}/evaluator/evaluation/start", json={"task": task.model_dump()}
            )
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False

    def reset(self) -> bool:
        """
        Reset the evaluator.

        Returns:
            bool: True if the evaluator was reset successfully, False otherwise.
        """
        try:
            response = requests.post(f"{self.evaluator_url}/evaluator/reset")
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False

    def get_results(self) -> Optional[EvaluationResult]:
        """
        Get the results of the evaluation.

        Returns:
            Optional[EvaluationResult]: The evaluation results, or None if not available.
        """
        try:
            response = requests.get(f"{self.evaluator_url}/evaluator/evaluation/results")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return None


# --- Concrete Implementation Examples --- #


class MyAgent(Agent):
    """Example implementation of an agent"""

    def _play_env(self, task: TaskConfig, stop_event: threading.Event):
        """
        Implement the environment playing logic here.
        This is what researchers would customize with their models and logic.
        """
        print(f"Playing environment: {task.env_name}")
        print(f"Task description: {task.task_description}")

        # Example: Super Hexagon implementation would use screen observations
        # and generate keyboard inputs based on those observations
        activate_module("owa.env.desktop")
        activate_module("owa.env.gst")

        # Setup screen recorder and other components
        # (Simplified for example)

        # Environment loop
        start_time = time.time()
        while not stop_event.is_set() and time.time() - start_time < task.max_duration_seconds:
            window_active = CALLABLES["window.is_active"](task.window_name)
            if not window_active:
                print(f"Window {task.window_name} is not active", end="\r")
                time.sleep(0.1)
                continue

            # Example: Get screen state and make decisions
            # This would use your ML model to generate actions

            # Simulate thinking and acting
            time.sleep(0.1)

            # Example keyboard input (pressing right arrow)
            if not stop_event.is_set():
                CALLABLES["keyboard.press"](39)  # Right arrow
                time.sleep(0.05)
                CALLABLES["keyboard.release"](39)
                print(f"key {39} pressed", end="\r")

            # Check for environment-specific success condition
            if self._check_success_condition(task):
                # Signal that we're done
                try:
                    requests.post("http://localhost:8000/agent/task/finished")
                except requests.exceptions.RequestException as e:
                    print(f"Request exception: {e}")
                break

        print()
        print("Finished playing environment")

    def _check_success_condition(self, task: TaskConfig) -> bool:
        """
        Check if the success condition for the task has been met.

        Args:
            task (TaskConfig): The task configuration.

        Returns:
            bool: True if the success condition has been met, False otherwise.
        """
        # This would implement environment-specific success detection
        # For example, detecting a "victory" screen or a specific score
        # task.success_criteria
        return False


class SimpleEvaluator(Evaluator):
    """Concrete implementation of an Evaluator"""

    def _score_task(self, task: TaskConfig, duration: float, note: str = "") -> EvaluationResult:
        """
        Simple scoring based on task duration and success criteria.

        Args:
            task (TaskConfig): The task configuration that was completed
            duration (float): How long the task took to complete

        Returns:
            EvaluationResult: The scoring results
        """
        # In a real implementation, this would be more sophisticated
        # and would actually verify the agent's success against ground truth
        # For this example, we use a simple time-based metric

        # Calculate a score: faster is better, up to a maximum of 1.0
        # Score decreases linearly from 1.0 to 0.1 as time approaches max_duration
        max_score = 1.0
        min_score = 0.1
        max_time = task.max_duration_seconds

        # Faster completion gets higher score
        score = max(min_score, max_score - (duration / max_time) * (max_score - min_score))

        # Here you would check if the agent actually achieved the success criteria
        # This could involve checking game state, screenshot analysis, etc.
        # For this example, we'll just assume success if they finished before timeout
        success = duration < task.max_duration_seconds

        # In a real evaluator, you might capture screenshots or other artifacts
        # CALLABLES["screen.capture"]()

        return EvaluationResult(
            task_id="task",
            score=score,
            metrics={"time": duration},
            duration_seconds=duration,
            success=success,
            notes=note,
        )

    def _setup_environment(self, task: TaskConfig):  # NOTE: make a separate ENV class?
        """
        Setup the environment for a task.

        Args:
            task (TaskConfig): Configuration for the task to setup
        """
        print(f"Setting up environment for {task.env_name}")
        # In a real implementation, this would launch games, configure windows, etc.


# --- Example Tasks --- #


def get_example_tasks() -> List[TaskConfig]:
    """
    Get a list of example tasks for testing.

    Returns:
        List[TaskConfig]: A list of example tasks.
    """
    return [
        TaskConfig(
            env_name="Super Hexagon",
            window_name="Super Hexagon",
            task_description="Survive for at least 5 seconds in the hardest level",
            max_duration_seconds=5,
            success_criteria={"time_survived": 1},
        ),
        # TaskConfig(
        #     env_name="ZType",
        #     window_name="ZType",
        #     task_description="Complete the first level with at least 90% accuracy",
        #     max_duration_seconds=120,
        #     success_criteria={"accuracy": 0.9, "level_completed": True},
        # ),
    ]


# --- Run Functions --- #


def run_agent(model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
    """
    Run the agent server.

    Args:
        model_id (str): The model ID to use for the agent.
    """
    agent = MyAgent(model_id=model_id)
    print(f"Starting agent server with model: {model_id}")
    agent.run(host="0.0.0.0", port=8000)


def run_agent_background(model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", port: int = 8000):
    """
    Run the agent server in a background thread and wait for it to be ready

    Args:
        model_id (str): The model ID to use for the agent
        port (int): The port to run the agent server on

    Returns:
        The URL of the agent server if it started successfully, None otherwise
    """
    import threading
    import time
    import requests

    # Start the agent in a background thread
    print("Starting agent server in background...")
    agent_thread = threading.Thread(target=run_agent, args=(model_id,), daemon=True)
    agent_thread.start()

    # Wait for agent server to be ready
    agent_url = f"http://localhost:{port}"
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{agent_url}/agent/status")
            if response.status_code == 200:
                print(f"Agent server is ready after {i + 1} attempts")
                return agent_url
        except requests.exceptions.RequestException:
            pass

        if i == max_retries - 1:
            print("Failed to connect to agent server")
            return None

        time.sleep(1)


def run_evaluator():
    """
    Run the evaluator server.
    """
    evaluator = SimpleEvaluator()
    print("Starting evaluator server")
    evaluator.run(host="0.0.0.0", port=8001)


def run_evaluator_background(port: int = 8001):
    """
    Run the evaluator server in a background thread and wait for it to be ready

    Args:
        port (int): The port to run the evaluator server on

    Returns:
        The URL of the evaluator server if it started successfully, None otherwise
    """
    import threading
    import time
    import requests

    # Start the evaluator in a background thread
    print("Starting evaluator server in background...")
    evaluator_thread = threading.Thread(target=run_evaluator, daemon=True)
    evaluator_thread.start()

    # Wait for evaluator server to be ready
    evaluator_url = f"http://localhost:{port}"
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{evaluator_url}/evaluator/status")
            if response.status_code == 200:
                print(f"Evaluator server is ready after {i + 1} attempts")
                return evaluator_url
        except requests.exceptions.RequestException:
            pass

        if i == max_retries - 1:
            print("Failed to connect to evaluator server")
            return None

        time.sleep(1)


def run_evaluation_client(agent_url: str, evaluator_url: str):
    """
    Run an example evaluation.

    Args:
        agent_url (str): The URL of the agent server.
        evaluator_url (str): The URL of the evaluator server.
    """
    # Create API clients
    evaluator_client = EvaluatorAPIClient(evaluator_url)

    # Register agent with evaluator
    print("Registering agent with evaluator...")
    if not evaluator_client.register_agent(agent_url):
        print("Failed to register agent. Exiting.")
        return

    # Get example task
    example_task = get_example_tasks()[0]  # Get first task

    # Start evaluation
    for _ in range(3):
        evaluator_client.reset()
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
                if results and "message" not in results:
                    print(f"Evaluation completed with results: {results}")
                    break
                time.sleep(2)
            except KeyboardInterrupt:
                print("Interrupted by user. Exiting.")
                break


def run_evaluation_client_with_server(
    model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
):
    """
    Run an example evaluation without blocking

    This function starts both the agent and evaluator servers in background threads,
    waits for them to be ready, and then runs an example evaluation against them.
    """
    # Start agent server in background
    print("Starting agent server...")
    agent_url = run_agent_background(model_id)
    if not agent_url:
        print("Failed to start agent server. Exiting.")
        return

    # Start evaluator server in background
    print("Starting evaluator server...")
    evaluator_url = run_evaluator_background()
    if not evaluator_url:
        print("Failed to start evaluator server. Exiting.")
        return

    # Short delay to ensure servers are ready
    time.sleep(1)

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
    model_id: Annotated[
        str,
        typer.Option(help="Model ID to use for the agent"),
    ] = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
):
    """
    Main entry point.

    Args:
        mode (Mode): The mode to run in.
        model_id (str): The model ID to use for the agent.
    """
    if mode == Mode.AGENT:
        run_agent(model_id)
    elif mode == Mode.EVALUATOR:
        run_evaluator()
    elif mode == Mode.RUN_CLIENT:
        run_evaluation_client("http://localhost:8000", "http://localhost:8001")
    elif mode == Mode.RUN_CLIENT_WITH_SERVER:
        run_evaluation_client_with_server(model_id)
    else:
        raise ValueError(f"Unknown mode: {mode=}")


if __name__ == "__main__":
    typer.run(main)
