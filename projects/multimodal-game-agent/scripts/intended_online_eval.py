"""
This script demonstrates the intended usage and interaction between Agent and Evaluator components
for online evaluation of game agents. Both components run independently and communicate via HTTP,
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

    IDLE = auto()
    READY = auto()
    RUNNING = auto()
    STOPPING = auto()
    FINISHED = auto()


class EvaluatorState(Enum):
    """States for the Evaluator state machine"""

    IDLE = auto()
    WAITING_FOR_AGENT = auto()
    AGENT_READY = auto()
    EVALUATING = auto()
    SCORING = auto()


class TaskConfig(BaseModel):
    """Configuration for a game task"""

    game_name: str
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


class AgentAPI:
    """API client for interacting with the Agent server"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def check_ready(self) -> bool:
        """Check if the agent is ready to accept tasks"""
        try:
            response = requests.get(f"{self.base_url}/status")
            response.raise_for_status()
            return response.json().get("state") == "READY"
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False

    def start_task(self, task_config: TaskConfig) -> bool:
        """Send a task to the agent"""
        try:
            response = requests.post(
                f"{self.base_url}/task/start", json=task_config.model_dump()
            )
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False

    def stop_task(self) -> bool:
        """Request the agent to stop the current task"""
        try:
            response = requests.post(f"{self.base_url}/task/stop")
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False

    def reset(self) -> bool:
        """Reset the agent state"""
        try:
            response = requests.post(f"{self.base_url}/reset")
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False

    def kill(self) -> bool:
        """Force kill the agent process"""
        try:
            response = requests.post(f"{self.base_url}/kill")
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False


class GameAgent(ABC):
    """
    Abstract base class for implementing a game agent.

    Researchers only need to implement the play_game method to create their own agent.
    This provides a simple interface that handles the communication with the evaluator.
    """

    def __init__(self, model_id: str = None):
        """Initialize the agent with optional model"""
        self.model_id = model_id
        self.app = FastAPI(title="Game Agent API")
        self.state = AgentState.IDLE
        self.current_task = None
        self.task_thread = None
        self.stop_event = threading.Event()

        # Register API endpoints
        self.app.get("/status")(self.get_status)
        self.app.post("/task/start")(self.start_task)
        self.app.post("/task/stop")(self.stop_task)
        self.app.post("/task/finished")(self.finished_task)
        self.app.post("/kill")(self.kill)
        self.app.post("/reset")(self.reset)  # Add reset endpoint

    def get_status(self):
        """Return the current state of the agent"""
        return {"state": self.state.name}

    def start_task(self, task: TaskConfig, background_tasks: BackgroundTasks):
        """Start a new task"""
        if self.state != AgentState.READY:
            return {
                "success": False,
                "error": f"Agent not ready. Current state: {self.state.name}",
            }

        self.current_task = task
        self.state = AgentState.RUNNING
        self.stop_event.clear()

        # Start the task in a background thread
        background_tasks.add_task(self._run_task, task)

        return {"success": True, "message": f"Started task for game: {task.game_name}"}

    def _run_task(self, task: TaskConfig):
        """Run the task in a background thread"""
        try:
            # Call the user-implemented method
            self.play_game(task, self.stop_event)

            # If we got here without being stopped, the task is finished
            if not self.stop_event.is_set():
                self.state = AgentState.FINISHED
                # Notify evaluator that we're done
                try:
                    requests.post("http://localhost:8001/agent/finished")
                except requests.exceptions.RequestException as e:
                    print(f"Request exception: {e}")
        except Exception as e:
            print(f"Error in task execution: {e}")
        finally:
            if self.state == AgentState.RUNNING:
                self.state = (
                    AgentState.READY
                )  # Change IDLE to READY to better support multiple evaluations

    def stop_task(self):
        """Stop the current task"""
        if self.state != AgentState.RUNNING:
            return {"success": False, "error": "No task is currently running"}

        self.state = AgentState.STOPPING
        self.stop_event.set()

        # Wait for a bit to allow for cleanup
        time.sleep(0.5)
        self.state = AgentState.READY

        return {"success": True, "message": "Task stopped"}

    def finished_task(self):
        """Signal that the current task is finished"""
        if self.state != AgentState.RUNNING:
            return {"success": False, "error": "No task is currently running"}

        self.state = AgentState.FINISHED
        # Reset to READY state after a short delay to allow for cleanup
        threading.Timer(1.0, self.reset_state).start()
        return {"success": True, "message": "Task marked as finished"}

    def reset_state(self):
        """Reset the agent to READY state for the next task"""
        self.state = AgentState.READY
        self.current_task = None
        self.stop_event.clear()

    def reset(self):
        """Reset the agent state for a new evaluation run"""
        if self.state == AgentState.RUNNING:
            self.stop_event.set()
            time.sleep(0.5)

        self.reset_state()
        return {"success": True, "message": "Agent reset and ready for new tasks"}

    def kill(self):
        """Force kill the agent process"""
        self.stop_event.set()
        # In a real implementation, this would shut down the server
        return {"success": True, "message": "Kill signal received"}

    @abstractmethod
    def play_game(self, task: TaskConfig, stop_event: threading.Event):
        """
        This method must be implemented by subclasses.

        Args:
            task: Configuration for the task to perform
            stop_event: Event that will be set when the task should stop
        """
        pass

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the agent server"""
        self.state = AgentState.READY
        uvicorn.run(self.app, host=host, port=port)


# --- Evaluator Components --- #


class EvaluatorAPI:
    """API client for interacting with the Evaluator server"""

    def __init__(self, evaluator_url: str = "http://localhost:8001"):
        self.evaluator_url = evaluator_url

    def register_agent(self, agent_url: str) -> bool:
        """Register an agent with the evaluator"""
        try:
            response = requests.post(
                f"{self.evaluator_url}/register_agent", json={"agent_url": agent_url}
            )
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False

    def start_evaluation(self, tasks: List[TaskConfig]) -> bool:
        """Start an evaluation with a list of tasks"""
        try:
            # First reset the evaluator
            self.reset()

            response = requests.post(
                f"{self.evaluator_url}/evaluation/start",
                json={"tasks": [t.model_dump() for t in tasks]},
            )
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False

    def reset(self) -> bool:
        """Reset the evaluator"""
        try:
            response = requests.post(f"{self.evaluator_url}/reset")
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return False

    def get_results(self) -> List[EvaluationResult]:
        """Get the results of the evaluation"""
        try:
            response = requests.get(f"{self.evaluator_url}/evaluation/results")
            response.raise_for_status()
            return [EvaluationResult(**r) for r in response.json()["results"]]
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return []


class Evaluator(ABC):
    """
    Abstract base class for evaluators of game agents.

    Handles setting up tasks, monitoring agent behavior, and scoring performance.
    """

    def __init__(self):
        """Initialize the evaluator"""
        self.app = FastAPI(title="Game Evaluator API")
        self.state = EvaluatorState.IDLE
        self.agent_url = None
        self.agent_api = None
        self.tasks = []
        self.current_task_index = 0
        self.current_task = None
        self.results = []
        self.task_start_time = None
        self.task_thread = None
        self.stop_event = threading.Event()

        # Register API endpoints
        self.app.post("/register_agent")(self.register_agent)
        self.app.post("/evaluation/start")(self.start_evaluation)
        self.app.post("/agent/finished")(self.agent_finished)
        self.app.post("/reset")(self.reset_evaluator)  # Add reset endpoint
        self.app.get("/status")(self.get_status)  # Add status endpoint
        self.app.get("/evaluation/results")(self.get_results)

    def register_agent(self, data: Dict[str, str]):
        """Register an agent with the evaluator"""
        self.agent_url = data["agent_url"]
        self.agent_api = AgentAPI(base_url=self.agent_url)
        self.state = EvaluatorState.IDLE
        return {"success": True, "message": f"Agent registered: {self.agent_url}"}

    def start_evaluation(
        self, data: Dict[str, List[Dict[str, Any]]], background_tasks: BackgroundTasks
    ):
        """Start an evaluation with a list of tasks"""
        if not self.agent_api:
            return {"success": False, "error": "No agent registered"}

        # Reset agent before starting a new evaluation
        try:
            requests.post(f"{self.agent_url}/reset")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not reset agent: {e}")

        self.tasks = [TaskConfig(**task) for task in data["tasks"]]
        self.current_task_index = 0
        self.results = []
        self.state = EvaluatorState.IDLE  # Reset state before starting

        # Start evaluation in background
        background_tasks.add_task(self._run_evaluation)

        return {
            "success": True,
            "message": f"Starting evaluation with {len(self.tasks)} tasks",
        }

    def reset_evaluator(self):
        """Reset the evaluator for a new evaluation run"""
        if self.agent_api and self.state != EvaluatorState.IDLE:
            # Try to stop any running tasks
            self.agent_api.stop_task()
            self.stop_event.set()

        self.state = EvaluatorState.IDLE
        self.current_task_index = 0
        self.current_task = None
        self.results = []
        self.task_start_time = None
        self.stop_event.clear()

        return {"success": True, "message": "Evaluator reset"}

    def get_status(self):
        """Get the current status of the evaluation"""
        status = {
            "state": self.state.name,
            "current_task_index": self.current_task_index if self.tasks else None,
            "total_tasks": len(self.tasks),
            "agent_url": self.agent_url,
        }
        return status

    def get_results(self):
        """Get the results of the evaluation"""
        return {"results": [r.model_dump() for r in self.results]}

    def agent_finished(self):
        """Called when the agent finishes a task"""
        if self.state == EvaluatorState.EVALUATING:
            self.state = EvaluatorState.SCORING
            # Process in background
            threading.Thread(target=self._process_finished_task).start()
        return {"success": True}

    def _process_finished_task(self):
        """Process a finished task"""
        # Score the task
        duration = time.time() - self.task_start_time

        # Score the task using the abstract method
        result = self.score_task(self.current_task, duration)

        self.results.append(result)

        # Move to next task or finish
        self.current_task_index += 1
        if self.current_task_index < len(self.tasks):
            # Wait a moment to ensure agent has time to reset
            time.sleep(1)
            self._start_next_task()
        else:
            # Finalize this evaluation run
            self.state = EvaluatorState.IDLE

            # Reset agent to ensure it's ready for future evaluations
            try:
                requests.post(f"{self.agent_url}/reset")
            except requests.exceptions.RequestException as e:
                print(f"Warning: Could not reset agent: {e}")

    @abstractmethod
    def score_task(self, task: TaskConfig, duration: float) -> EvaluationResult:
        """
        Calculate score for a completed task.
        Must be implemented by subclasses.

        Args:
            task: The task configuration that was completed
            duration: How long the task took to complete

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
            task: Configuration for the task to setup
        """
        pass

    def _run_evaluation(self):
        """Run the entire evaluation process"""
        if not self.tasks:
            self.state = EvaluatorState.IDLE
            return

        # Wait for agent to be ready
        self.state = EvaluatorState.WAITING_FOR_AGENT
        max_retries = 10
        retries = 0

        while retries < max_retries:
            if self.agent_api.check_ready():
                self.state = EvaluatorState.AGENT_READY
                break
            retries += 1
            time.sleep(1)

        if self.state != EvaluatorState.AGENT_READY:
            print("Agent not ready after maximum retries")
            self.state = EvaluatorState.IDLE
            return

        # Start the first task
        self._start_next_task()

    def _start_next_task(self):
        """Start the next task in the queue"""
        if self.current_task_index >= len(self.tasks):
            self.state = EvaluatorState.IDLE
            return

        self.current_task = self.tasks[self.current_task_index]
        self.state = EvaluatorState.EVALUATING

        # Configure environment for task
        self._setup_environment(self.current_task)

        # Start the task on the agent
        if not self.agent_api.start_task(self.current_task):
            print(f"Failed to start task {self.current_task_index}")
            self.state = EvaluatorState.IDLE
            return

        self.task_start_time = time.time()

        # Start monitoring thread for timeout
        self.stop_event.clear()
        self.task_thread = threading.Thread(
            target=self._monitor_task, args=(self.current_task,)
        )
        self.task_thread.start()

    def _monitor_task(self, task: TaskConfig):
        """Monitor a running task and handle timeouts"""
        # Wait for the maximum duration plus a buffer
        timeout = task.max_duration_seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if task was marked as finished
            if self.state == EvaluatorState.SCORING:
                return

            # Check for stop event (externally triggered)
            if self.stop_event.wait(0.1):
                return

        # If we get here, we've timed out
        print(f"Task timed out after {timeout} seconds")

        # Try to stop gracefully
        if not self.agent_api.stop_task():
            # If stop fails, try to kill
            self.agent_api.kill()

        # Record timeout result
        result = EvaluationResult(
            task_id=f"task_{self.current_task_index}",
            score=0.0,
            metrics={"timeout": True},
            duration_seconds=timeout,
            success=False,
            notes="Task timed out",
        )

        self.results.append(result)

        # Move to next task
        self.current_task_index += 1
        self.state = EvaluatorState.AGENT_READY
        self._start_next_task()

    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the evaluator server"""
        self.state = EvaluatorState.IDLE
        uvicorn.run(self.app, host=host, port=port)


# --- Concrete Implementation Examples --- #


class MyGameAgent(GameAgent):
    """Example implementation of a game agent"""

    def play_game(self, task: TaskConfig, stop_event: threading.Event):
        """
        Implement the game playing logic here.
        This is what researchers would customize with their models and logic.
        """
        print(f"Playing game: {task.game_name}")
        print(f"Task description: {task.task_description}")

        # Example: Super Hexagon implementation would use screen observations
        # and generate keyboard inputs based on those observations
        activate_module("owa.env.desktop")
        activate_module("owa.env.gst")

        # Setup screen recorder and other components
        # (Simplified for example)

        # Game loop
        start_time = time.time()
        while (
            not stop_event.is_set()
            and time.time() - start_time < task.max_duration_seconds
        ):
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

            # Check for game-specific success condition
            if self._check_success_condition(task):
                # Signal that we're done
                try:
                    requests.post(
                        f"{self.base_url}/agent/finished"
                    )  # TODO: make it internal class function
                except requests.exceptions.RequestException as e:
                    print(f"Request exception: {e}")
                break

        print()
        print("Finished playing game")

    def _check_success_condition(self, task: TaskConfig) -> bool:
        """Check if the success condition for the task has been met"""
        # This would implement game-specific success detection
        # For example, detecting a "victory" screen or a specific score
        # task.success_criteria
        return False


class SimpleEvaluator(Evaluator):
    """Concrete implementation of an Evaluator"""

    def score_task(self, task: TaskConfig, duration: float) -> EvaluationResult:
        """Simple scoring based on task duration and success criteria"""
        # Very basic scoring logic as an example
        success = duration < task.max_duration_seconds
        score = 1.0 if success else duration / task.max_duration_seconds

        # In the real implementation, might want to check env to decide score
        # CALLABLES["screen.capture"]()

        return EvaluationResult(
            task_id=f"task_{self.current_task_index}",
            score=score,
            metrics={"time": duration},
            duration_seconds=duration,
            success=success,
            notes="Task completed successfully" if success else "Task took too long",
        )

    def _setup_environment(self, task: TaskConfig):  # NOTE: make a separate ENV class?
        """Setup the environment for a task"""
        print(f"Setting up environment for {task.game_name}")
        # In a real implementation, this would launch games, configure windows, etc.


# --- Example Tasks --- #


def get_example_tasks() -> List[TaskConfig]:
    """Get a list of example tasks for testing"""
    return [
        TaskConfig(
            game_name="Super Hexagon",
            window_name="Super Hexagon",
            task_description="Survive for at least 5 seconds in the hardest level",
            max_duration_seconds=5,
            success_criteria={"time_survived": 1},
        ),
        # TaskConfig(
        #     game_name="ZType",
        #     window_name="ZType",
        #     task_description="Complete the first level with at least 90% accuracy",
        #     max_duration_seconds=120,
        #     success_criteria={"accuracy": 0.9, "level_completed": True},
        # ),
    ]


# --- Run Functions --- #


def run_agent(model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
    """Run the agent server"""
    agent = MyGameAgent(model_id=model_id)
    print(f"Starting agent server with model: {model_id}")
    agent.run(host="0.0.0.0", port=8000)


def run_agent_background(
    model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", port: int = 8000
):
    """Run the agent server in a background thread and wait for it to be ready

    Args:
        model_id: The model ID to use for the agent
        port: The port to run the agent server on

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
            response = requests.get(f"{agent_url}/status")
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
    """Run the evaluator server"""
    evaluator = SimpleEvaluator()
    print("Starting evaluator server")
    evaluator.run(host="0.0.0.0", port=8001)


def run_evaluator_background(port: int = 8001):
    """Run the evaluator server in a background thread and wait for it to be ready

    Args:
        port: The port to run the evaluator server on

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
            response = requests.get(f"{evaluator_url}/status")
            if response.status_code == 200:
                print(f"Evaluator server is ready after {i + 1} attempts")
                return evaluator_url
        except requests.exceptions.RequestException:
            pass

        if i == max_retries - 1:
            print("Failed to connect to evaluator server")
            return None

        time.sleep(1)


def run_example_evaluation(agent_url: str, evaluator_url: str):
    """Run an example evaluation"""
    # This would be run after both agent and evaluator servers are running
    api = EvaluatorAPI(evaluator_url=evaluator_url)

    # Register the agent
    print("Registering agent with evaluator...")
    if not api.register_agent(agent_url=agent_url):
        print("Failed to register agent")
        return

    # Start evaluation
    tasks = get_example_tasks()
    print(f"Starting evaluation with {len(tasks)} tasks...")
    if not api.start_evaluation(tasks):
        print("Failed to start evaluation")
        return

    # Wait for completion and get results
    print("Waiting for evaluation to complete...")
    max_wait = 300  # 5 minutes
    start_time = time.time()
    results = []

    while time.time() - start_time < max_wait:
        results = api.get_results()
        if len(results) == len(tasks):
            break
        time.sleep(5)

    # Print results
    print("\nEvaluation Results:")
    for i, result in enumerate(results):
        print(f"Task {i + 1}: Score = {result.score:.2f}, Success = {result.success}")
        print(f"  Duration: {result.duration_seconds:.1f} seconds")
        print(f"  Notes: {result.notes}")
        print()

    # Run a second evaluation to demonstrate multiple evaluation support
    print("\nRunning a second evaluation with the same tasks...")
    if not api.start_evaluation(tasks):
        print("Failed to start second evaluation")
        return

    # Wait for completion of the second evaluation
    print("Waiting for second evaluation to complete...")
    start_time = time.time()
    results = []

    while time.time() - start_time < max_wait:
        results = api.get_results()
        if len(results) == len(tasks):
            break
        time.sleep(5)

    # Print results of second evaluation
    print("\nSecond Evaluation Results:")
    for i, result in enumerate(results):
        print(f"Task {i + 1}: Score = {result.score:.2f}, Success = {result.success}")
        print(f"  Duration: {result.duration_seconds:.1f} seconds")
        print(f"  Notes: {result.notes}")
        print()


def run_example_nonblocking(
    model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
):
    """Run an example evaluation without blocking

    This function starts both the agent and evaluator servers in background threads,
    waits for them to be ready, and then runs an example evaluation against them.
    """
    import time

    # Start the agent and evaluator in background threads and wait for them to be ready
    print("Starting servers and waiting for them to be ready...")
    agent_url = run_agent_background(model_id)
    evaluator_url = run_evaluator_background()

    if not agent_url or not evaluator_url:
        print("Failed to start servers")
        return

    # Run the example evaluation
    print("Both servers are ready, running example evaluation...")
    run_example_evaluation(agent_url, evaluator_url)

    # Keep the main thread running to allow the background servers to continue
    print("\nServers are running in the background. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")


class Mode(Enum):
    AGENT = "agent"
    EVALUATOR = "evaluator"
    EXAMPLE = "example"
    NONBLOCKING = "nonblocking"


def main(
    mode: Annotated[
        Mode,
        typer.Option(
            help="Mode to run: 'agent', 'evaluator', 'example', or 'nonblocking'",
        ),
    ] = Mode.NONBLOCKING.value,
    model_id: Annotated[
        str,
        typer.Option(help="Model ID to use for the agent"),
    ] = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
):
    """Main entry point"""
    if mode == Mode.AGENT:
        run_agent(model_id)
    elif mode == Mode.EVALUATOR:
        run_evaluator()
    elif mode == Mode.EXAMPLE:
        run_example_evaluation("http://localhost:8000", "http://localhost:8001")
    elif mode == Mode.NONBLOCKING:
        run_example_nonblocking(model_id)
    else:
        raise ValueError(f"Unknown mode: {mode=}")


if __name__ == "__main__":
    typer.run(main)
