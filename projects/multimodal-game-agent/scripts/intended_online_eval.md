# Documentation for Users of Agent and Evaluator


## Agent
```python

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
        while not stop_event.is_set() and time.time() - start_time < task.max_duration_seconds:
            window_active = CALLABLES["window.is_active"](task.window_name)
            if not window_active:
                print(f"Window {task.window_name} is not active")
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
                print(f"key {39} pressed")

            # Check for game-specific success condition
            if self._check_success_condition(task):
                # Signal that we're done
                try:
                    requests.post(f"{self.base_url}/agent/finished")  # TODO: make it internal class function
                except requests.exceptions.RequestException as e:
                    print(f"Request exception: {e}")
                break

        print("Finished playing game")

    def _check_success_condition(self, task: TaskConfig) -> bool:
        """Check if the success condition for the task has been met"""
        # This would implement game-specific success detection
        # For example, detecting a "victory" screen or a specific score
        # task.success_criteria
        return False

```

## Evaluator
```python
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

    def _setup_environment(self, task: TaskConfig):
        """Setup the environment for a task"""
        print(f"Setting up environment for {task.game_name}")
        # In a real implementation, this would launch games, configure windows, etc.


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


```

## Run
```python

def run_example_nonblocking(model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
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

```