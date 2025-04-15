# Documentation for Users of Agent and Evaluator

## Overview
The Agent-Evaluator framework provides a robust infrastructure for evaluating autonomous game-playing agents in real-time. This documentation helps users implement and utilize this framework effectively.

## Agent
```python
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

        def on_keyboard_event(keyboard_event: KeyboardEvent):
            if keyboard_event.vk == VK.F10:
                logger.debug("Stopping agent")
                self.stop_event.set()

        keyboard_listener = LISTENERS["keyboard"]().configure(callback=on_keyboard_event)
        keyboard_listener.start()

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

        with env_lock:
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
                time.sleep(TIMES.KEYBOARD_PRESS_DELAY)
                CALLABLES["keyboard.release"](VK.RIGHT)
                logger.debug(f"key {VK.RIGHT} pressed")

                return True  # Continue the task

            else:
                # logger.error(f"{self._play_env.__name__}(): task should not continue. This should not happen.")
                return False  # Do not continue the task. Evaluation will be made by the evaluator.
```

### Agent Implementation Guide

When implementing your own agent, you need to create a subclass of the `Agent` class and implement the required methods:

1. **Constructor (`__init__`)**: Initialize your agent with necessary components:
   - Activate required modules (`owa.env.desktop`, `owa.env.gst`, etc.)
   - Set up any external clients (like OpenAI)
   - Configure input listeners (keyboard, mouse)
   - Set up any model or ML components

2. **`_play_env(task: Task) -> bool`**: This is the core method where your agent logic lives:
   - This method is called repeatedly by the framework during evaluation
   - It should perform a single step/action in the environment
   - Return `True` if the agent should continue playing, `False` otherwise
   - Use environment observations to determine actions
   - Implement any game-specific logic

### Key Agent Components

- **Window Management**: Use `CALLABLES["window.make_active"](task.window_name)` to focus the game window
- **Input Generation**: Use `CALLABLES["keyboard.press/release"]` to simulate keyboard/mouse inputs
- **Safety Controls**: Implement keyboard listeners for manual intervention (e.g., F10 to stop)

## Evaluator
```python
class MySuperHexagonEvaluator(Evaluator):
    """Example implementation of an Evaluator"""

    def __init__(self):
        super().__init__()

        def on_keyboard_event(keyboard_event: KeyboardEvent):
            if keyboard_event.vk == VK.F10:
                logger.debug("Stopping evaluator")
                self.stop_event.set()

        keyboard_listener = LISTENERS["keyboard"]().configure(callback=on_keyboard_event)
        keyboard_listener.start()

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

        with env_lock:
            # make the window active
            CALLABLES["window.make_active"](task.window_name)

            # for super hexagon, we need to press space
            CALLABLES["keyboard.press"](VK.SPACE)
            time.sleep(TIMES.KEYBOARD_PRESS_DELAY)
            CALLABLES["keyboard.release"](VK.SPACE)
```

### Evaluator Implementation Guide

When implementing your own evaluator, create a subclass of the `Evaluator` class and implement these methods:

1. **Constructor (`__init__`)**: Set up your evaluator with:
   - External clients (like OpenAI for vision analysis)
   - Input listeners for manual control
   - Game-specific assessment tools

2. **`_check_env_continue(task: Task) -> bool`**: This method determines if a task should continue:
   - Analyze game screens to detect game-over states
   - Look for specific visual indicators (like "retry" buttons)
   - Return `False` when the game has ended, `True` otherwise

3. **`_score_task(task: Task, task_elapsed_time: float, note: str = "") -> EvaluationResult`**: This method calculates the final score:
   - Extract performance metrics from the game (score, time, etc.)
   - Compare against success criteria defined in the task
   - Return an `EvaluationResult` with metrics and success determination

4. **`_setup_environment(task: Task)`**: This method prepares the environment for evaluation:
   - Focus the game window
   - Reset the game state (press restart buttons, etc.)
   - Configure any game-specific settings

### Key Evaluator Components

- **Game State Detection**: Use screen analysis to determine game state
- **Scoring Logic**: Implement game-specific scoring and success criteria
- **Environment Setup**: Handle game resets and configuration


## Running Evaluations

There are several ways to run evaluations:

```python
def run_evaluation_client_with_server(
    task: Task,
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
    agent = MySuperHexagonAgent()
    evaluator = MySuperHexagonEvaluator()

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
    time.sleep(TIMEOUTS.SERVER_STARTUP_RETRY_INTERVAL)

    # Run evaluation
    print("Running evaluation...")
    run_evaluation_client(agent_url, evaluator_url, task)

    # Wait for user input to exit
    input("Press Enter to exit...")
```

### Running Options:

1. **Combined Client/Server (Recommended)**:
   - Use `run_evaluation_client_with_server(task)` to start both agent and evaluator
   - This is the simplest way to run an evaluation

2. **Separate runs of Server and Client**:
   - Run `agent.run(host, port)` and `evaluator.run(host, port)` separately
   - Use `run_evaluation_client(agent_url, evaluator_url, task)` to connect them

3. **Parallel Evaluations**:
   - Use `run_evaluation_client_with_server_parallel()` to evaluate multiple tasks concurrently



