"""
Constants used in the online evaluation system.
This file centralizes all configuration values to make them easier to modify.
"""

# Network constants
class Network:
    # Host configuration
    DEFAULT_HOST = "0.0.0.0"
    LOCALHOST = "localhost"
    
    # Port configuration
    AGENT_PORT = 8000
    EVALUATOR_PORT = 8001
    
    # URL construction
    AGENT_URL = f"http://{LOCALHOST}:{AGENT_PORT}"
    EVALUATOR_URL = f"http://{LOCALHOST}:{EVALUATOR_PORT}"

# API Endpoints
class Endpoints:
    # Health check endpoints
    AGENT_STATUS = "/agent/status"
    EVALUATOR_STATUS = "/evaluator/status"
    
    # Agent endpoints
    AGENT_TASK_START = "/agent/task/start"
    AGENT_TASK_STOP = "/agent/task/stop"
    AGENT_TASK_FINISHED = "/agent/task/finished"
    AGENT_KILL = "/agent/kill"
    AGENT_RESET = "/agent/reset"
    
    # Evaluator endpoints
    EVALUATOR_REGISTER_AGENT = "/evaluator/register_agent"
    EVALUATOR_EVALUATION_START = "/evaluator/evaluation/start"
    EVALUATOR_AGENT_FINISHED = "/evaluator/agent_finished"
    EVALUATOR_RESET = "/evaluator/reset"
    EVALUATOR_EVALUATION_RESULTS = "/evaluator/evaluation/results"

# Timeouts and retry settings
class Timeouts:
    SERVER_STARTUP_MAX_RETRIES = 10
    SERVER_STARTUP_RETRY_INTERVAL = 1.0  # seconds
    TASK_CLEANUP_DELAY = 0.5  # seconds
    AGENT_RESET_DELAY = 1.0  # seconds
    THREAD_JOIN_TIMEOUT = 2.0  # seconds
    EVALUATION_POLL_INTERVAL = 2.0  # seconds

# Default values
class Defaults:
    DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    KEYBOARD_PRESS_DELAY = 0.05  # seconds
    ENV_CHECK_INTERVAL = 0.1  # seconds
    RIGHT_ARROW_KEY = 39  # Key code for right arrow 