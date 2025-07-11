from owa.env.desktop.constants import VK

RECORD_START_STOP_KEY = VK.F9
RECORD_PAUSE_KEY = VK.F10


# Network constants
class NETWORK:
    # Host configuration
    DEFAULT_HOST = "0.0.0.0"
    LOCALHOST = "127.0.0.1"

    # Port configuration
    AGENT_PORT = 8000
    EVALUATOR_PORT = 8001

    # NOTE: URL constants should not be used directly, rather passed as arguments
    _AGENT_URL = f"http://{LOCALHOST}:{AGENT_PORT}"
    _EVALUATOR_URL = f"http://{LOCALHOST}:{EVALUATOR_PORT}"


# API Endpoints
class ENDPOINTS:
    # Health check endpoints
    AGENT_STATUS = "/agent/status"
    EVALUATOR_STATUS = "/evaluator/status"

    # Agent endpoints
    AGENT_TASK_START = "/agent/task/start"
    AGENT_TASK_STOP = "/agent/task/stop"
    AGENT_TASK_FINISHED = "/agent/task/finished"
    AGENT_KILL = "/agent/kill"
    AGENT_RESET = "/agent/reset"
    AGENT_REGISTER_EVALUATOR = "/agent/register_evaluator"

    # Evaluator endpoints
    EVALUATOR_REGISTER_AGENT = "/evaluator/register_agent"
    EVALUATOR_EVALUATION_START = "/evaluator/evaluation/start"
    EVALUATOR_AGENT_FINISHED = "/evaluator/agent_finished"
    EVALUATOR_RESET = "/evaluator/reset"
    EVALUATOR_EVALUATION_RESULTS = "/evaluator/evaluation/results"


# Timeouts and retry settings
class TIMES:
    SERVER_STARTUP_MAX_RETRIES = 10
    SERVER_STARTUP_RETRY_INTERVAL = 1.0  # seconds
    TASK_CLEANUP_TIMEOUT = 2.0  # seconds
    AGENT_RESET_DELAY = 1.0  # seconds
    THREAD_JOIN_TIMEOUT = 2.0  # seconds
    EVALUATION_POLL_INTERVAL = 0.5  # seconds
    KEYBOARD_PRESS_DELAY = 0.05  # seconds
    BUSY_WAIT_PREVENT_EVALUATOR = 0.1  # seconds
    BUSY_WAIT_PREVENT_AGENT = 0.01  # seconds, we want to be more responsive for the agent
    TIMESTAMP_INTERVAL = 0.05  # seconds
