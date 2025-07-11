import logging
import threading
import time
import uuid
from typing import Any, Optional

import requests
from fastapi import Response
from pydantic import BaseModel, Field

from owa_game_agent.constants import ENDPOINTS, NETWORK, TIMES

logger = logging.getLogger(__name__)

# ----- Data Classes ----- #


class Task(BaseModel):
    """Configuration for a task in an environment"""

    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )  # NOTE: every task should have a unique ID. DO NOT reuse the same task object.
    env_name: str
    window_name: str  # NOTE: currently owa.env.desktop finds first window that is a superstring of window_name
    task_description: str
    timeout: int
    check_env_interval_seconds: float = 1.0  # Used for an interval between calling _check_env_continue_timer()
    success_criteria: dict[str, Any]
    num_frames: int = 3  # number of frames to capture and handle for the task


class EvaluationResult(BaseModel):
    """Results from an evaluation"""

    task_id: str
    metrics: dict[str, Any]
    notes: Optional[str] = None


# ----- Helper functions ----- #


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
    max_retries = TIMES.SERVER_STARTUP_MAX_RETRIES
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

        time.sleep(TIMES.SERVER_STARTUP_RETRY_INTERVAL)


def handle_response_errors(response: Response, raise_error: bool = False):
    """Helper function to handle HTTP errors from responses"""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP {response.status_code}: {response.text}"
        logger.warning(error_msg)
        if raise_error:
            raise requests.exceptions.HTTPError(error_msg) from e
