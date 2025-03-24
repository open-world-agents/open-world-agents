# evaluator/infrastructure/agent_client.py
import json
import logging
import time
from typing import Any, Dict, Optional

import requests
from shared.protocol import Message, MessageType, TaskResult, TaskSpecification

logger = logging.getLogger(__name__)


class AgentClient:
    def __init__(self, agent_endpoint: str, timeout: int = 10):
        self.agent_endpoint = agent_endpoint
        self.timeout = timeout

    def check_health(self) -> bool:
        """Check if the agent is ready and responding."""
        try:
            message = Message(type=MessageType.PING, payload={})

            response = requests.post(
                f"{self.agent_endpoint}/agent",
                json={"type": message.type.value, "payload": message.payload},
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.error(f"Agent health check failed with status code: {response.status_code}")
                return False

            data = response.json()
            return data.get("type") == MessageType.READY.value

        except Exception as e:
            logger.exception(f"Error checking agent health: {str(e)}")
            return False

    def start_task(self, task: TaskSpecification) -> bool:
        """Send a task to the agent."""
        try:
            message = Message(type=MessageType.TASK, payload=task.to_dict())

            response = requests.post(
                f"{self.agent_endpoint}/agent",
                json={"type": message.type.value, "payload": message.payload},
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.error(f"Failed to start task: {response.status_code}")
                return False

            data = response.json()
            return data.get("type") == MessageType.ACK.value

        except Exception as e:
            logger.exception(f"Error starting task: {str(e)}")
            return False

    def stop_task(self) -> Optional[TaskResult]:
        """Stop the current task and get results."""
        try:
            message = Message(type=MessageType.STOP, payload={})

            response = requests.post(
                f"{self.agent_endpoint}/agent",
                json={"type": message.type.value, "payload": message.payload},
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.error(f"Failed to stop task: {response.status_code}")
                return None

            data = response.json()
            if data.get("type") != MessageType.COMPLETION.value:
                logger.error(f"Unexpected response type: {data.get('type')}")
                return None

            return TaskResult.from_dict(data["payload"])

        except Exception as e:
            logger.exception(f"Error stopping task: {str(e)}")
            return None

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get the current status of the agent."""
        try:
            response = requests.get(f"{self.agent_endpoint}/agent/status", timeout=self.timeout)

            if response.status_code != 200:
                logger.error(f"Failed to get status: {response.status_code}")
                return None

            return response.json()

        except Exception as e:
            logger.exception(f"Error getting status: {str(e)}")
            return None
