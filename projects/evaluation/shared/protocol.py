# shared/protocol.py
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageType(Enum):
    PING = "ping"
    READY = "ready"
    TASK = "task"
    ACK = "ack"
    HEARTBEAT = "heartbeat"
    STOP = "stop"
    COMPLETION = "completion"
    ERROR = "error"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    STOPPED = "stopped"


@dataclass
class Message:
    type: MessageType
    payload: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({"type": self.type.value, "payload": self.payload})

    @classmethod
    def from_json(cls, data: str) -> "Message":
        parsed = json.loads(data)
        return cls(type=MessageType(parsed["type"]), payload=parsed["payload"])


@dataclass
class TaskSpecification:
    id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    timeout: int  # in seconds
    success_criteria: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "timeout": self.timeout,
            "success_criteria": self.success_criteria,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSpecification":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            steps=data["steps"],
            timeout=data["timeout"],
            success_criteria=data["success_criteria"],
        )


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    success: bool
    error_message: Optional[str] = None
    screenshots: List[str] = None  # Base64 encoded screenshots
    metrics: Dict[str, Any] = None  # Performance metrics
    logs: List[str] = None  # Execution logs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "success": self.success,
            "error_message": self.error_message,
            "screenshots": self.screenshots,
            "metrics": self.metrics,
            "logs": self.logs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResult":
        return cls(
            task_id=data["task_id"],
            status=TaskStatus(data["status"]),
            success=data["success"],
            error_message=data.get("error_message"),
            screenshots=data.get("screenshots"),
            metrics=data.get("metrics"),
            logs=data.get("logs"),
        )
