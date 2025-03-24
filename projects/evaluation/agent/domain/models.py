# agent/domain/models.py
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskState(Enum):
    """Possible states for a task."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    TIMEOUT = "timeout"


@dataclass
class TaskContext:
    """Context for a running task."""

    task_id: str
    name: str
    start_time: datetime
    state: TaskState = TaskState.PENDING
    end_time: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, task_id: str, name: str) -> "TaskContext":
        return cls(task_id=task_id, name=name, start_time=datetime.now())

    def add_log(self, message: str) -> None:
        """Add a log message."""
        self.logs.append(message)

    def add_screenshot(self, screenshot: str) -> None:
        """Add a screenshot (base64 encoded)."""
        self.screenshots.append(screenshot)

    def add_metric(self, name: str, value: Any) -> None:
        """Add a performance metric."""
        self.metrics[name] = value

    def complete(self, success: bool) -> None:
        """Mark the task as completed."""
        self.state = TaskState.COMPLETED if success else TaskState.FAILED
        self.end_time = datetime.now()

    def stop(self) -> None:
        """Mark the task as stopped."""
        self.state = TaskState.STOPPED
        self.end_time = datetime.now()


@dataclass
class Agent:
    """Agent information."""

    id: str
    name: str
    version: str
    capabilities: List[str]
    current_task: Optional[TaskContext] = None

    @classmethod
    def create(cls, name: str, version: str, capabilities: List[str]) -> "Agent":
        return cls(id=str(uuid.uuid4()), name=name, version=version, capabilities=capabilities)

    def start_task(self, task_id: str, name: str) -> TaskContext:
        """Start a new task."""
        if self.current_task is not None and self.current_task.state == TaskState.RUNNING:
            raise ValueError("Cannot start a new task while another is running")

        self.current_task = TaskContext.create(task_id, name)
        self.current_task.state = TaskState.RUNNING
        return self.current_task

    def stop_current_task(self) -> Optional[TaskContext]:
        """Stop the current task."""
        if self.current_task is None:
            return None

        self.current_task.stop()
        return self.current_task

    def complete_current_task(self, success: bool) -> Optional[TaskContext]:
        """Complete the current task."""
        if self.current_task is None:
            return None

        self.current_task.complete(success)
        return self.current_task
