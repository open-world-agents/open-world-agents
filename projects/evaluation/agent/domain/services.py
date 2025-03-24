# agent/domain/services.py
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from shared.protocol import TaskResult, TaskSpecification, TaskStatus

logger = logging.getLogger(__name__)


class TaskExecutionService:
    def __init__(self):
        self.current_task: Optional[TaskSpecification] = None
        self.task_status: TaskStatus = TaskStatus.PENDING
        self.start_time: float = 0
        self.task_handlers: Dict[str, Callable] = {}
        self.execution_logs: List[str] = []
        self.screenshots: List[str] = []

    def register_task_handler(self, task_name: str, handler: Callable):
        """Register a handler function for a specific task type."""
        self.task_handlers[task_name] = handler

    def start_task(self, task: TaskSpecification) -> bool:
        """Begin execution of a task."""
        if self.current_task is not None and self.task_status == TaskStatus.RUNNING:
            logger.error("Cannot start new task while another is running")
            return False

        self.current_task = task
        self.task_status = TaskStatus.RUNNING
        self.start_time = time.time()
        self.execution_logs = [f"Started task {task.id}: {task.name}"]
        self.screenshots = []

        logger.info(f"Starting task: {task.name} (ID: {task.id})")
        return True

    def stop_task(self) -> TaskResult:
        """Stop the current task and return results."""
        if self.current_task is None:
            logger.warning("No task is currently running")
            return TaskResult(
                task_id="unknown", status=TaskStatus.STOPPED, success=False, error_message="No task was running"
            )

        duration = time.time() - self.start_time
        self.task_status = TaskStatus.STOPPED

        result = TaskResult(
            task_id=self.current_task.id,
            status=self.task_status,
            success=False,  # Stopped tasks are considered unsuccessful
            error_message="Task was manually stopped",
            screenshots=self.screenshots,
            metrics={"duration": duration},
            logs=self.execution_logs,
        )

        logger.info(f"Task stopped: {self.current_task.name} (Duration: {duration:.2f}s)")
        self.current_task = None
        return result

    def execute_task(self) -> TaskResult:
        """Execute the current task and return results."""
        if self.current_task is None or self.task_status != TaskStatus.RUNNING:
            logger.error("No task is currently running")
            return TaskResult(
                task_id="unknown", status=TaskStatus.FAILED, success=False, error_message="No task was running"
            )

        task = self.current_task
        handler = self.task_handlers.get(task.name)

        if not handler:
            error_msg = f"No handler registered for task type: {task.name}"
            logger.error(error_msg)
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                success=False,
                error_message=error_msg,
                logs=self.execution_logs,
            )

        try:
            # Execute the task handler
            self.log(f"Executing task: {task.name}")
            success = handler(task, self)

            duration = time.time() - self.start_time
            status = TaskStatus.COMPLETED if success else TaskStatus.FAILED

            result = TaskResult(
                task_id=task.id,
                status=status,
                success=success,
                screenshots=self.screenshots,
                metrics={"duration": duration},
                logs=self.execution_logs,
            )

            self.log(f"Task completed with success={success}")
            self.task_status = status
            return result

        except Exception as e:
            duration = time.time() - self.start_time
            error_msg = f"Error executing task: {str(e)}"
            logger.exception(error_msg)
            self.log(error_msg)

            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                success=False,
                error_message=error_msg,
                screenshots=self.screenshots,
                metrics={"duration": duration},
                logs=self.execution_logs,
            )

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent and any running task."""
        if self.current_task is None:
            return {"status": "ready", "task": None}

        elapsed = time.time() - self.start_time
        return {
            "status": self.task_status.value,
            "task": {
                "id": self.current_task.id,
                "name": self.current_task.name,
                "elapsed_time": elapsed,
                "timeout": self.current_task.timeout,
            },
        }

    def log(self, message: str):
        """Add a log message."""
        logger.info(message)
        self.execution_logs.append(message)

    def add_screenshot(self, screenshot_base64: str):
        """Add a screenshot (base64 encoded)."""
        self.screenshots.append(screenshot_base64)
