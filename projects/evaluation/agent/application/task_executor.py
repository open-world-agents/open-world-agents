# agent/application/task_executor.py
import logging
import threading
import time
from typing import Any, Dict, Optional

from agent.domain.services import TaskExecutionService
from shared.protocol import TaskResult, TaskSpecification, TaskStatus

logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(self):
        self.service = TaskExecutionService()
        self.task_thread: Optional[threading.Thread] = None
        self.is_running = False

    def start_task(self, task_spec: TaskSpecification) -> bool:
        """Start a task in a separate thread."""
        if self.is_running:
            logger.warning("Cannot start a new task while another is running")
            return False

        if not self.service.start_task(task_spec):
            return False

        self.is_running = True
        self.task_thread = threading.Thread(target=self._execute_task_with_timeout, args=(task_spec,))
        self.task_thread.daemon = True
        self.task_thread.start()
        return True

    def _execute_task_with_timeout(self, task_spec: TaskSpecification):
        """Execute the task with a timeout."""
        result = None
        try:
            result = self.service.execute_task()
        except Exception as e:
            logger.exception(f"Unhandled exception in task execution: {str(e)}")
            result = TaskResult(
                task_id=task_spec.id,
                status=TaskStatus.FAILED,
                success=False,
                error_message=f"Unhandled exception: {str(e)}",
            )
        finally:
            self.is_running = False

    def stop_task(self) -> TaskResult:
        """Stop the currently running task."""
        if not self.is_running:
            return TaskResult(
                task_id="unknown", status=TaskStatus.STOPPED, success=False, error_message="No task is running"
            )

        result = self.service.stop_task()
        self.is_running = False
        return result

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of any running task."""
        return self.service.get_status()
