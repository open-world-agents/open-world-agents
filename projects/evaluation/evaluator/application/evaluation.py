# evaluator/application/evaluation.py
import logging
import time
from typing import Optional

from evaluator.domain.models import AgentProfile, Benchmark, EvaluationResult
from evaluator.infrastructure.agent_client import AgentClient
from shared.protocol import TaskResult, TaskStatus

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    def __init__(self):
        self.current_evaluation: Optional[EvaluationResult] = None

    def run_benchmark(self, benchmark: Benchmark, agent: AgentProfile) -> EvaluationResult:
        """Run a benchmark against an agent."""
        logger.info(f"Starting benchmark: {benchmark.name} against agent: {agent.name}")

        # Create a new evaluation result
        self.current_evaluation = EvaluationResult.create(benchmark_id=benchmark.id, agent_id=agent.id)

        # Create an agent client
        client = AgentClient(agent.endpoint)

        # Check if the agent is healthy
        if not client.check_health():
            logger.error(f"Agent is not healthy: {agent.name}")
            self.current_evaluation.complete()
            return self.current_evaluation

        # Execute each task in the benchmark
        for task in benchmark.tasks:
            logger.info(f"Starting task: {task.name}")

            # Start the task
            if not client.start_task(task):
                logger.error(f"Failed to start task: {task.name}")
                self.current_evaluation.add_task_result(
                    task.id,
                    TaskResult(
                        task_id=task.id, status=TaskStatus.FAILED, success=False, error_message="Failed to start task"
                    ),
                )
                continue

            # Monitor the task until completion or timeout
            start_time = time.time()
            timeout_occurred = False

            while time.time() - start_time < task.timeout:
                # Check status
                status = client.get_status()

                if status is None:
                    logger.error(f"Failed to get status for task: {task.name}")
                    break

                if status.get("status") == "completed":
                    logger.info(f"Task completed: {task.name}")
                    break

                # Wait before checking again
                time.sleep(1)
            else:
                # Timeout occurred
                logger.warning(f"Task timed out: {task.name}")
                timeout_occurred = True

            # Stop the task and get results
            result = client.stop_task()

            if result is None:
                logger.error(f"Failed to get results for task: {task.name}")
                self.current_evaluation.add_task_result(
                    task.id,
                    TaskResult(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        success=False,
                        error_message="Failed to get task results",
                    ),
                )
            else:
                # If timeout occurred, override the status
                if timeout_occurred and result.status != TaskStatus.COMPLETED:
                    result = TaskResult(
                        task_id=result.task_id,
                        status=TaskStatus.TIMEOUT,
                        success=False,
                        error_message="Task exceeded timeout",
                        screenshots=result.screenshots,
                        metrics=result.metrics,
                        logs=result.logs,
                    )

                self.current_evaluation.add_task_result(task.id, result)

        # Complete the evaluation
        self.current_evaluation.complete()
        logger.info(f"Benchmark completed: {benchmark.name}, Score: {self.current_evaluation.score}%")
        return self.current_evaluation
