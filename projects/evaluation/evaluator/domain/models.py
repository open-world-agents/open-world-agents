# evaluator/domain/models.py
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from shared.protocol import TaskResult, TaskSpecification, TaskStatus


@dataclass
class Benchmark:
    id: str
    name: str
    description: str
    tasks: List[TaskSpecification]

    @classmethod
    def create(cls, name: str, description: str, tasks: List[TaskSpecification]) -> "Benchmark":
        return cls(id=str(uuid.uuid4()), name=name, description=description, tasks=tasks)


@dataclass
class AgentProfile:
    id: str
    name: str
    endpoint: str

    @classmethod
    def create(cls, name: str, endpoint: str) -> "AgentProfile":
        return cls(id=str(uuid.uuid4()), name=name, endpoint=endpoint)


@dataclass
class EvaluationResult:
    benchmark_id: str
    agent_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    task_results: Dict[str, TaskResult] = None
    score: Optional[float] = None

    @classmethod
    def create(cls, benchmark_id: str, agent_id: str) -> "EvaluationResult":
        return cls(benchmark_id=benchmark_id, agent_id=agent_id, start_time=datetime.now(), task_results={})

    def add_task_result(self, task_id: str, result: TaskResult):
        if self.task_results is None:
            self.task_results = {}
        self.task_results[task_id] = result

    def complete(self):
        self.end_time = datetime.now()
        # Calculate overall score based on task results
        successful_tasks = sum(1 for result in self.task_results.values() if result.success)
        total_tasks = len(self.task_results)
        self.score = (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0
