# evaluator/infrastructure/api.py
import logging
import uuid
from typing import Any, Dict, List, Optional

from evaluator.application.evaluation import BenchmarkRunner
from evaluator.domain.models import AgentProfile, Benchmark, EvaluationResult
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from shared.protocol import TaskSpecification

logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Evaluator API")
benchmark_runner = BenchmarkRunner()

# In-memory storage for demo purposes
# In a real implementation, this would be a database
benchmarks = {}
agents = {}
evaluations = {}


# Request/Response Models
class TaskSpecModel(BaseModel):
    name: str
    description: str
    steps: List[Dict[str, Any]]
    timeout: int
    success_criteria: Dict[str, Any]


class BenchmarkModel(BaseModel):
    name: str
    description: str
    tasks: List[TaskSpecModel]


class AgentModel(BaseModel):
    name: str
    endpoint: str


class EvaluationRequestModel(BaseModel):
    benchmark_id: str
    agent_id: str


class EvaluationStatusModel(BaseModel):
    id: str
    benchmark_id: str
    agent_id: str
    status: str
    progress: float
    completed_tasks: int
    total_tasks: int


# Routes
@app.post("/benchmarks", response_model=Dict[str, Any])
async def create_benchmark(benchmark: BenchmarkModel):
    """Create a new benchmark specification."""
    # Convert task models to TaskSpecification objects
    tasks = []
    for task_model in benchmark.tasks:
        task = TaskSpecification(
            id=str(uuid.uuid4()),
            name=task_model.name,
            description=task_model.description,
            steps=task_model.steps,
            timeout=task_model.timeout,
            success_criteria=task_model.success_criteria,
        )
        tasks.append(task)

    # Create the benchmark
    new_benchmark = Benchmark.create(name=benchmark.name, description=benchmark.description, tasks=tasks)

    # Store the benchmark
    benchmarks[new_benchmark.id] = new_benchmark

    return {
        "id": new_benchmark.id,
        "name": new_benchmark.name,
        "description": new_benchmark.description,
        "task_count": len(new_benchmark.tasks),
    }


@app.get("/benchmarks", response_model=List[Dict[str, Any]])
async def list_benchmarks():
    """List all available benchmarks."""
    return [
        {"id": b.id, "name": b.name, "description": b.description, "task_count": len(b.tasks)}
        for b in benchmarks.values()
    ]


@app.get("/benchmarks/{benchmark_id}", response_model=Dict[str, Any])
async def get_benchmark(benchmark_id: str):
    """Get details for a specific benchmark."""
    if benchmark_id not in benchmarks:
        raise HTTPException(status_code=404, detail="Benchmark not found")

    benchmark = benchmarks[benchmark_id]
    return {
        "id": benchmark.id,
        "name": benchmark.name,
        "description": benchmark.description,
        "tasks": [
            {"id": task.id, "name": task.name, "description": task.description, "timeout": task.timeout}
            for task in benchmark.tasks
        ],
    }


@app.post("/agents", response_model=Dict[str, Any])
async def register_agent(agent: AgentModel):
    """Register a new agent for evaluation."""
    new_agent = AgentProfile.create(name=agent.name, endpoint=agent.endpoint)

    agents[new_agent.id] = new_agent

    return {"id": new_agent.id, "name": new_agent.name, "endpoint": new_agent.endpoint}


@app.get("/agents", response_model=List[Dict[str, Any]])
async def list_agents():
    """List all registered agents."""
    return [{"id": a.id, "name": a.name, "endpoint": a.endpoint} for a in agents.values()]


@app.post("/evaluations", response_model=Dict[str, Any])
async def start_evaluation(request: EvaluationRequestModel, background_tasks: BackgroundTasks):
    """Start a new benchmark evaluation."""
    # Check if benchmark and agent exist
    if request.benchmark_id not in benchmarks:
        raise HTTPException(status_code=404, detail="Benchmark not found")

    if request.agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    benchmark = benchmarks[request.benchmark_id]
    agent = agents[request.agent_id]

    # Create evaluation result
    evaluation = EvaluationResult.create(benchmark_id=benchmark.id, agent_id=agent.id)
    evaluations[evaluation.benchmark_id + "_" + evaluation.agent_id] = evaluation

    # Run evaluation in background
    background_tasks.add_task(_run_evaluation, benchmark, agent, evaluation)

    return {"benchmark_id": benchmark.id, "agent_id": agent.id, "status": "running", "message": "Evaluation started"}


@app.get("/evaluations", response_model=List[Dict[str, Any]])
async def list_evaluations():
    """List all evaluations."""
    return [
        {
            "benchmark_id": e.benchmark_id,
            "agent_id": e.agent_id,
            "start_time": e.start_time,
            "end_time": e.end_time,
            "score": e.score,
            "status": "completed" if e.end_time else "running",
        }
        for e in evaluations.values()
    ]


@app.get("/evaluations/{benchmark_id}/{agent_id}", response_model=Dict[str, Any])
async def get_evaluation_result(benchmark_id: str, agent_id: str):
    """Get detailed results for a specific evaluation."""
    eval_key = benchmark_id + "_" + agent_id
    if eval_key not in evaluations:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    evaluation = evaluations[eval_key]

    # If evaluation is still running
    if not evaluation.end_time:
        completed_tasks = len(evaluation.task_results)
        total_tasks = len(benchmarks[benchmark_id].tasks)
        progress = completed_tasks / total_tasks if total_tasks > 0 else 0

        return {
            "benchmark_id": evaluation.benchmark_id,
            "agent_id": evaluation.agent_id,
            "start_time": evaluation.start_time,
            "status": "running",
            "progress": progress,
            "completed_tasks": completed_tasks,
            "total_tasks": total_tasks,
        }

    # If evaluation is completed
    task_results = []
    for task_id, result in evaluation.task_results.items():
        task_results.append(
            {
                "task_id": task_id,
                "success": result.success,
                "status": result.status.value,
                "error_message": result.error_message,
            }
        )

    return {
        "benchmark_id": evaluation.benchmark_id,
        "agent_id": evaluation.agent_id,
        "start_time": evaluation.start_time,
        "end_time": evaluation.end_time,
        "score": evaluation.score,
        "status": "completed",
        "task_results": task_results,
    }


async def _run_evaluation(benchmark: Benchmark, agent: AgentProfile, evaluation: EvaluationResult):
    """Run the evaluation in background."""
    try:
        result = benchmark_runner.run_benchmark(benchmark, agent)
        # Update the evaluation with results
        evaluation.end_time = result.end_time
        evaluation.task_results = result.task_results
        evaluation.score = result.score

    except Exception as e:
        logger.exception(f"Error running evaluation: {str(e)}")
        evaluation.end_time = evaluation.start_time  # Mark as completed
        evaluation.score = 0
