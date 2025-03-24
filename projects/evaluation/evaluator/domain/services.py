# evaluator/domain/services.py
import logging
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from evaluator.domain.models import Benchmark, AgentProfile, EvaluationResult
from shared.protocol import TaskSpecification

logger = logging.getLogger(__name__)

class BenchmarkService:
    """Service for managing benchmarks."""
    
    def __init__(self):
        self.benchmarks: Dict[str, Benchmark] = {}
    
    def create_benchmark(self, name: str, description: str, 
                        tasks: List[Dict[str, Any]]) -> Benchmark:
        """Create a new benchmark."""
        # Convert task dictionaries to TaskSpecification objects
        task_specs = []
        for task_dict in tasks:
            task_id = str(uuid.uuid4())
            task_spec = TaskSpecification(
                id=task_id,
                name=task_dict.get("name", ""),
                description=task_dict.get("description", ""),
                steps=task_dict.get("steps", []),
                timeout=task_dict.get("timeout", 60),
                success_criteria=task_dict.get("success_criteria", {})
            )
            task_specs.append(task_spec)
        
        # Create the benchmark
        benchmark = Benchmark.create(name, description, task_specs)
        self.benchmarks[benchmark.id] = benchmark
        return benchmark
    
    def get_benchmark(self, benchmark_id: str) -> Optional[Benchmark]:
        """Get a benchmark by ID."""
        return self.benchmarks.get(benchmark_id)
    
    def list_benchmarks(self) -> List[Benchmark]:
        """List all benchmarks."""
        return list(self.benchmarks.values())
    
    def update_benchmark(self, benchmark_id: str, name: str, description: str, 
                        tasks: List[Dict[str, Any]]) -> Optional[Benchmark]:
        """Update an existing benchmark."""
        if benchmark_id not in self.benchmarks:
            return None
        
        # Convert task dictionaries to TaskSpecification objects
        task_specs = []
        for task_dict in tasks:
            task_id = task_dict.get("id", str(uuid.uuid4()))
            task_spec = TaskSpecification(
                id=task_id,
                name=task_dict.get("name", ""),
                description=task_dict.get("description", ""),
                steps=task_dict.get("steps", []),
                timeout=task_dict.get("timeout", 60),
                success_criteria=task_dict.get("success_criteria", {})
            )
            task_specs.append(task_spec)
        
        # Update the benchmark
        benchmark = self.benchmarks[benchmark_id]
        benchmark.name = name
        benchmark.description = description
        benchmark.tasks = task_specs
        return benchmark
    
    def delete_benchmark(self, benchmark_id: str) -> bool:
        """Delete a benchmark by ID."""
        if benchmark_id in self.benchmarks:
            del self.benchmarks[benchmark_id]
            return True
        return False

class AgentService:
    """Service for managing agent profiles."""
    
    def __init__(self):
        self.agents: Dict[str, AgentProfile] = {}
    
    def register_agent(self, name: str, endpoint: str) -> AgentProfile:
        """Register a new agent."""
        agent = AgentProfile.create(name, endpoint)
        self.agents[agent.id] = agent
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[AgentProfile]:
        """List all agents."""
        return list(self.agents.values())
    
    def update_agent(self, agent_id: str, name: str, endpoint: str) -> Optional[AgentProfile]: