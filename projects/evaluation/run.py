# Example script for creating a benchmark and running an evaluation
import json
import time

import requests

# Configuration
EVALUATOR_URL = "http://localhost:8001"
AGENT_URL = "http://localhost:8000"

# Register the agent
agent_data = {"name": "Desktop Agent v1", "endpoint": AGENT_URL}

response = requests.post(f"{EVALUATOR_URL}/agents", json=agent_data)
agent_id = response.json()["id"]
print(f"Registered agent with ID: {agent_id}")

# Create a benchmark
benchmark_data = {
    "name": "Web Booking Benchmark",
    "description": "Tests the agent's ability to book flights and fill forms",
    "tasks": [
        {
            "name": "book_flight",
            "description": "Book a flight from New York to Los Angeles",
            "steps": [
                {
                    "action": "navigate",
                    "params": {
                        "website": "https://demo.travel-website.com",
                        "from": "New York",
                        "to": "Los Angeles",
                        "departure_date": "2023-12-15",
                        "return_date": "2023-12-22",
                    },
                }
            ],
            "timeout": 60,
            "success_criteria": {"confirmation": True},
        },
        {
            "name": "search_web",
            "description": "Search for Python automation on Google",
            "steps": [
                {
                    "action": "search",
                    "params": {"search_engine": "https://www.google.com", "query": "python automation"},
                }
            ],
            "timeout": 30,
            "success_criteria": {"results_found": True},
        },
    ],
}

response = requests.post(f"{EVALUATOR_URL}/benchmarks", json=benchmark_data)
benchmark_id = response.json()["id"]
print(f"Created benchmark with ID: {benchmark_id}")

# Start the evaluation
eval_data = {"benchmark_id": benchmark_id, "agent_id": agent_id}

response = requests.post(f"{EVALUATOR_URL}/evaluations", json=eval_data)
print(f"Started evaluation: {response.json()}")

# Poll for results
while True:
    response = requests.get(f"{EVALUATOR_URL}/evaluations/{benchmark_id}/{agent_id}")
    data = response.json()

    if data.get("status") == "running":
        progress = data.get("progress", 0) * 100
        print(f"Evaluation in progress: {progress:.1f}% complete")
    else:
        print(f"Evaluation completed with score: {data.get('score')}%")
        print("\nTask Results:")
        for task in data.get("task_results", []):
            print(f"- Task {task['task_id']}: {'✅ Success' if task['success'] else '❌ Failed'}")
            if task.get("error_message"):
                print(f"  Error: {task['error_message']}")
        break

    time.sleep(5)  # Wait 5 seconds before checking again
