# evaluator/main.py
import logging
import os

import uvicorn
from evaluator.infrastructure.api import app

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main():
    # Start the server
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8001))  # Use a different port from the agent

    # Log startup information
    logging.info(f"Starting Evaluator API on {host}:{port}")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
