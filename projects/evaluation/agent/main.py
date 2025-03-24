# agent/main.py
import logging

import uvicorn
from agent.application.task_executor import task_manager
from agent.infrastructure.api import app
from agent.infrastructure.desktop_actions.browser import setup_browser_actions
from agent.infrastructure.desktop_actions.system import setup_system_actions

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main():
    # Register task handlers
    setup_browser_actions(task_manager.service)
    setup_system_actions(task_manager.service)

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
