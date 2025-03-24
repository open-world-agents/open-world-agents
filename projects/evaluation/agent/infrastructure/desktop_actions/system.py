# agent/infrastructure/desktop_actions/system.py
import base64
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict

from agent.domain.services import TaskExecutionService
from PIL import ImageGrab
from shared.protocol import TaskSpecification

logger = logging.getLogger(__name__)


def setup_system_actions(service: TaskExecutionService):
    """Register system-related task handlers."""
    service.register_task_handler("create_file", create_file_handler)
    service.register_task_handler("find_file", find_file_handler)
    service.register_task_handler("system_info", system_info_handler)


def create_file_handler(task: TaskSpecification, service: TaskExecutionService) -> bool:
    """Task handler for creating a file."""
    service.log("Starting file creation task")

    # Extract parameters
    params = task.steps[0].get("params", {})
    file_path = params.get("path", os.path.join(tempfile.gettempdir(), "test_file.txt"))
    content = params.get("content", "This is a test file created by the agent.")

    try:
        # Create the directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            service.log(f"Creating directory: {directory}")
            os.makedirs(directory)

        # Create the file
        service.log(f"Creating file at: {file_path}")
        with open(file_path, "w") as file:
            file.write(content)

        # Take a screenshot
        screenshot = _take_desktop_screenshot()
        service.add_screenshot(screenshot)

        # Verify the file was created
        success = os.path.exists(file_path)

        if success:
            service.log(f"File created successfully at {file_path}")
        else:
            service.log("Failed to create file")

        return success

    except Exception as e:
        service.log(f"Error creating file: {str(e)}")
        return False


def find_file_handler(task: TaskSpecification, service: TaskExecutionService) -> bool:
    """Task handler for finding files."""
    service.log("Starting file search task")

    # Extract parameters
    params = task.steps[0].get("params", {})
    search_dir = params.get("directory", os.path.expanduser("~"))
    filename_pattern = params.get("pattern", "*.txt")

    try:
        service.log(f"Searching for {filename_pattern} in {search_dir}")

        # Use appropriate command based on platform
        if platform.system() == "Windows":
            command = f'dir /s /b "{search_dir}\\{filename_pattern}"'
            shell = True
        else:  # Linux/Mac
            command = ["find", search_dir, "-name", filename_pattern]
            shell = False

        # Execute search command
        process = subprocess.Popen(command, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        stdout, stderr = process.communicate()

        # Log the results
        if stdout:
            service.log(f"Found files:\n{stdout}")

        if stderr:
            service.log(f"Search errors:\n{stderr}")

        # Take a screenshot
        screenshot = _take_desktop_screenshot()
        service.add_screenshot(screenshot)

        # Check if any files were found
        success = len(stdout.strip()) > 0

        if success:
            service.log("File search completed successfully")
        else:
            service.log("No matching files found")

        return success

    except Exception as e:
        service.log(f"Error during file search: {str(e)}")
        return False


def system_info_handler(task: TaskSpecification, service: TaskExecutionService) -> bool:
    """Task handler for gathering system information."""
    service.log("Starting system information task")

    try:
        # Collect system information
        system_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
        }

        # Log system information
        for key, value in system_info.items():
            service.log(f"{key}: {value}")

        # Get disk usage
        if platform.system() == "Windows":
            # Windows
            temp_dir = os.environ.get("TEMP")
            disk_usage = shutil.disk_usage(temp_dir)
        else:
            # Linux/Mac
            disk_usage = shutil.disk_usage("/")

        total_gb = disk_usage.total / (1024**3)
        used_gb = disk_usage.used / (1024**3)
        free_gb = disk_usage.free / (1024**3)

        service.log(f"Disk Total: {total_gb:.2f} GB")
        service.log(f"Disk Used: {used_gb:.2f} GB")
        service.log(f"Disk Free: {free_gb:.2f} GB")

        # Take a screenshot
        screenshot = _take_desktop_screenshot()
        service.add_screenshot(screenshot)

        # Always succeed unless there's an exception
        service.log("System information task completed successfully")
        return True

    except Exception as e:
        service.log(f"Error gathering system information: {str(e)}")
        return False


def _take_desktop_screenshot() -> str:
    """Take a screenshot of the desktop and return as base64 string."""
    screenshot = ImageGrab.grab()

    # Convert PIL Image to base64
    import io

    buffer = io.BytesIO()
    screenshot.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
