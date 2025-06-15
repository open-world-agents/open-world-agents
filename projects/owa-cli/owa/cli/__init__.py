"""
OWA CLI - Command-line interface for Open World Agents.

This module provides the main entry point for the OWA command-line tools,
including MCAP file management, video processing, and system utilities.
"""

import importlib
import platform
import shutil

# TODO: replace to https://github.com/BrianPugh/cyclopts
import typer
from loguru import logger

from . import env, mcap, messages, video
from .utils import check_for_update


def create_app() -> typer.Typer:
    """
    Create and configure the main CLI application.

    Returns:
        Configured Typer application
    """
    app = typer.Typer(name="owl", help="owl - Open World agents cLi - Tools for managing OWA data and environments")

    # Add core commands
    app.add_typer(mcap.app, name="mcap", help="MCAP file management commands")
    app.add_typer(env.app, name="env", help="Environment plugin management commands")
    app.add_typer(messages.app, name="messages", help="Message registry management commands")

    # Add optional commands based on available dependencies
    _add_optional_commands(app)

    return app


def _add_optional_commands(app: typer.Typer) -> None:
    """Add optional commands based on available dependencies."""

    # Video processing commands (requires FFmpeg)
    if _check_ffmpeg_available():
        app.add_typer(video.app, name="video", help="Video processing commands")
    else:
        logger.warning("FFmpeg not found. Video processing commands disabled.")

    # Window management commands (Windows only, requires owa.env.desktop)
    if _check_window_commands_available():
        from . import window

        app.add_typer(window.app, name="window", help="Window management commands")
    else:
        if platform.system() != "Windows":
            logger.debug("Window commands disabled: not running on Windows")
        elif not importlib.util.find_spec("owa.env.desktop"):
            logger.debug("Window commands disabled: owa.env.desktop not installed")


def _check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available."""
    return shutil.which("ffmpeg") is not None


def _check_window_commands_available() -> bool:
    """Check if window management commands are available."""
    return platform.system() == "Windows" and importlib.util.find_spec("owa.env.desktop") is not None


def main() -> None:
    """Main CLI entry point with global options."""
    check_for_update()


# Create the main application
app = create_app()

# Add global options to the main command
app.callback()(main)
