"""
OWA CLI - Command-line interface for Open World Agents.

This module provides the main entry point for the OWA command-line tools,
including MCAP file management, video processing, and system utilities.
"""

import importlib
import importlib.util
import platform
import shutil

# TODO?: replace to https://github.com/BrianPugh/cyclopts
import typer
from loguru import logger

from . import env, mcap, messages, video
from .console import console
from .utils import check_for_update

# Disable logger by default for library usage (following loguru best practices)
# Reference: https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("owa.cli")

# Store warnings to show later
_dependency_warnings = []


def create_app(**kwargs) -> typer.Typer:
    """
    Create and configure the main CLI application.

    Args:
        **kwargs: Additional arguments passed to typer.Typer

    Returns:
        Configured Typer application
    """
    app = typer.Typer(
        name="owl", help="owl - Open World agents cLi - Tools for managing OWA data and environments", **kwargs
    )

    # Add core commands
    app.add_typer(mcap.app, name="mcap", help="MCAP file management commands")
    app.add_typer(env.app, name="env", help="Environment plugin management commands")
    app.add_typer(messages.app, name="messages", help="Message registry management commands")

    # Add optional commands
    _add_optional_commands(app)

    return app


def _add_optional_commands(app: typer.Typer) -> None:
    """Add optional commands based on available dependencies."""

    # Video processing commands (requires FFmpeg)
    if _check_ffmpeg_available():
        app.add_typer(video.app, name="video", help="Video processing commands")
    else:
        _dependency_warnings.append("[yellow]⚠ FFmpeg not found. Video processing commands disabled.[/yellow]")

    # Window management commands (Windows only, requires owa.env.desktop)
    if _check_window_commands_available():
        from . import window

        app.add_typer(window.app, name="window", help="Window management commands")
    else:
        if platform.system() != "Windows":
            _dependency_warnings.append("[dim]ℹ Window commands disabled: not running on Windows[/dim]")
        elif not importlib.util.find_spec("owa.env.desktop"):
            _dependency_warnings.append("[dim]ℹ Window commands disabled: owa.env.desktop not installed[/dim]")


def _check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available."""
    return shutil.which("ffmpeg") is not None


def _check_window_commands_available() -> bool:
    """Check if window management commands are available."""
    return platform.system() == "Windows" and importlib.util.find_spec("owa.env.desktop") is not None


def callback(
    silent: bool = typer.Option(False, "--silent", "-s", help="Suppress non-essential output"),
    no_update_check: bool = typer.Option(False, "--no-update-check", help="Skip version update check"),
) -> None:
    """Main CLI entry point with global options."""
    # Show dependency warnings (unless silent)
    if not silent:
        _show_dependency_warnings()

    # Check for updates unless disabled
    if not no_update_check:
        check_for_update(silent=silent)


def _show_dependency_warnings() -> None:
    """Show warnings about missing optional dependencies."""
    for warning in _dependency_warnings:
        console.print(warning)


# Create the main application with callback
app = create_app(callback=callback)
