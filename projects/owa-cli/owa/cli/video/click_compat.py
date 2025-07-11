"""
Click compatibility module for mkdocs-click integration.

This module exposes the underlying Click command from the Typer application
so that mkdocs-click can generate documentation for it.
"""

import typer.main
from . import app

# Convert the Typer app to a Click command for mkdocs-click
click_command = typer.main.get_command(app)
