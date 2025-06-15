"""
Message management commands for OWA CLI.

This module provides commands for browsing, validating, and managing
OWA message types through the message registry system.
"""

import typer

from .list import list_messages
from .show import show_message
from .validate import validate_messages

# Create the messages command group
app = typer.Typer(help="Message registry management commands")

# Add subcommands
app.command("list", help="List all available message types")(list_messages)
app.command("show", help="Show details about a specific message type")(show_message)
app.command("validate", help="Validate message registry and definitions")(validate_messages)

__all__ = ["app"]
