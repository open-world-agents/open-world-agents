"""
Shared console instance for OWA CLI.

This module provides a centralized rich console instance that should be used
across all owa-cli modules for consistent output formatting and styling.
"""

import os

from rich.console import Console


def _create_console() -> Console:
    """Create a console instance with appropriate settings."""
    disable_console_styling = os.environ.get("OWA_DISABLE_CONSOLE_STYLING")
    if disable_console_styling:
        # Disable all styling features. Reference: https://rich.readthedocs.io/en/latest/console.html
        environ = os.environ.copy()
        environ["NO_COLOR"] = "1"
        environ["TERM"] = "dumb"
        environ["TTY_COMPATIBLE"] = "0"
        environ["TTY_INTERACTIVE"] = "0"
        return Console(_environ=environ)
    else:
        # Normal console with default settings
        raise NotImplementedError("Console styling is not yet implemented")


# Shared console instance for all CLI output
console = _create_console()

__all__ = ["console"]
