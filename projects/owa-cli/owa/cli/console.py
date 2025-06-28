"""
Shared console instance for OWA CLI.

This module provides a centralized rich console instance that should be used
across all owa-cli modules for consistent output formatting and styling.
"""

from rich.console import Console

# Shared console instance for all CLI output
console = Console()

__all__ = ["console"]
