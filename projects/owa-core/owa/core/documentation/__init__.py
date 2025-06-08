"""
Documentation validation and generation system for OWA plugins.

This module implements OEP-0004 functionality for validating plugin documentation
and providing mkdocstrings integration.
"""

from .validator import DocumentationValidator, ValidationResult

__all__ = [
    "DocumentationValidator",
    "ValidationResult",
]
