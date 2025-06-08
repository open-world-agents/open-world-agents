"""
OEP-0004: Automatic EnvPlugin Documentation Generation and Display

This module provides automatic documentation generation for EnvPlugins that extracts
and aggregates docstrings directly from component source code into comprehensive
plugin documentation, similar to mkdocstrings.
"""

from .generator import PluginDocumentationGenerator
from .inspector import SignatureInspector
from .models import (
    AttributeDocumentation,
    ComponentDocumentation,
    ExampleDocumentation,
    ExceptionDocumentation,
    ParameterDocumentation,
    PluginDocumentation,
    ReturnDocumentation,
)
from .parser import DocstringParser
from .templates import TemplateEngine

__all__ = [
    # Main generator
    "PluginDocumentationGenerator",
    # Core components
    "DocstringParser",
    "SignatureInspector",
    "TemplateEngine",
    # Data models
    "PluginDocumentation",
    "ComponentDocumentation",
    "ParameterDocumentation",
    "ReturnDocumentation",
    "ExceptionDocumentation",
    "ExampleDocumentation",
    "AttributeDocumentation",
]
