"""
Data models for OEP-0004 documentation system.

These models define the structure of extracted documentation from plugin source code.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ParameterDocumentation:
    """Documentation for a function/method parameter."""
    
    name: str
    type: Optional[str]
    default: Optional[str]
    description: str
    is_optional: bool


@dataclass
class ReturnDocumentation:
    """Documentation for a function/method return value."""
    
    type: Optional[str]
    description: str


@dataclass
class ExceptionDocumentation:
    """Documentation for an exception that may be raised."""
    
    exception_type: str
    description: str


@dataclass
class ExampleDocumentation:
    """Documentation for a code example."""
    
    code: str
    description: Optional[str]
    expected_output: Optional[str]
    is_doctest: bool


@dataclass
class AttributeDocumentation:
    """Documentation for a class attribute."""
    
    name: str
    type: Optional[str]
    description: str


@dataclass
class ComponentDocumentation:
    """Complete documentation for a single component."""
    
    name: str
    full_name: str  # namespace/name
    type: str  # callables, listeners, runnables
    source_file: str
    line_number: int

    # Extracted from docstring
    summary: str
    description: str
    parameters: List[ParameterDocumentation]
    returns: Optional[ReturnDocumentation]
    raises: List[ExceptionDocumentation]
    examples: List[ExampleDocumentation]
    notes: List[str]

    # Extracted from signature analysis
    signature: str
    type_hints: Dict[str, str]
    is_async: bool
    is_method: bool
    is_classmethod: bool
    is_staticmethod: bool

    # For classes
    methods: List['ComponentDocumentation']
    attributes: List[AttributeDocumentation]
    inheritance: List[str]


@dataclass
class PluginDocumentation:
    """Complete documentation for a plugin."""
    
    namespace: str
    version: str  # From PluginSpec
    description: str  # From PluginSpec
    author: str  # From PluginSpec
    components: Dict[str, List[ComponentDocumentation]]
    generated_at: datetime
    source_files: List[str]  # Source files analyzed
