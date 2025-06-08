"""
Docstring parsing for OEP-0004 documentation system.

Supports multiple docstring formats: Google, NumPy, and Sphinx styles.
"""

import re
from typing import Dict, List, Optional, Tuple

from .models import (
    ExampleDocumentation,
    ExceptionDocumentation,
    ParameterDocumentation,
    ReturnDocumentation,
)


class DocstringInfo:
    """Parsed docstring information."""
    
    def __init__(self):
        self.summary: str = ""
        self.description: str = ""
        self.parameters: List[ParameterDocumentation] = []
        self.returns: Optional[ReturnDocumentation] = None
        self.raises: List[ExceptionDocumentation] = []
        self.examples: List[ExampleDocumentation] = []
        self.notes: List[str] = []


class DocstringParser:
    """
    Multi-format docstring parser supporting Google, NumPy, and Sphinx styles.
    
    This parser extracts structured information from docstrings automatically,
    similar to how mkdocstrings processes documentation.
    """
    
    def parse(self, docstring: str) -> DocstringInfo:
        """
        Parse a docstring and extract structured information.
        
        Args:
            docstring: The docstring to parse
            
        Returns:
            Parsed docstring information
        """
        if not docstring:
            return DocstringInfo()
            
        # Clean and normalize the docstring
        cleaned = self._clean_docstring(docstring)
        
        # Try different parsing strategies
        info = self._parse_google_style(cleaned)
        if not info.summary:
            info = self._parse_numpy_style(cleaned)
        if not info.summary:
            info = self._parse_sphinx_style(cleaned)
        if not info.summary:
            info = self._parse_plain_text(cleaned)
            
        return info
    
    def _clean_docstring(self, docstring: str) -> str:
        """Clean and normalize a docstring."""
        lines = docstring.strip().split('\n')
        
        # Remove common leading whitespace
        if lines:
            # Find minimum indentation (excluding empty lines)
            non_empty_lines = [line for line in lines[1:] if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                lines = [lines[0]] + [line[min_indent:] if len(line) > min_indent else line 
                                     for line in lines[1:]]
        
        return '\n'.join(lines)
    
    def _parse_google_style(self, docstring: str) -> DocstringInfo:
        """Parse Google-style docstring."""
        info = DocstringInfo()
        lines = docstring.split('\n')
        
        # Extract summary and description
        summary_lines = []
        desc_lines = []
        current_section = None
        i = 0
        
        # Get summary (first non-empty line)
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i < len(lines):
            info.summary = lines[i].strip()
            i += 1
            
        # Skip empty line after summary
        while i < len(lines) and not lines[i].strip():
            i += 1
            
        # Get description until we hit a section
        while i < len(lines):
            line = lines[i].strip()
            if line.endswith(':') and line[:-1] in ['Args', 'Arguments', 'Parameters', 'Returns', 'Return', 'Yields', 'Yield', 'Raises', 'Raise', 'Examples', 'Example', 'Note', 'Notes']:
                break
            if line:
                desc_lines.append(line)
            i += 1
            
        info.description = ' '.join(desc_lines)
        
        # Parse sections
        while i < len(lines):
            line = lines[i].strip()
            if line.endswith(':'):
                section_name = line[:-1].lower()
                i += 1
                
                if section_name in ['args', 'arguments', 'parameters']:
                    i = self._parse_google_parameters(lines, i, info)
                elif section_name in ['returns', 'return']:
                    i = self._parse_google_returns(lines, i, info)
                elif section_name in ['raises', 'raise']:
                    i = self._parse_google_raises(lines, i, info)
                elif section_name in ['examples', 'example']:
                    i = self._parse_google_examples(lines, i, info)
                elif section_name in ['note', 'notes']:
                    i = self._parse_google_notes(lines, i, info)
                else:
                    i += 1
            else:
                i += 1
                
        return info
    
    def _parse_google_parameters(self, lines: List[str], start: int, info: DocstringInfo) -> int:
        """Parse Google-style parameters section."""
        i = start
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line.endswith(':') and not line.startswith(' '):
                break  # Next section
                
            # Parse parameter line: "param_name (type): description"
            match = re.match(r'^\s*(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.+)', line)
            if match:
                name, param_type, desc = match.groups()
                
                # Check for default value
                default = None
                is_optional = False
                if param_type:
                    if 'optional' in param_type.lower():
                        is_optional = True
                    if 'default' in param_type.lower():
                        default_match = re.search(r'default[=\s]+([^,)]+)', param_type.lower())
                        if default_match:
                            default = default_match.group(1).strip()
                
                info.parameters.append(ParameterDocumentation(
                    name=name,
                    type=param_type,
                    default=default,
                    description=desc.strip(),
                    is_optional=is_optional
                ))
            i += 1
            
        return i
    
    def _parse_google_returns(self, lines: List[str], start: int, info: DocstringInfo) -> int:
        """Parse Google-style returns section."""
        i = start
        return_lines = []
        return_type = None
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line.endswith(':') and not line.startswith(' '):
                break  # Next section
                
            # Check if line starts with type
            type_match = re.match(r'^\s*([^:]+):\s*(.+)', line)
            if type_match and not return_type:
                return_type, desc = type_match.groups()
                return_lines.append(desc.strip())
            else:
                return_lines.append(line)
            i += 1
            
        if return_lines:
            info.returns = ReturnDocumentation(
                type=return_type,
                description=' '.join(return_lines)
            )
            
        return i
    
    def _parse_google_raises(self, lines: List[str], start: int, info: DocstringInfo) -> int:
        """Parse Google-style raises section."""
        i = start
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line.endswith(':') and not line.startswith(' '):
                break  # Next section
                
            # Parse exception line: "ExceptionType: description"
            match = re.match(r'^\s*(\w+)\s*:\s*(.+)', line)
            if match:
                exc_type, desc = match.groups()
                info.raises.append(ExceptionDocumentation(
                    exception_type=exc_type,
                    description=desc.strip()
                ))
            i += 1
            
        return i
    
    def _parse_google_examples(self, lines: List[str], start: int, info: DocstringInfo) -> int:
        """Parse Google-style examples section."""
        i = start
        example_lines = []
        
        while i < len(lines):
            line = lines[i]
            if line.strip().endswith(':') and not line.startswith(' '):
                break  # Next section
                
            example_lines.append(line)
            i += 1
            
        if example_lines:
            # Join all example lines
            example_text = '\n'.join(example_lines).strip()
            
            # Split by >>> to find individual examples
            examples = re.split(r'\n(?=>>>)', example_text)
            for example in examples:
                if example.strip():
                    is_doctest = '>>>' in example
                    info.examples.append(ExampleDocumentation(
                        code=example.strip(),
                        description=None,
                        expected_output=None,
                        is_doctest=is_doctest
                    ))
                    
        return i
    
    def _parse_google_notes(self, lines: List[str], start: int, info: DocstringInfo) -> int:
        """Parse Google-style notes section."""
        i = start
        note_lines = []
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line.endswith(':') and not line.startswith(' '):
                break  # Next section
                
            note_lines.append(line)
            i += 1
            
        if note_lines:
            info.notes.extend(note_lines)
            
        return i
    
    def _parse_numpy_style(self, docstring: str) -> DocstringInfo:
        """Parse NumPy-style docstring."""
        # Simplified NumPy parsing - can be expanded
        info = DocstringInfo()
        lines = docstring.split('\n')
        
        if lines:
            info.summary = lines[0].strip()
            
        # Look for Parameters section
        for i, line in enumerate(lines):
            if line.strip() in ['Parameters', 'Parameters\n----------']:
                # Found parameters section
                break
                
        return info
    
    def _parse_sphinx_style(self, docstring: str) -> DocstringInfo:
        """Parse Sphinx-style docstring."""
        # Simplified Sphinx parsing - can be expanded
        info = DocstringInfo()
        lines = docstring.split('\n')
        
        if lines:
            info.summary = lines[0].strip()
            
        return info
    
    def _parse_plain_text(self, docstring: str) -> DocstringInfo:
        """Parse plain text docstring."""
        info = DocstringInfo()
        lines = docstring.split('\n')
        
        if lines:
            info.summary = lines[0].strip()
            if len(lines) > 1:
                info.description = ' '.join(line.strip() for line in lines[1:] if line.strip())
                
        return info
