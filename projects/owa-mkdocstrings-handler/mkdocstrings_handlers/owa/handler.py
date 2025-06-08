"""
Custom mkdocstrings handler for OWA EnvPlugin components.

This module implements a minimal mkdocstrings handler that understands
OWA's plugin structure and can generate documentation automatically.
"""

import inspect
import re
from typing import Any, Dict, List, Optional, Union

try:
    import griffe
    from mkdocstrings import BaseHandler

    MKDOCSTRINGS_AVAILABLE = True
    GRIFFE_AVAILABLE = True
except ImportError:
    # mkdocstrings not available, create a dummy base class
    MKDOCSTRINGS_AVAILABLE = False
    GRIFFE_AVAILABLE = False

    try:
        import griffe

        GRIFFE_AVAILABLE = True
    except ImportError:
        GRIFFE_AVAILABLE = False

    class BaseHandler:
        """Dummy base handler when mkdocstrings is not available."""

        name = "owa"
        domain = "py"
        fallback_theme = "material"

        def __init__(self, **kwargs):
            pass


from owa.core.plugin_discovery import get_plugin_discovery
from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES


class OWAHandler(BaseHandler):
    """
    Minimal mkdocstrings handler for OWA EnvPlugin components.

    This handler enables automatic documentation generation for OWA plugins
    using the familiar mkdocstrings syntax.
    """

    name = "owa"
    domain = "py"
    fallback_theme = "material"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plugin_discovery = get_plugin_discovery()

        # Store global options from kwargs (passed by mkdocstrings)
        self.global_options = kwargs.get("options", {})

    def get_options(self, local_options: dict) -> dict:
        """
        Build options for the handler by merging global and local options.

        This method is required for mkdocstrings v1 compatibility.

        Args:
            local_options: Local options from the specific documentation block

        Returns:
            Merged options dictionary
        """
        # Start with default options for OWA handler
        default_options = {
            "show_source": True,
            "show_signature": True,
            "show_examples": True,
            "show_griffe_info": True,
            "truncate_source": 2000,
        }

        # Merge: defaults -> global -> local (local takes highest precedence)
        options = {**default_options, **self.global_options, **local_options}

        return options

    def collect(self, identifier: str, options: dict) -> dict:
        """
        Collect documentation data for an EnvPlugin component.

        Args:
            identifier: Plugin namespace or namespace/component
            options: Collection options

        Returns:
            Documentation data dictionary
        """
        if not MKDOCSTRINGS_AVAILABLE:
            return {"error": "mkdocstrings not available"}

        try:
            if "/" in identifier:
                namespace, component = identifier.split("/", 1)
                return self._collect_component(namespace, component)
            else:
                return self._collect_plugin(identifier)
        except Exception as e:
            return {"error": f"Failed to collect documentation: {e}"}

    def _collect_plugin(self, namespace: str) -> dict:
        """Collect plugin-level documentation."""
        plugin_spec = self.plugin_discovery.discovered_plugins.get(namespace)
        if not plugin_spec:
            return {"error": f"Plugin '{namespace}' not found"}

        # Extract component information
        components = {}
        for comp_type, comp_dict in plugin_spec.components.items():
            components[comp_type] = []
            for comp_name, import_path in comp_dict.items():
                full_name = f"{namespace}/{comp_name}"
                try:
                    component = self._load_component(comp_type, full_name)
                    comp_info = self._analyze_component(component, comp_name, full_name, import_path)
                    components[comp_type].append(comp_info)
                except Exception as e:
                    comp_info = {
                        "name": comp_name,
                        "full_name": full_name,
                        "import_path": import_path,
                        "error": str(e),
                    }
                    components[comp_type].append(comp_info)

        return {
            "type": "plugin",
            "namespace": namespace,
            "version": plugin_spec.version,
            "description": plugin_spec.description,
            "author": plugin_spec.author,
            "components": components,
        }

    def _collect_component(self, namespace: str, component_name: str) -> dict:
        """Collect component-level documentation."""
        # Find the component in any type
        full_name = f"{namespace}/{component_name}"
        component = None
        comp_type_found = None
        import_path = None

        # Check plugin spec for component info
        plugin_spec = self.plugin_discovery.discovered_plugins.get(namespace)
        if not plugin_spec:
            return {"error": f"Plugin '{namespace}' not found"}

        # Find component in plugin spec
        for comp_type, comp_dict in plugin_spec.components.items():
            if component_name in comp_dict:
                comp_type_found = comp_type
                import_path = comp_dict[component_name]
                try:
                    component = self._load_component(comp_type, full_name)
                    break
                except Exception as e:
                    return {"error": f"Failed to load component: {e}"}

        if component is None:
            return {"error": f"Component '{full_name}' not found"}

        # Get comprehensive component analysis
        comp_info = self._analyze_component(component, component_name, full_name, import_path)

        # Add component-specific metadata
        comp_info.update(
            {
                "type": "component",
                "namespace": namespace,
                "component_type": comp_type_found,
            }
        )

        return comp_info

    def _load_component(self, component_type: str, full_name: str) -> Any:
        """Load a component from the appropriate registry."""
        if component_type == "callables":
            return CALLABLES[full_name]
        elif component_type == "listeners":
            return LISTENERS[full_name]
        elif component_type == "runnables":
            return RUNNABLES[full_name]
        else:
            raise ValueError(f"Unknown component type: {component_type}")

    def _analyze_component(self, component: Any, name: str, full_name: str, import_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a component using Griffe when available.

        Args:
            component: The component object to analyze
            name: Component name
            full_name: Full component name (namespace/name)
            import_path: Import path for the component

        Returns:
            Dictionary with comprehensive component information
        """
        # Try to use Griffe for enhanced analysis
        griffe_data = self._analyze_with_griffe(import_path) if GRIFFE_AVAILABLE else None

        # Fallback to inspect-based analysis
        docstring = inspect.getdoc(component)
        signature_info = self._get_detailed_signature(component)
        parsed_doc = self._parse_docstring(docstring) if docstring else {}

        # Merge Griffe data if available
        if griffe_data:
            # Use Griffe's superior docstring parsing
            if griffe_data.get("docstring_parsed"):
                parsed_doc = griffe_data["docstring_parsed"]

            # Use Griffe's parameter information if available
            if griffe_data.get("parameters"):
                signature_info["griffe_parameters"] = griffe_data["parameters"]

            # Add Griffe-specific information
            signature_info["griffe_info"] = griffe_data

        return {
            "name": name,
            "full_name": full_name,
            "import_path": import_path,
            "docstring": docstring,
            "signature": signature_info.get("signature_str"),
            "signature_info": signature_info,
            "parsed_docstring": parsed_doc,
            "source": self._get_source(component),
            "component_class": self._get_component_class_info(component),
            "usage_examples": self._generate_usage_examples(full_name, signature_info, parsed_doc),
            "griffe_data": griffe_data,
        }

    def _analyze_with_griffe(self, import_path: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a component using Griffe for enhanced documentation extraction.

        Args:
            import_path: Import path like "module.path:function_name"

        Returns:
            Dictionary with Griffe analysis data or None if analysis fails
        """
        if not GRIFFE_AVAILABLE:
            return None

        try:
            # Parse the import path
            if ":" not in import_path:
                return None

            module_path, obj_name = import_path.split(":", 1)
            full_import_path = f"{module_path}.{obj_name}"

            # Load with Griffe
            obj = griffe.load(full_import_path)

            # Extract comprehensive information
            griffe_info = {
                "kind": obj.kind.value if hasattr(obj.kind, "value") else str(obj.kind),
                "name": obj.name,
                "lineno": getattr(obj, "lineno", None),
                "endlineno": getattr(obj, "endlineno", None),
                "module_path": module_path,
                "qualname": getattr(obj, "canonical_path", str(obj)),
            }

            # Extract docstring information
            if hasattr(obj, "docstring") and obj.docstring:
                griffe_info["docstring_raw"] = obj.docstring.value
                griffe_info["docstring_parsed"] = self._parse_griffe_docstring(obj.docstring)

            # Extract parameters for functions/methods
            if hasattr(obj, "parameters") and obj.parameters:
                griffe_info["parameters"] = self._extract_griffe_parameters(obj.parameters)

            # Extract return annotation
            if hasattr(obj, "returns") and obj.returns:
                griffe_info["returns"] = self._format_griffe_annotation(obj.returns)

            # Extract decorators
            if hasattr(obj, "decorators") and obj.decorators:
                griffe_info["decorators"] = [str(dec) for dec in obj.decorators]

            # Extract labels/tags
            if hasattr(obj, "labels") and obj.labels:
                griffe_info["labels"] = list(obj.labels)

            return griffe_info

        except Exception as e:
            # Griffe analysis failed, return error info for debugging
            return {"error": f"Griffe analysis failed: {e}", "import_path": import_path}

    def _parse_griffe_docstring(self, docstring) -> Dict[str, Any]:
        """
        Parse a Griffe docstring object into structured information.

        Args:
            docstring: Griffe docstring object

        Returns:
            Dictionary with parsed docstring information
        """
        if not docstring:
            return {}

        parsed = {
            "summary": "",
            "description": "",
            "args": [],
            "returns": "",
            "raises": [],
            "examples": [],
            "notes": [],
        }

        # Extract summary and description
        if hasattr(docstring, "value"):
            lines = docstring.value.strip().split("\n")
            if lines:
                parsed["summary"] = lines[0].strip()
                if len(lines) > 1:
                    parsed["description"] = "\n".join(lines[1:]).strip()

        # Extract structured sections if Griffe parsed them
        if hasattr(docstring, "sections"):
            for section in docstring.sections:
                section_kind = getattr(section, "kind", None)
                if hasattr(section_kind, "value"):
                    section_kind = section_kind.value
                else:
                    section_kind = str(section_kind)

                if section_kind == "parameters":
                    parsed["args"] = self._extract_griffe_docstring_parameters(section)
                elif section_kind == "returns":
                    parsed["returns"] = getattr(section, "value", "")
                elif section_kind == "raises":
                    parsed["raises"] = self._extract_griffe_docstring_raises(section)
                elif section_kind == "examples":
                    parsed["examples"] = self._extract_griffe_docstring_examples(section)

        return parsed

    def _extract_griffe_parameters(self, parameters) -> List[Dict[str, Any]]:
        """Extract parameter information from Griffe parameters."""
        param_list = []

        for param in parameters:
            param_info = {
                "name": param.name,
                "kind": param.kind.value if hasattr(param.kind, "value") else str(param.kind),
                "annotation": self._format_griffe_annotation(param.annotation) if param.annotation else None,
                "default": str(param.default) if param.default else None,
                "has_default": param.default is not None,
            }
            param_list.append(param_info)

        return param_list

    def _format_griffe_annotation(self, annotation) -> str:
        """Format a Griffe annotation for display."""
        if not annotation:
            return ""

        # Handle different annotation types
        if hasattr(annotation, "name"):
            return annotation.name
        elif hasattr(annotation, "canonical_path"):
            return str(annotation.canonical_path)
        else:
            return str(annotation)

    def _extract_griffe_docstring_parameters(self, section) -> List[Dict[str, str]]:
        """Extract parameters from Griffe docstring parameters section."""
        params = []
        if hasattr(section, "value") and section.value:
            for param in section.value:
                param_info = {
                    "name": getattr(param, "name", ""),
                    "description": getattr(param, "description", ""),
                }
                params.append(param_info)
        return params

    def _extract_griffe_docstring_raises(self, section) -> List[Dict[str, str]]:
        """Extract raises from Griffe docstring raises section."""
        raises = []
        if hasattr(section, "value") and section.value:
            for exc in section.value:
                exc_info = {
                    "exception": getattr(exc, "annotation", getattr(exc, "name", "")),
                    "description": getattr(exc, "description", ""),
                }
                raises.append(exc_info)
        return raises

    def _extract_griffe_docstring_examples(self, section) -> List[str]:
        """Extract examples from Griffe docstring examples section."""
        examples = []
        if hasattr(section, "value"):
            if isinstance(section.value, str):
                examples.append(section.value)
            elif hasattr(section.value, "__iter__"):
                for example in section.value:
                    examples.append(str(example))
        return examples

    def _get_signature(self, component: Any) -> Optional[str]:
        """Get the signature of a component."""
        try:
            if inspect.isclass(component):
                # For classes, get __init__ signature
                if hasattr(component, "__init__"):
                    sig = inspect.signature(component.__init__)
                    return str(sig)
            else:
                sig = inspect.signature(component)
                return str(sig)
        except (ValueError, TypeError):
            pass
        return None

    def _get_detailed_signature(self, component: Any) -> Dict[str, Any]:
        """
        Get detailed signature information including parameters, types, and defaults.

        Args:
            component: The component to analyze

        Returns:
            Dictionary with detailed signature information
        """
        try:
            # Determine what to inspect
            if inspect.isclass(component):
                if hasattr(component, "__call__") and not hasattr(component, "__init__"):
                    # Callable class with __call__ method
                    sig = inspect.signature(component.__call__)
                    target_func = component.__call__
                elif hasattr(component, "__init__"):
                    # Regular class, inspect __init__
                    sig = inspect.signature(component.__init__)
                    target_func = component.__init__
                else:
                    return {"signature_str": None, "parameters": [], "return_annotation": None}
            else:
                # Function or callable object
                sig = inspect.signature(component)
                target_func = component

            # Extract parameter information
            parameters = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue  # Skip self parameter

                param_info = {
                    "name": param_name,
                    "annotation": self._format_annotation(param.annotation),
                    "default": self._format_default(param.default),
                    "kind": param.kind.name,
                    "has_default": param.default != inspect.Parameter.empty,
                }
                parameters.append(param_info)

            return {
                "signature_str": str(sig),
                "parameters": parameters,
                "return_annotation": self._format_annotation(sig.return_annotation),
                "is_class": inspect.isclass(component),
                "is_method": inspect.ismethod(target_func),
                "is_function": inspect.isfunction(target_func),
            }
        except (ValueError, TypeError) as e:
            return {
                "signature_str": None,
                "parameters": [],
                "return_annotation": None,
                "error": str(e),
            }

    def _format_annotation(self, annotation) -> Optional[str]:
        """Format a type annotation for display."""
        if annotation == inspect.Parameter.empty or annotation == inspect.Signature.empty:
            return None

        # Handle common type annotations
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        elif hasattr(annotation, "__module__") and hasattr(annotation, "__qualname__"):
            if annotation.__module__ == "builtins":
                return annotation.__qualname__
            else:
                return f"{annotation.__module__}.{annotation.__qualname__}"
        else:
            return str(annotation)

    def _format_default(self, default) -> Optional[str]:
        """Format a default value for display."""
        if default == inspect.Parameter.empty:
            return None

        if isinstance(default, str):
            return f'"{default}"'
        elif default is None:
            return "None"
        else:
            return str(default)

    def _get_source(self, component: Any) -> Optional[str]:
        """Get the source code of a component."""
        try:
            return inspect.getsource(component)
        except (OSError, TypeError):
            return None

    def _get_component_class_info(self, component: Any) -> Dict[str, Any]:
        """
        Get information about the component's class/type.

        Args:
            component: The component to analyze

        Returns:
            Dictionary with class information
        """
        info = {
            "is_class": inspect.isclass(component),
            "is_function": inspect.isfunction(component),
            "is_method": inspect.ismethod(component),
            "is_callable": callable(component),
            "module": getattr(component, "__module__", None),
            "qualname": getattr(component, "__qualname__", None),
        }

        if inspect.isclass(component):
            # Get base classes
            info["bases"] = [base.__name__ for base in component.__bases__ if base is not object]
            info["mro"] = [cls.__name__ for cls in component.__mro__[1:]]  # Skip self

        return info

    def _generate_usage_examples(
        self, full_name: str, signature_info: Dict[str, Any], parsed_doc: Dict[str, Any]
    ) -> List[str]:
        """
        Generate usage examples for the component.

        Args:
            full_name: Full component name (namespace/name)
            signature_info: Signature information
            parsed_doc: Parsed docstring information

        Returns:
            List of usage example strings
        """
        examples = []

        # Extract examples from docstring if available
        if parsed_doc.get("examples"):
            examples.extend(parsed_doc["examples"])

        # Generate basic usage example based on component type
        if full_name.split("/")[0] in ["std", "desktop", "gst", "example"]:
            namespace = full_name.split("/")[0]

            # Generate registry access example
            if signature_info.get("parameters"):
                # Has parameters - show with example values
                param_examples = []
                for param in signature_info["parameters"]:
                    if param["annotation"] == "str":
                        param_examples.append(f'"{param["name"]}_value"')
                    elif param["annotation"] in ["int", "float"]:
                        param_examples.append("42" if param["annotation"] == "int" else "3.14")
                    elif param["annotation"] == "bool":
                        param_examples.append("True")
                    elif param["has_default"]:
                        continue  # Skip parameters with defaults
                    else:
                        param_examples.append(f"{param['name']}")

                if param_examples:
                    examples.append(f'CALLABLES["{full_name}"]({", ".join(param_examples)})')
                else:
                    examples.append(f'CALLABLES["{full_name}"]()')
            else:
                # No parameters
                examples.append(f'CALLABLES["{full_name}"]()')

            # Add get_component example
            name_part = full_name.split("/", 1)[1]
            examples.append(f'get_component("callables", namespace="{namespace}", name="{name_part}")')

        return examples

    def _parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """Parse a docstring into structured components with enhanced parsing."""
        if not docstring:
            return {}

        lines = docstring.strip().split("\n")
        parsed = {
            "summary": "",
            "description": "",
            "args": [],
            "returns": "",
            "raises": [],
            "examples": [],
            "notes": [],
        }

        # Extract summary (first line)
        if lines:
            parsed["summary"] = lines[0].strip()

        # Enhanced parsing for Google/Sphinx style docstrings
        current_section = "description"
        current_content = []

        for line in lines[1:]:
            stripped = line.strip()

            # Check for section headers
            if stripped.lower().startswith(("args:", "arguments:", "parameters:")):
                self._finalize_section(parsed, current_section, current_content)
                current_section = "args"
                current_content = []
            elif stripped.lower().startswith("returns:"):
                self._finalize_section(parsed, current_section, current_content)
                current_section = "returns"
                current_content = []
            elif stripped.lower().startswith("raises:"):
                self._finalize_section(parsed, current_section, current_content)
                current_section = "raises"
                current_content = []
            elif stripped.lower().startswith(("example", "examples:")):
                self._finalize_section(parsed, current_section, current_content)
                current_section = "examples"
                current_content = []
            elif stripped.lower().startswith(("note:", "notes:")):
                self._finalize_section(parsed, current_section, current_content)
                current_section = "notes"
                current_content = []
            else:
                # Add content to current section
                if stripped or current_content:  # Include empty lines if we have content
                    current_content.append(line)

        # Finalize the last section
        self._finalize_section(parsed, current_section, current_content)

        return parsed

    def _finalize_section(self, parsed: Dict[str, Any], section: str, content: List[str]):
        """Finalize a docstring section by processing its content."""
        if not content:
            return

        if section == "description":
            # Join description lines, preserving paragraph breaks
            desc_text = "\n".join(content).strip()
            parsed["description"] = desc_text
        elif section == "args":
            # Parse argument descriptions
            parsed["args"] = self._parse_args_section(content)
        elif section == "returns":
            # Join return description
            parsed["returns"] = "\n".join(line.strip() for line in content if line.strip())
        elif section == "raises":
            # Parse raises section
            parsed["raises"] = self._parse_raises_section(content)
        elif section == "examples":
            # Extract code examples
            parsed["examples"] = self._parse_examples_section(content)
        elif section == "notes":
            # Join notes
            parsed["notes"] = [line.strip() for line in content if line.strip()]

    def _parse_args_section(self, lines: List[str]) -> List[Dict[str, str]]:
        """Parse the Args section of a docstring."""
        args = []
        current_arg = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this line starts a new argument (has a colon)
            if ":" in stripped and not stripped.startswith(" "):
                # Finalize previous argument
                if current_arg:
                    args.append(current_arg)

                # Start new argument
                parts = stripped.split(":", 1)
                arg_name = parts[0].strip()
                arg_desc = parts[1].strip() if len(parts) > 1 else ""

                current_arg = {
                    "name": arg_name,
                    "description": arg_desc,
                }
            elif current_arg and stripped:
                # Continue description of current argument
                current_arg["description"] += " " + stripped

        # Don't forget the last argument
        if current_arg:
            args.append(current_arg)

        return args

    def _parse_raises_section(self, lines: List[str]) -> List[Dict[str, str]]:
        """Parse the Raises section of a docstring."""
        raises = []
        current_exception = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this line starts a new exception (has a colon)
            if ":" in stripped and not stripped.startswith(" "):
                # Finalize previous exception
                if current_exception:
                    raises.append(current_exception)

                # Start new exception
                parts = stripped.split(":", 1)
                exc_name = parts[0].strip()
                exc_desc = parts[1].strip() if len(parts) > 1 else ""

                current_exception = {
                    "exception": exc_name,
                    "description": exc_desc,
                }
            elif current_exception and stripped:
                # Continue description of current exception
                current_exception["description"] += " " + stripped

        # Don't forget the last exception
        if current_exception:
            raises.append(current_exception)

        return raises

    def _parse_examples_section(self, lines: List[str]) -> List[str]:
        """Parse the Examples section of a docstring."""
        examples = []
        current_example = []
        in_code_block = False

        for line in lines:
            stripped = line.strip()

            # Detect code blocks
            if stripped.startswith("```"):
                if in_code_block:
                    # End of code block
                    if current_example:
                        examples.append("\n".join(current_example))
                        current_example = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
            elif in_code_block:
                # Inside code block
                current_example.append(line)
            elif stripped and not in_code_block:
                # Regular example line (not in code block)
                current_example.append(stripped)
            elif not stripped and current_example:
                # Empty line - finalize current example
                examples.append("\n".join(current_example))
                current_example = []

        # Don't forget the last example
        if current_example:
            examples.append("\n".join(current_example))

        return examples

    def render(self, data: dict, options: dict) -> str:
        """
        Render documentation using specialized templates.

        Args:
            data: Documentation data from collect()
            options: Rendering options

        Returns:
            Rendered HTML string
        """
        if not MKDOCSTRINGS_AVAILABLE:
            return "<div class='error'>mkdocstrings not available</div>"

        if data.get("error"):
            return f"<div class='error'>Error: {data['error']}</div>"

        # Simple HTML rendering - in a real implementation,
        # this would use Jinja2 templates
        if data.get("type") == "plugin":
            return self._render_plugin(data, options)
        else:
            return self._render_component(data, options)

    def _render_plugin(self, data: dict, options: dict) -> str:
        """Render comprehensive plugin overview with detailed component information."""
        # Calculate component statistics
        total_components = sum(len(comps) for comps in data["components"].values())
        component_stats = {comp_type: len(comps) for comp_type, comps in data["components"].items() if comps}

        html = f"""
        <div class="owa-plugin">
            <h1>🔌 Plugin: {data["namespace"]}</h1>

            <div class="plugin-metadata">
                <p><strong>📦 Version:</strong> <code>{data["version"]}</code></p>
                <p><strong>📝 Description:</strong> {data["description"]}</p>
                {f"<p><strong>👤 Author:</strong> {data['author']}</p>" if data.get("author") else ""}
                <p><strong>📊 Total Components:</strong> {total_components}</p>
                {self._render_component_stats(component_stats)}
            </div>

            <h2>📋 Components Overview</h2>
        """

        for comp_type, components in data["components"].items():
            if components:
                html += f"""
                <div class="component-section">
                    <h3>🔧 {comp_type.title()} ({len(components)})</h3>
                    <div class="components-grid">
                """

                for comp in components:
                    html += self._render_plugin_component_card(comp)

                html += """
                    </div>
                </div>
                """

        html += "</div>"
        return html

    def _render_component_stats(self, stats: Dict[str, int]) -> str:
        """Render component statistics as a small table."""
        if not stats:
            return ""

        html = "<div class='component-stats'><strong>📈 Component Breakdown:</strong><ul>"
        for comp_type, count in stats.items():
            html += f"<li>{comp_type.title()}: {count}</li>"
        html += "</ul></div>"
        return html

    def _render_plugin_component_card(self, comp: Dict[str, Any]) -> str:
        """Render a component card for the plugin overview."""
        if comp.get("error"):
            return f"""
            <div class="component-card error">
                <h4>❌ {comp["full_name"]}</h4>
                <p class="error">Error: {comp["error"]}</p>
                <p><strong>Import Path:</strong> <code>{comp["import_path"]}</code></p>
            </div>
            """

        # Get summary from parsed docstring or fallback to first line
        summary = ""
        if comp.get("parsed_docstring", {}).get("summary"):
            summary = comp["parsed_docstring"]["summary"]
        elif comp.get("docstring"):
            summary = comp["docstring"].split("\n")[0]

        # Get signature info
        signature = comp.get("signature", "")
        if signature and len(signature) > 80:
            signature = signature[:77] + "..."

        return f"""
        <div class="component-card">
            <h4>🎯 <code>{comp["full_name"]}</code></h4>
            {f"<p class='summary'>{summary}</p>" if summary else ""}
            {f"<p><strong>Signature:</strong> <code>{signature}</code></p>" if signature else ""}
            <p><strong>Import:</strong> <code>{comp["import_path"]}</code></p>
            {self._render_usage_examples_brief(comp.get("usage_examples", []))}
        </div>
        """

    def _render_usage_examples_brief(self, examples: List[str]) -> str:
        """Render brief usage examples for component cards."""
        if not examples:
            return ""

        # Show only the first example to keep cards compact
        example = examples[0]
        if len(example) > 60:
            example = example[:57] + "..."

        return f"<p><strong>Usage:</strong> <code>{example}</code></p>"

    def _render_component(self, data: dict, options: dict) -> str:
        """Render comprehensive individual component documentation."""
        html = f"""
        <div class="owa-component">
            <h1>🎯 {data["full_name"]}</h1>

            <div class="component-metadata">
                <p><strong>🏷️ Type:</strong> <span class="component-type">{data["component_type"]}</span></p>
                <p><strong>📦 Namespace:</strong> <code>{data["namespace"]}</code></p>
                <p><strong>📂 Import Path:</strong> <code>{data["import_path"]}</code></p>
                {self._render_component_class_info(data.get("component_class", {}))}
            </div>
        """

        # Render signature information
        if options.get("show_signature", True):
            html += self._render_signature_section(data.get("signature_info", {}), options)

        # Render documentation
        html += self._render_documentation_section(data.get("parsed_docstring", {}), data.get("docstring"))

        # Render usage examples
        if options.get("show_examples", True):
            html += self._render_usage_examples_section(data.get("usage_examples", []))

        # Render source code
        if options.get("show_source", True):
            html += self._render_source_section(data.get("source"), options)

        html += "</div>"
        return html

    def _render_component_class_info(self, class_info: Dict[str, Any]) -> str:
        """Render component class/type information."""
        if not class_info:
            return ""

        html = "<div class='class-info'>"

        # Component type indicators
        type_indicators = []
        if class_info.get("is_class"):
            type_indicators.append("Class")
        if class_info.get("is_function"):
            type_indicators.append("Function")
        if class_info.get("is_method"):
            type_indicators.append("Method")
        if class_info.get("is_callable"):
            type_indicators.append("Callable")

        if type_indicators:
            html += f"<p><strong>🔧 Type:</strong> {', '.join(type_indicators)}</p>"

        # Module information
        if class_info.get("module"):
            html += f"<p><strong>📁 Module:</strong> <code>{class_info['module']}</code></p>"

        # Base classes for classes
        if class_info.get("bases"):
            bases_str = ", ".join(f"<code>{base}</code>" for base in class_info["bases"])
            html += f"<p><strong>🏗️ Base Classes:</strong> {bases_str}</p>"

        html += "</div>"
        return html

    def _render_signature_section(self, sig_info: Dict[str, Any], options: Dict[str, Any] = None) -> str:
        """Render detailed signature information using Griffe data when available."""
        if options is None:
            options = {}
        if not sig_info or not sig_info.get("signature_str"):
            return "<h2>📝 Signature</h2><p>No signature information available.</p>"

        html = f"""
        <h2>📝 Signature</h2>
        <div class="signature-section">
            <pre class="signature"><code>{sig_info["signature_str"]}</code></pre>
        """

        # Use Griffe parameters if available, otherwise fallback to inspect-based
        parameters = sig_info.get("griffe_parameters", sig_info.get("parameters", []))

        if parameters:
            html += "<h3>Parameters</h3><div class='parameters'>"
            for param in parameters:
                html += self._render_parameter_enhanced(param)
            html += "</div>"

        # Render return type - prefer Griffe data
        griffe_info = sig_info.get("griffe_info", {})
        return_type = griffe_info.get("returns") or sig_info.get("return_annotation")

        if return_type:
            html += f"<h3>Returns</h3><p><strong>Type:</strong> <code>{return_type}</code></p>"

        # Add Griffe-specific information if available
        if griffe_info and options.get("show_griffe_info", True):
            html += self._render_griffe_metadata(griffe_info)

        html += "</div>"
        return html

    def _render_parameter_enhanced(self, param: Dict[str, Any]) -> str:
        """Render a parameter with enhanced information from Griffe."""
        html = f"<div class='parameter'><strong>{param['name']}</strong>"

        # Type annotation
        if param.get("annotation"):
            html += f" : <code>{param['annotation']}</code>"

        # Parameter kind (for Griffe parameters)
        if param.get("kind") and param["kind"] not in ["positional or keyword", "POSITIONAL_OR_KEYWORD"]:
            kind_display = param["kind"].replace("_", " ").lower()
            html += f" <em>({kind_display})</em>"

        # Default value
        if param.get("has_default") and param.get("default"):
            html += f" = <code>{param['default']}</code>"
        elif param.get("has_default"):
            html += " = <em>optional</em>"

        html += "</div>"
        return html

    def _render_griffe_metadata(self, griffe_info: Dict[str, Any]) -> str:
        """Render additional metadata from Griffe analysis."""
        html = ""

        # Component kind
        if griffe_info.get("kind"):
            html += f"<h3>Component Type</h3><p><strong>Kind:</strong> {griffe_info['kind']}</p>"

        # Source location
        if griffe_info.get("lineno"):
            location = f"Line {griffe_info['lineno']}"
            if griffe_info.get("endlineno") and griffe_info["endlineno"] != griffe_info["lineno"]:
                location += f"-{griffe_info['endlineno']}"
            html += f"<p><strong>Source Location:</strong> {location}</p>"

        # Decorators
        if griffe_info.get("decorators"):
            decorators_str = ", ".join(f"<code>@{dec}</code>" for dec in griffe_info["decorators"])
            html += f"<p><strong>Decorators:</strong> {decorators_str}</p>"

        # Labels/tags
        if griffe_info.get("labels"):
            labels_str = ", ".join(f"<span class='label'>{label}</span>" for label in griffe_info["labels"])
            html += f"<p><strong>Labels:</strong> {labels_str}</p>"

        return html

    def _render_parameter(self, param: Dict[str, Any]) -> str:
        """Render a single parameter."""
        html = f"<div class='parameter'><strong>{param['name']}</strong>"

        if param.get("annotation"):
            html += f" : <code>{param['annotation']}</code>"

        if param.get("has_default") and param.get("default"):
            html += f" = <code>{param['default']}</code>"
        elif param.get("has_default"):
            html += " = <em>optional</em>"

        html += "</div>"
        return html

    def _render_documentation_section(self, parsed_doc: Dict[str, Any], raw_docstring: Optional[str]) -> str:
        """Render the documentation section with structured docstring information."""
        if not parsed_doc and not raw_docstring:
            return "<h2>📚 Documentation</h2><p>No documentation available.</p>"

        html = "<h2>📚 Documentation</h2><div class='documentation'>"

        # Summary
        if parsed_doc.get("summary"):
            html += f"<h3>Summary</h3><p class='summary'>{parsed_doc['summary']}</p>"

        # Description
        if parsed_doc.get("description"):
            desc = parsed_doc["description"].replace("\n", "<br>")
            html += f"<h3>Description</h3><div class='description'>{desc}</div>"

        # Arguments
        if parsed_doc.get("args"):
            html += "<h3>Arguments</h3><div class='arguments'>"
            for arg in parsed_doc["args"]:
                html += f"<div class='argument'><strong>{arg['name']}</strong>: {arg['description']}</div>"
            html += "</div>"

        # Returns
        if parsed_doc.get("returns"):
            html += f"<h3>Returns</h3><div class='returns'>{parsed_doc['returns']}</div>"

        # Raises
        if parsed_doc.get("raises"):
            html += "<h3>Raises</h3><div class='raises'>"
            for exc in parsed_doc["raises"]:
                html += f"<div class='exception'><strong>{exc['exception']}</strong>: {exc['description']}</div>"
            html += "</div>"

        # Notes
        if parsed_doc.get("notes"):
            html += "<h3>Notes</h3><div class='notes'>"
            for note in parsed_doc["notes"]:
                html += f"<p>{note}</p>"
            html += "</div>"

        # Raw docstring fallback
        if not any(parsed_doc.get(key) for key in ["summary", "description", "args", "returns"]) and raw_docstring:
            html += f"<div class='raw-docstring'><pre>{raw_docstring}</pre></div>"

        html += "</div>"
        return html

    def _render_usage_examples_section(self, examples: List[str]) -> str:
        """Render usage examples section."""
        if not examples:
            return ""

        html = "<h2>💡 Usage Examples</h2><div class='usage-examples'>"

        for i, example in enumerate(examples, 1):
            html += f"""
            <div class='example'>
                <h4>Example {i}</h4>
                <pre><code>{example}</code></pre>
            </div>
            """

        html += "</div>"
        return html

    def _render_source_section(self, source: Optional[str], options: Dict[str, Any] = None) -> str:
        """Render source code section."""
        if not source:
            return ""

        if options is None:
            options = {}

        # Truncate very long source code based on options
        truncate_limit = options.get("truncate_source", 2000)
        if len(source) > truncate_limit:
            source = source[:truncate_limit] + "\n\n# ... (source truncated)"

        html = f"""
        <h2>🔍 Source Code</h2>
        <div class='source-code'>
            <details>
                <summary>Click to view source</summary>
                <pre><code>{source}</code></pre>
            </details>
        </div>
        """
        return html


def get_handler(**kwargs) -> OWAHandler:
    """Entry point for mkdocstrings handler registration."""
    return OWAHandler(**kwargs)
