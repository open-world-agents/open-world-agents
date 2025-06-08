"""
Custom mkdocstrings handler for OWA EnvPlugin components.

This module implements a minimal mkdocstrings handler that understands
OWA's plugin structure and can generate documentation automatically.
"""

import inspect
from typing import Any, Optional

try:
    from mkdocstrings import BaseHandler

    MKDOCSTRINGS_AVAILABLE = True
except ImportError:
    # mkdocstrings not available, create a dummy base class
    MKDOCSTRINGS_AVAILABLE = False

    class BaseHandler:
        """Dummy base handler when mkdocstrings is not available."""

        name = "owa"
        domain = "py"
        fallback_theme = "material"

        def __init__(self, **kwargs):
            pass


from ..plugin_discovery import get_plugin_discovery
from ..registry import CALLABLES, LISTENERS, RUNNABLES


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
                    comp_info = {
                        "name": comp_name,
                        "full_name": full_name,
                        "import_path": import_path,
                        "docstring": inspect.getdoc(component),
                        "signature": self._get_signature(component),
                    }
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

        # Extract detailed component information
        docstring = inspect.getdoc(component)
        signature = self._get_signature(component)

        # Parse docstring for structured information
        parsed_doc = self._parse_docstring(docstring) if docstring else {}

        return {
            "type": "component",
            "name": component_name,
            "full_name": full_name,
            "namespace": namespace,
            "component_type": comp_type_found,
            "import_path": import_path,
            "docstring": docstring,
            "signature": signature,
            "parsed_docstring": parsed_doc,
            "source": self._get_source(component),
        }

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

    def _get_source(self, component: Any) -> Optional[str]:
        """Get the source code of a component."""
        try:
            return inspect.getsource(component)
        except (OSError, TypeError):
            return None

    def _parse_docstring(self, docstring: str) -> dict:
        """Parse a docstring into structured components."""
        if not docstring:
            return {}

        lines = docstring.strip().split("\n")
        parsed = {
            "summary": "",
            "description": "",
            "args": [],
            "returns": "",
            "examples": [],
        }

        # Extract summary (first line)
        if lines:
            parsed["summary"] = lines[0].strip()

        # Simple parsing - could be enhanced for Google/Sphinx style
        current_section = "description"
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("Args:") or line.startswith("Arguments:"):
                current_section = "args"
            elif line.startswith("Returns:"):
                current_section = "returns"
            elif line.startswith("Example"):
                current_section = "examples"
            elif current_section == "description" and line:
                parsed["description"] += line + " "

        return parsed

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
        """Render plugin overview."""
        html = f"""
        <div class="owa-plugin">
            <h2>Plugin: {data["namespace"]}</h2>
            <p><strong>Version:</strong> {data["version"]}</p>
            <p><strong>Description:</strong> {data["description"]}</p>
            {f"<p><strong>Author:</strong> {data['author']}</p>" if data.get("author") else ""}

            <h3>Components</h3>
        """

        for comp_type, components in data["components"].items():
            if components:
                html += f"<h4>{comp_type.title()}</h4><ul>"
                for comp in components:
                    html += f"<li><code>{comp['full_name']}</code>"
                    if comp.get("docstring"):
                        summary = comp["docstring"].split("\n")[0]
                        html += f" - {summary}"
                    html += "</li>"
                html += "</ul>"

        html += "</div>"
        return html

    def _render_component(self, data: dict, options: dict) -> str:
        """Render individual component."""
        html = f"""
        <div class="owa-component">
            <h2>{data["full_name"]}</h2>
            <p><strong>Type:</strong> {data["component_type"]}</p>
            {f"<p><strong>Signature:</strong> <code>{data['signature']}</code></p>" if data.get("signature") else ""}

            {f"<div class='docstring'>{data['docstring']}</div>" if data.get("docstring") else "<p>No documentation available.</p>"}
        </div>
        """
        return html


def get_handler(**kwargs) -> OWAHandler:
    """Entry point for mkdocstrings handler registration."""
    return OWAHandler(**kwargs)
