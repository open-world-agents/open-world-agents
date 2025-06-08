"""
Signature inspection for OEP-0004 documentation system.

Provides automatic signature extraction and analysis for components.
"""

import importlib
import inspect
from typing import Any, Dict, List, Optional

from ..component_access import get_registry
from .models import ComponentDocumentation
from .parser import DocstringParser


class SignatureInspector:
    """
    Automatic signature extraction and analysis for plugin components.

    This class analyzes component source code to extract function signatures,
    type hints, and other structural information.
    """

    def __init__(self):
        self.parser = DocstringParser()

    def inspect_component(self, component_type: str, full_name: str) -> Optional[ComponentDocumentation]:
        """
        Inspect a component and extract its documentation.

        Args:
            component_type: Type of component (callables, listeners, runnables)
            full_name: Full component name (namespace/name)

        Returns:
            Component documentation or None if inspection fails
        """
        try:
            # Get the component
            registry = get_registry(component_type)
            if not registry:
                return None

            import_path = registry._import_paths.get(full_name)
            if not import_path:
                return None

            # Parse import path
            module_path, obj_name = import_path.rsplit(":", 1)

            # Import the module and get the object
            module = importlib.import_module(module_path)
            obj = getattr(module, obj_name)

            # Extract documentation
            return self._extract_component_documentation(obj, full_name, component_type, module_path, obj_name)

        except Exception:
            # Log error but don't fail completely
            return None

    def _extract_component_documentation(
        self, obj: Any, full_name: str, component_type: str, module_path: str, obj_name: str
    ) -> ComponentDocumentation:
        """Extract documentation from a component object."""

        # Get source file and line number
        source_file = "unknown"
        line_number = 0
        try:
            source_file = inspect.getfile(obj)
            _, line_number = inspect.getsourcelines(obj)
        except (OSError, TypeError):
            pass

        # Parse docstring
        docstring = inspect.getdoc(obj) or ""
        parsed_doc = self.parser.parse(docstring)

        # Extract signature information
        signature_info = self._extract_signature_info(obj)

        # Determine if it's a class and extract methods/attributes
        methods = []
        attributes = []
        inheritance = []

        if inspect.isclass(obj):
            inheritance = [cls.__name__ for cls in obj.__mro__[1:]]  # Exclude self

            # Extract methods (simplified to avoid recursion)
            for name, method in inspect.getmembers(obj, inspect.ismethod):
                if not name.startswith("_"):  # Skip private methods
                    try:
                        method_signature = str(inspect.signature(method))
                        method_doc = inspect.getdoc(method) or ""
                        parsed_method_doc = self.parser.parse(method_doc)

                        # Create simplified method documentation
                        method_component = ComponentDocumentation(
                            name=name,
                            full_name=f"{full_name}.{name}",
                            type="method",
                            source_file=source_file,
                            line_number=0,
                            summary=parsed_method_doc.summary,
                            description=parsed_method_doc.description,
                            parameters=parsed_method_doc.parameters,
                            returns=parsed_method_doc.returns,
                            raises=parsed_method_doc.raises,
                            examples=parsed_method_doc.examples,
                            notes=parsed_method_doc.notes,
                            signature=method_signature,
                            type_hints={},
                            is_async=inspect.iscoroutinefunction(method),
                            is_method=True,
                            is_classmethod=isinstance(method, classmethod),
                            is_staticmethod=isinstance(method, staticmethod),
                            methods=[],
                            attributes=[],
                            inheritance=[],
                        )
                        methods.append(method_component)
                    except Exception:
                        pass  # Skip methods that can't be inspected

            # Extract functions (unbound methods)
            for name, func in inspect.getmembers(obj, inspect.isfunction):
                if not name.startswith("_"):  # Skip private methods
                    try:
                        func_signature = str(inspect.signature(func))
                        func_doc = inspect.getdoc(func) or ""
                        parsed_func_doc = self.parser.parse(func_doc)

                        # Create simplified function documentation
                        func_component = ComponentDocumentation(
                            name=name,
                            full_name=f"{full_name}.{name}",
                            type="method",
                            source_file=source_file,
                            line_number=0,
                            summary=parsed_func_doc.summary,
                            description=parsed_func_doc.description,
                            parameters=parsed_func_doc.parameters,
                            returns=parsed_func_doc.returns,
                            raises=parsed_func_doc.raises,
                            examples=parsed_func_doc.examples,
                            notes=parsed_func_doc.notes,
                            signature=func_signature,
                            type_hints={},
                            is_async=inspect.iscoroutinefunction(func),
                            is_method=False,
                            is_classmethod=False,
                            is_staticmethod=False,
                            methods=[],
                            attributes=[],
                            inheritance=[],
                        )
                        methods.append(func_component)
                    except Exception:
                        pass  # Skip functions that can't be inspected

        # Create component documentation
        _, name = full_name.split("/", 1) if "/" in full_name else (full_name, "")

        return ComponentDocumentation(
            name=name,
            full_name=full_name,
            type=component_type,
            source_file=source_file,
            line_number=line_number,
            # From docstring
            summary=parsed_doc.summary,
            description=parsed_doc.description,
            parameters=parsed_doc.parameters,
            returns=parsed_doc.returns,
            raises=parsed_doc.raises,
            examples=parsed_doc.examples,
            notes=parsed_doc.notes,
            # From signature
            signature=signature_info["signature"],
            type_hints=signature_info["type_hints"],
            is_async=signature_info["is_async"],
            is_method=signature_info["is_method"],
            is_classmethod=signature_info["is_classmethod"],
            is_staticmethod=signature_info["is_staticmethod"],
            # Class-specific
            methods=methods,
            attributes=attributes,
            inheritance=inheritance,
        )

    def _extract_signature_info(self, obj: Any) -> Dict[str, Any]:
        """Extract signature information from an object."""
        info = {
            "signature": "",
            "type_hints": {},
            "is_async": False,
            "is_method": False,
            "is_classmethod": False,
            "is_staticmethod": False,
        }

        try:
            # Get signature
            if callable(obj):
                sig = inspect.signature(obj)
                info["signature"] = str(sig)

                # Extract type hints
                for param_name, param in sig.parameters.items():
                    if param.annotation != inspect.Parameter.empty:
                        info["type_hints"][param_name] = str(param.annotation)

                if sig.return_annotation != inspect.Signature.empty:
                    info["type_hints"]["return"] = str(sig.return_annotation)

            # Check if it's async
            if inspect.iscoroutinefunction(obj):
                info["is_async"] = True

            # Check method types
            if inspect.ismethod(obj):
                info["is_method"] = True
            elif isinstance(obj, classmethod):
                info["is_classmethod"] = True
            elif isinstance(obj, staticmethod):
                info["is_staticmethod"] = True

        except (ValueError, TypeError):
            pass

        return info

    def get_all_plugin_components(self, namespace: str) -> Dict[str, List[ComponentDocumentation]]:
        """
        Get documentation for all components in a plugin namespace.

        Args:
            namespace: Plugin namespace

        Returns:
            Dictionary mapping component types to lists of component documentation
        """
        components = {}

        for component_type in ["callables", "listeners", "runnables"]:
            registry = get_registry(component_type)
            if not registry:
                continue

            type_components = []
            prefix = f"{namespace}/"

            for full_name in registry._import_paths.keys():
                if full_name.startswith(prefix):
                    doc = self.inspect_component(component_type, full_name)
                    if doc:
                        type_components.append(doc)

            if type_components:
                components[component_type] = type_components

        return components
