# ================ Entry Points-Based Plugin Discovery ================================
# Implements automatic plugin discovery and registration using Python entry points

import os
import sys
from typing import Dict, Optional, Union

from loguru import logger

from .plugin_spec import PluginSpec
from .registry import CALLABLES, LISTENERS, RUNNABLES

# Import entry_points based on Python version
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

if not os.getenv("OWA_ENABLE_PLUGIN_DISCOVERY_LOGS", "false").lower() == "true":
    logger.disable("owa.core.plugin_discovery")


class PluginDiscovery:
    """
    Handles automatic discovery and registration of plugins via entry points.

    This class implements the core of OEP-0003 by scanning for plugins declared
    in entry points and registering their components with lazy loading.
    """

    ENTRY_POINT_GROUP = "owa.env.plugins"

    def __init__(self):
        self.discovered_plugins: Dict[str, PluginSpec] = {}
        self.failed_plugins: Dict[str, str] = {}  # plugin_name -> error_message

    def discover_plugins(self) -> None:
        """
        Discover all plugins declared via entry points.

        This method scans all installed packages for entry points in the
        'owa.env.plugins' group and loads their plugin specifications.
        """
        logger.info("Starting plugin discovery via entry points...")

        try:
            discovered_eps = entry_points(group=self.ENTRY_POINT_GROUP)
        except Exception as e:
            logger.error(f"Failed to discover entry points: {e}")
            return

        for ep in discovered_eps:
            try:
                self._load_plugin_spec(ep)
            except Exception as e:
                logger.error(f"Failed to load plugin '{ep.name}': {e}")
                self.failed_plugins[ep.name] = str(e)

        logger.info(
            f"Plugin discovery complete. Loaded: {len(self.discovered_plugins)}, Failed: {len(self.failed_plugins)}"
        )

    def _load_plugin_spec(self, entry_point) -> None:
        """
        Load and validate a plugin specification from an entry point.

        Args:
            entry_point: The entry point to load

        Raises:
            Exception: If plugin loading or validation fails
        """
        logger.debug(f"Loading plugin '{entry_point.name}' from {entry_point.value}")

        # Load the plugin specification object
        plugin_spec_obj = entry_point.load()

        # Validate it's a PluginSpec instance
        if not isinstance(plugin_spec_obj, PluginSpec):
            raise TypeError(
                f"Entry point '{entry_point.name}' must point to a PluginSpec instance, got {type(plugin_spec_obj)}"
            )

        # Validate component types
        plugin_spec_obj.validate_components()

        # Store the plugin spec
        self.discovered_plugins[entry_point.name] = plugin_spec_obj

        logger.info(
            f"Loaded plugin '{entry_point.name}' "
            f"(namespace: {plugin_spec_obj.namespace}, "
            f"version: {plugin_spec_obj.version})"
        )

    def register_all_components(self) -> None:
        """
        Register all discovered plugin components with lazy loading.

        This method registers component metadata with the global registries
        without actually importing the components (lazy loading).
        """
        logger.info("Registering plugin components...")

        total_registered = 0

        for plugin_name, plugin_spec in self.discovered_plugins.items():
            try:
                registered_count = self._register_plugin_components(plugin_name, plugin_spec)
                total_registered += registered_count
            except Exception as e:
                logger.error(f"Failed to register components for plugin '{plugin_name}': {e}")
                self.failed_plugins[plugin_name] = str(e)

        logger.info(f"Registered {total_registered} components across all plugins")

    def _register_plugin_components(self, plugin_name: str, plugin_spec: PluginSpec) -> int:
        """
        Register components for a single plugin.

        Args:
            plugin_name: Name of the plugin
            plugin_spec: Plugin specification

        Returns:
            Number of components registered
        """
        registered_count = 0

        # Register callables
        if "callables" in plugin_spec.components:
            for name, import_path in plugin_spec.components["callables"].items():
                full_name = f"{plugin_spec.namespace}/{name}"
                CALLABLES.register(full_name, obj_or_import_path=import_path)
                registered_count += 1
                logger.debug(f"Registered callable: {full_name} -> {import_path}")

        # Register listeners
        if "listeners" in plugin_spec.components:
            for name, import_path in plugin_spec.components["listeners"].items():
                full_name = f"{plugin_spec.namespace}/{name}"
                LISTENERS.register(full_name, obj_or_import_path=import_path)
                registered_count += 1
                logger.debug(f"Registered listener: {full_name} -> {import_path}")

        # Register runnables
        if "runnables" in plugin_spec.components:
            for name, import_path in plugin_spec.components["runnables"].items():
                full_name = f"{plugin_spec.namespace}/{name}"
                RUNNABLES.register(full_name, obj_or_import_path=import_path)
                registered_count += 1
                logger.debug(f"Registered runnable: {full_name} -> {import_path}")

        logger.info(f"Registered {registered_count} components for plugin '{plugin_name}'")
        return registered_count

    def get_plugin_info(self, plugin_name: Optional[Union[str, list[str]]] = None) -> tuple[Dict, Dict]:
        """
        Get information about discovered plugins.

        Args:
            plugin_name: Specific plugin name(s), or None for all plugins.
                        Can be a single plugin name (str) or a list of plugin names (list[str]).

        Returns:
            Tuple of (discovered_plugins, failed_plugins) dictionaries
        """
        discovered = {}
        failed = {}

        if plugin_name is not None:
            # Convert single plugin name to list for uniform handling
            plugin_names = [plugin_name] if isinstance(plugin_name, str) else plugin_name

            for name in plugin_names:
                if name in self.discovered_plugins:
                    discovered[name] = self.discovered_plugins[name].model_dump()
                elif name in self.failed_plugins:
                    failed[name] = self.failed_plugins[name]
            return discovered, failed

        # Return all plugins
        for name, spec in self.discovered_plugins.items():
            discovered[name] = spec.model_dump()

        for name, error in self.failed_plugins.items():
            failed[name] = error

        return discovered, failed


# Global plugin discovery instance
_plugin_discovery = PluginDiscovery()


def discover_and_register_plugins() -> None:
    """
    Discover and register all plugins via entry points.

    This is the main entry point for OEP-0003 plugin discovery.
    Call this function to automatically discover and register all
    installed plugins.
    """
    _plugin_discovery.discover_plugins()
    _plugin_discovery.register_all_components()


def get_plugin_discovery() -> PluginDiscovery:
    """Get the global plugin discovery instance."""
    return _plugin_discovery
