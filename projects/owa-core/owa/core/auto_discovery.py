# ================ OEP-0003: Automatic Plugin Discovery ================================
# Handles automatic plugin discovery on import

import os

from loguru import logger

from .plugin_discovery import discover_and_register_plugins


def _should_auto_discover() -> bool:
    """
    Check if automatic plugin discovery should be enabled.

    Returns:
        True if auto-discovery should run, False otherwise
    """
    # Allow disabling auto-discovery via environment variable
    if os.environ.get("OWA_DISABLE_AUTO_DISCOVERY", "").lower() in ("1", "true", "yes"):
        return False

    return True


def auto_discover_plugins() -> None:
    """
    Automatically discover and register plugins if enabled.

    This function is called when owa-core is imported to automatically
    discover and register all installed plugins via entry points.
    """
    if not _should_auto_discover():
        logger.debug("Auto-discovery disabled")
        return

    try:
        discover_and_register_plugins()
    except Exception as e:
        logger.warning(f"Auto-discovery failed: {e}")
        # Don't raise - allow the application to continue without plugins


# Note: Auto-discovery is now triggered explicitly from __init__.py
# to avoid circular import issues during module loading
