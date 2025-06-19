"""
MCAP migrators package.

This package contains individual migrator implementations for different MCAP version transitions.
Each migrator handles a specific version-to-version migration with verification and rollback capabilities.

Migrators are automatically discovered by scanning for classes that inherit from BaseMigrator.
"""

import importlib
import pkgutil
from pathlib import Path
from typing import List, Type

from .base import BaseMigrator, MigrationResult


def discover_migrators() -> List[Type[BaseMigrator]]:
    """
    Automatically discover all migrator classes in this package using __all__ exports.

    This approach is safer and more predictable than scanning for classes,
    as it relies on explicit exports defined in each migrator module.

    Returns:
        List of migrator classes that inherit from BaseMigrator
    """
    migrators = []
    package_dir = Path(__file__).parent

    # Scan all Python files in the migrators package
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name.startswith("v") and "_to_v" in module_info.name:
            try:
                # Import the module
                module = importlib.import_module(f".{module_info.name}", package=__name__)

                # Use __all__ if defined, otherwise skip the module
                if hasattr(module, "__all__"):
                    for name in module.__all__:
                        attr = getattr(module, name, None)
                        if (
                            attr is not None
                            and isinstance(attr, type)
                            and issubclass(attr, BaseMigrator)
                            and attr is not BaseMigrator
                        ):
                            migrators.append(attr)
                else:
                    print(f"Warning: Migrator module {module_info.name} does not define __all__. Skipping.")

            except ImportError as e:
                # Skip modules that can't be imported
                print(f"Warning: Could not import migrator module {module_info.name}: {e}")
                continue

    return migrators


def get_all_migrators() -> List[BaseMigrator]:
    """
    Get instances of all discovered migrators.

    Returns:
        List of instantiated migrator objects
    """
    migrator_classes = discover_migrators()
    return [migrator_class() for migrator_class in migrator_classes]


__all__ = [
    "BaseMigrator",
    "MigrationResult",
    "discover_migrators",
    "get_all_migrators",
]
