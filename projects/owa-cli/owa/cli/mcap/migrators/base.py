"""
Base classes for MCAP migrators.

This module defines the abstract base class and common data structures
used by all MCAP migrators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    success: bool
    version_from: str
    version_to: str
    changes_made: int
    error_message: Optional[str] = None
    backup_path: Optional[Path] = None


class BaseMigrator(ABC):
    """Base class for MCAP file migrators."""

    @property
    @abstractmethod
    def from_version(self) -> str:
        """Source version this migrator handles."""
        pass

    @property
    @abstractmethod
    def to_version(self) -> str:
        """Target version this migrator produces."""
        pass

    @abstractmethod
    def migrate(self, file_path: Path, backup_path: Path, console: Console, verbose: bool) -> MigrationResult:
        """
        Perform the migration.

        Note: Backup creation is handled by the orchestrator before calling this method.
        The backup_path is provided for reference in the MigrationResult.
        """
        pass

    @abstractmethod
    def verify_migration(self, file_path: Path, console: Console) -> bool:
        """Verify that migration was successful."""
        pass
