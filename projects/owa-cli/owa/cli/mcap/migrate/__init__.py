import typer

from .cleanup import cleanup
from .migrate import (
    MigrationOrchestrator,
    MigrationResult,
    ScriptMigrator,
    detect_files_needing_migration,
    validate_migration_output,
    validate_verification_output,
)
from .migrate import (
    migrate as migrate_command,
)
from .rollback import rollback

# Create the migrate app with subcommands
app = typer.Typer(help="MCAP migration commands with rollback and cleanup support.")

# Add subcommands
app.command(name="run")(migrate_command)
app.command(name="rollback")(rollback)
app.command(name="cleanup")(cleanup)

# Expose the app as migrate for CLI registration
migrate = app

__all__ = [
    "migrate",
    "app",
    "MigrationOrchestrator",
    "ScriptMigrator",
    "MigrationResult",
    "detect_files_needing_migration",
    "validate_migration_output",
    "validate_verification_output",
]
