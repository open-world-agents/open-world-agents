"""
Migrator from MCAP v0.3.0 to v0.3.2.

This migrator handles the keyboard state field changes:
- Changes `pressed_vk_list` field to `buttons` in keyboard/state messages
"""

from pathlib import Path
from typing import Optional

from rich.console import Console

try:
    from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
except ImportError as e:
    raise ImportError(f"Required packages not available: {e}. Please install: pip install mcap-owa-support") from e

from .base import BaseMigrator, MigrationResult, verify_migration_integrity


class V030ToV032Migrator(BaseMigrator):
    """Migrator from v0.3.0 to v0.3.2 (keyboard state field changes)."""

    @property
    def from_version(self) -> str:
        return "0.3.0"

    @property
    def to_version(self) -> str:
        return "0.3.2"

    def migrate(self, file_path: Path, console: Console, verbose: bool) -> MigrationResult:
        """Migrate keyboard state field from pressed_vk_list to buttons."""
        changes_made = 0

        try:
            # Note: Backup is now handled by the orchestrator before calling this method
            msgs = []
            with OWAMcapReader(file_path) as reader:
                for schema, channel, message, decoded in reader.reader.iter_decoded_messages():
                    if channel.topic == "keyboard/state" and "pressed_vk_list" in decoded:
                        buttons = decoded.pop("pressed_vk_list")
                        decoded["buttons"] = buttons
                        changes_made += 1

                        if verbose:
                            console.print("  Updated keyboard/state: pressed_vk_list → buttons")

                    # Reconstruct message object
                    try:
                        module, class_name = schema.name.rsplit(".", 1)
                        import importlib

                        module = importlib.import_module(module)
                        cls = getattr(module, class_name)
                        decoded = cls(**decoded)
                    except Exception:
                        pass

                    msgs.append((message.log_time, channel.topic, decoded))

            # Write back the file
            if changes_made > 0:
                with OWAMcapWriter(file_path) as writer:
                    for log_time, topic, msg in msgs:
                        writer.write_message(topic=topic, message=msg, log_time=log_time)

            return MigrationResult(
                success=True,
                version_from=self.from_version,
                version_to=self.to_version,
                changes_made=changes_made,
            )

        except Exception as e:
            return MigrationResult(
                success=False,
                version_from=self.from_version,
                version_to=self.to_version,
                changes_made=0,
                error_message=str(e),
            )

    def verify_migration(self, file_path: Path, backup_path: Optional[Path], console: Console) -> bool:
        """Verify that migration was successful."""
        try:
            # First, verify that pressed_vk_list fields are gone
            with OWAMcapReader(file_path) as reader:
                for msg in reader.iter_messages(topics=["keyboard/state"]):
                    if hasattr(msg.decoded, "pressed_vk_list"):
                        console.print("[red]pressed_vk_list field still present in keyboard/state message[/red]")
                        return False

            # If backup is available, perform integrity verification
            if backup_path and backup_path.exists():
                return verify_migration_integrity(file_path, backup_path, console)

            # If no backup, just verify the field migration
            console.print("[green]✓ pressed_vk_list fields successfully migrated[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Verification error: {e}[/red]")
            return False


# Explicit export - only this migrator should be discovered
__all__ = ["V030ToV032Migrator"]
