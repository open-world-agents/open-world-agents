"""
Migrator from MCAP v0.3.0 to v0.3.2.

This migrator handles the keyboard state field changes:
- Changes `pressed_vk_list` field to `buttons` in keyboard/state messages
"""

from pathlib import Path

from rich.console import Console

try:
    from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
except ImportError as e:
    raise ImportError(f"Required packages not available: {e}. Please install: pip install mcap-owa-support") from e

from .base import BaseMigrator, MigrationResult


class V030ToV032Migrator(BaseMigrator):
    """Migrator from v0.3.0 to v0.3.2 (keyboard state field changes)."""

    @property
    def from_version(self) -> str:
        return "0.3.0"

    @property
    def to_version(self) -> str:
        return "0.3.2"

    def migrate(self, file_path: Path, backup_path: Path, console: Console, verbose: bool) -> MigrationResult:
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
                backup_path=backup_path,
            )

        except Exception as e:
            return MigrationResult(
                success=False,
                version_from=self.from_version,
                version_to=self.to_version,
                changes_made=0,
                error_message=str(e),
                backup_path=backup_path,
            )

    def verify_migration(self, file_path: Path, console: Console) -> bool:
        """Verify that pressed_vk_list fields are gone."""
        try:
            with OWAMcapReader(file_path) as reader:
                for msg in reader.iter_messages(topics=["keyboard/state"]):
                    if hasattr(msg.decoded, "pressed_vk_list"):
                        return False
            return True
        except Exception:
            return False


# Explicit export - only this migrator should be discovered
__all__ = ["V030ToV032Migrator"]
