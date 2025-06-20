"""
Migrator from MCAP v0.3.2 to v0.4.1.

This migrator handles the schema format changes:
- Migrates from module-based schema names (e.g., 'owa.env.desktop.msg.KeyboardEvent')
  to domain-based format (e.g., 'desktop/KeyboardEvent')
"""

import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console

try:
    from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter
    from owa.core import MESSAGES
except ImportError as e:
    raise ImportError(f"Required packages not available: {e}. Please install: pip install mcap-owa-support") from e

from .base import BaseMigrator, MigrationResult, verify_migration_integrity


class V032ToV041Migrator(BaseMigrator):
    """Migrator from v0.3.2 to v0.4.1 (domain-based schema format)."""

    # Legacy to new message type mapping
    LEGACY_MESSAGE_MAPPING = {
        "owa.env.desktop.msg.KeyboardEvent": "desktop/KeyboardEvent",
        "owa.env.desktop.msg.KeyboardState": "desktop/KeyboardState",
        "owa.env.desktop.msg.MouseEvent": "desktop/MouseEvent",
        "owa.env.desktop.msg.MouseState": "desktop/MouseState",
        "owa.env.desktop.msg.WindowInfo": "desktop/WindowInfo",
        "owa.env.gst.msg.ScreenEmitted": "desktop/ScreenCaptured",
    }

    @property
    def from_version(self) -> str:
        return "0.3.2"

    @property
    def to_version(self) -> str:
        return "0.4.1"

    def migrate(self, file_path: Path, console: Console, verbose: bool) -> MigrationResult:
        """Migrate legacy schemas to domain-based format."""
        changes_made = 0

        try:
            # Note: Backup is now handled by the orchestrator before calling this method

            # Analyze file first
            analysis = self._analyze_file(file_path)

            if not analysis["has_legacy_messages"]:
                return MigrationResult(
                    success=True,
                    version_from=self.from_version,
                    version_to=self.to_version,
                    changes_made=0,
                )

            # Perform migration using temporary file
            with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_file:
                temp_path = Path(tmp_file.name)

            try:
                with OWAMcapWriter(str(temp_path)) as writer:
                    with OWAMcapReader(str(file_path)) as reader:
                        for msg in reader.iter_messages():
                            schema_name = msg.message_type

                            if schema_name in analysis["conversions"]:
                                new_schema_name = analysis["conversions"][schema_name]
                                new_message_class = MESSAGES[new_schema_name]

                                if hasattr(msg, "decoded") and msg.decoded:
                                    if hasattr(msg.decoded, "model_dump"):
                                        data = msg.decoded.model_dump()
                                    else:
                                        data = dict(msg.decoded)

                                    new_message = new_message_class(**data)
                                    writer.write_message(
                                        msg.topic,
                                        new_message,
                                        publish_time=msg.publish_time,
                                        log_time=msg.log_time,
                                    )
                                    changes_made += 1

                                    if verbose:
                                        console.print(f"  Migrated: {schema_name} → {new_schema_name}")
                            else:
                                # Copy message as-is if no conversion needed
                                writer.write_message(
                                    msg.topic,
                                    msg.decoded,
                                    publish_time=msg.publish_time,
                                    log_time=msg.log_time,
                                )

                # Replace original file with migrated version
                shutil.move(str(temp_path), str(file_path))

                return MigrationResult(
                    success=True,
                    version_from=self.from_version,
                    version_to=self.to_version,
                    changes_made=changes_made,
                )

            except Exception as e:
                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()
                raise e

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
            # First, verify that no legacy schemas remain
            with OWAMcapReader(file_path) as reader:
                for schema in reader.schemas.values():
                    if schema.name in self.LEGACY_MESSAGE_MAPPING:
                        console.print(f"[red]Legacy schema still present: {schema.name}[/red]")
                        return False

            # If backup is available, perform integrity verification
            if backup_path and backup_path.exists():
                return verify_migration_integrity(file_path, backup_path, console)

            # If no backup, just verify no legacy schemas remain
            console.print("[green]✓ No legacy schemas found[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Verification error: {e}[/red]")
            return False

    def _analyze_file(self, file_path: Path) -> Dict:
        """Analyze file to determine what needs migration."""
        analysis = {
            "total_messages": 0,
            "legacy_message_count": 0,
            "has_legacy_messages": False,
            "conversions": {},
            "schemas": {},
        }

        with OWAMcapReader(file_path) as reader:
            # Analyze schemas
            for schema_id, schema in reader.schemas.items():
                analysis["schemas"][schema_id] = schema.name

                if schema.name in self.LEGACY_MESSAGE_MAPPING:
                    new_name = self.LEGACY_MESSAGE_MAPPING[schema.name]
                    analysis["conversions"][schema.name] = new_name
                    analysis["has_legacy_messages"] = True

            # Count messages
            for message in reader.iter_messages():
                analysis["total_messages"] += 1
                if message.message_type in self.LEGACY_MESSAGE_MAPPING:
                    analysis["legacy_message_count"] += 1

        return analysis


# Explicit export - only this migrator should be discovered
__all__ = ["V032ToV041Migrator"]
