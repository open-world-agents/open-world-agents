# MCAP Migrators

This package contains individual migrator implementations for different MCAP version transitions. Each migrator is responsible for handling a specific version-to-version migration with verification and rollback capabilities.

**Important TODO**:
- [ ] Contanerized migration script is needed to ensure sequential migration.

## Structure

```
migrators/
├── __init__.py              # Package exports
├── base.py                  # Base classes and common structures
├── v0_3_0_to_v0_3_2.py      # v0.3.0 → v0.3.2 migration
├── v0_3_2_to_v0_4_1.py      # v0.3.2 → v0.4.1 migration
└── README.md                # This file
```

## Base Classes

### `BaseMigrator`
Abstract base class that all migrators must inherit from. Defines the interface:

- `from_version`: Source version this migrator handles
- `to_version`: Target version this migrator produces
- `migrate()`: Perform the migration (backup is handled by orchestrator)
- `verify_migration()`: Verify that migration was successful

### `MigrationResult`
Data class containing the result of a migration operation:

- `success`: Whether the migration succeeded
- `version_from`/`version_to`: Version transition
- `changes_made`: Number of changes applied
- `error_message`: Error details if migration failed
- `backup_path`: Path to backup file created

## Backup Strategy

Backup creation is handled centrally by the `MigrationOrchestrator` for high reliability:

- **Centralized Logic**: All backup operations go through `MigrationOrchestrator.create_backup()`
- **High Reliability**: Includes verification of backup size and existence
- **Consistent Behavior**: All migrators benefit from the same robust backup logic
- **Error Handling**: Failed backups prevent migration from proceeding

Individual migrators should **not** create their own backups. The backup path is provided to migrators for reference in the `MigrationResult` only.

## Version Detection

Version detection is simplified and reliable:

- **Primary Method**: Uses `OWAMcapReader.file_version` which reads the version from the MCAP file header
- **Stored at Write Time**: The version is written by `mcap-owa-support` when the file is created
- **No Complex Logic**: Eliminates the need for schema inspection or message content analysis
- **Fallback**: If version cannot be read, defaults to current `mcap-owa-support` version

This approach is more reliable than the previous complex detection logic that tried to infer versions from file content.

## Automatic Discovery

The migration system automatically discovers and registers migrators using `__all__` exports for safety and predictability:

### Discovery Rules
- **File naming**: Must follow pattern `v{X}_{Y}_{Z}_to_v{A}_{B}_{C}.py` (e.g., `v0_3_0_to_v0_3_2.py`, `v1_113_0_to_v2_0_0.py`)
- **Explicit exports**: Must define `__all__` with migrator class names
- **Class inheritance**: Exported classes must inherit from `BaseMigrator`
- **Location**: Must be in the `migrators/` package

### File Naming Convention
The naming convention supports full semantic versioning by replacing dots with underscores:

**Examples:**
- `v0_3_0_to_v0_3_2.py` for version 0.3.0 → 0.3.2
- `v0_4_1_to_v1_0_0.py` for version 0.4.1 → 1.0.0
- `v1_113_0_to_v2_0_0.py` for version 1.113.0 → 2.0.0
- `v2_5_17_to_v3_0_0.py` for version 2.5.17 → 3.0.0

**Class naming follows the same pattern:**
- `V1_113_0_To_V2_0_0_Migrator` for the class in `v1_113_0_to_v2_0_0.py`

**Benefits:**
- ✅ **Full semantic versioning support**: Handles any X.Y.Z version format
- ✅ **Clear version mapping**: Easy to see source and target versions
- ✅ **Filesystem safe**: No dots in filenames that could cause issues
- ✅ **Sortable**: Files naturally sort by version order

### Discovery Process
1. Scans all Python files in the migrators package matching naming pattern
2. Imports modules and checks for `__all__` definition
3. Only processes classes explicitly listed in `__all__`
4. Verifies classes inherit from `BaseMigrator`
5. Automatically instantiates and registers valid migrators

### Safety Features
- ✅ **Explicit control**: Only classes in `__all__` are considered
- ✅ **No accidental discovery**: Helper classes and imports are ignored
- ✅ **Predictable behavior**: Clear warning if `__all__` is missing
- ✅ **Type safety**: Validates inheritance from `BaseMigrator`

### Benefits
- ✅ **Zero configuration**: Just add the file, class, and `__all__`
- ✅ **No manual registration**: No need to update central lists
- ✅ **Automatic ordering**: Migration orchestrator builds paths automatically
- ✅ **Safe and predictable**: Explicit exports prevent surprises

## Existing Migrators

### `V030ToV032Migrator`
**Purpose**: Migrates keyboard state field changes
**Changes**: 
- `pressed_vk_list` field → `buttons` field in keyboard/state messages

### `V032ToV041Migrator`
**Purpose**: Migrates to domain-based schema format
**Changes**:
- `owa.env.desktop.msg.KeyboardEvent` → `desktop/KeyboardEvent`
- `owa.env.desktop.msg.MouseEvent` → `desktop/MouseEvent`
- And other module-based → domain-based schema name conversions

## Adding New Migrators

To add a new migrator for version X.Y.Z → A.B.C:

1. **Create the migrator file**: `vX_Y_Z_to_vA_B_C.py` (e.g., `v1_113_0_to_v2_0_0.py`)

2. **Implement the migrator class**:
```python
from .base import BaseMigrator, MigrationResult

class VX_Y_Z_ToVA_B_C_Migrator(BaseMigrator):
    @property
    def from_version(self) -> str:
        return "X.Y.Z"

    @property
    def to_version(self) -> str:
        return "A.B.C"
    
    def migrate(self, file_path: Path, backup_path: Path, console: Console, verbose: bool) -> MigrationResult:
        # Implementation here
        pass
    
    def verify_migration(self, file_path: Path, console: Console) -> bool:
        # Verification logic here
        pass
```

3. **Add explicit export** at the end of the file:
```python
# Explicit export - only this migrator should be discovered
__all__ = ["VX_Y_Z_ToVA_B_C_Migrator"]
```

4. **That's it!** The migrator will be automatically discovered and registered.

The system uses automatic discovery based on:
- File naming pattern: `vX_Y_Z_to_vA_B_C.py` (supports semantic versioning)
- Explicit exports: Must define `__all__` with migrator class names
- Class inheritance: Must inherit from `BaseMigrator`
- No manual registration required!

## Migration Guidelines

### Safety First
- Always create backups before making changes
- Implement comprehensive verification
- Handle errors gracefully with detailed messages
- Use temporary files for complex transformations

### High-Level API Usage
- Use `OWAMcapReader` and `OWAMcapWriter` instead of low-level APIs
- Leverage exposed schema information from the reader
- Follow the established patterns from existing migrators

### Error Handling
- Return `MigrationResult` with detailed error information
- Clean up temporary files on failure
- Provide helpful error messages for debugging

### Testing
- Add tests for new migrators in `../../../tests/test_migration_system.py`
- Test both successful migration and error cases
- Verify that the orchestrator can find migration paths

## Example Migration Pattern

```python
def migrate(self, file_path: Path, backup_path: Path, console: Console, verbose: bool) -> MigrationResult:
    changes_made = 0
    
    try:
        # Create backup
        shutil.copy2(file_path, backup_path)
        
        # Analyze what needs migration
        analysis = self._analyze_file(file_path)
        
        if not analysis["needs_migration"]:
            return MigrationResult(success=True, ...)
        
        # Perform migration with temporary file
        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)
        
        try:
            # Migration logic here
            with OWAMcapWriter(str(temp_path)) as writer:
                with OWAMcapReader(str(file_path)) as reader:
                    # Process messages...
                    changes_made += 1
            
            # Replace original with migrated version
            temp_path.replace(file_path)
            
            return MigrationResult(success=True, changes_made=changes_made, ...)
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e
            
    except Exception as e:
        return MigrationResult(success=False, error_message=str(e), ...)
```

This modular structure makes it easy to add new migrations while keeping the code organized and maintainable.
