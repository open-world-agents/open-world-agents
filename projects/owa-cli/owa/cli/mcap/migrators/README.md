# MCAP Migrators

This package contains individual migrator implementations for different MCAP version transitions. Each migrator is responsible for handling a specific version-to-version migration with verification and rollback capabilities.

## Structure

```
migrators/
├── __init__.py          # Package exports
├── base.py              # Base classes and common structures
├── v030_to_v032.py      # v0.3.0 → v0.3.2 migration
├── v032_to_v040.py      # v0.3.2 → v0.4.0 migration
└── README.md            # This file
```

## Base Classes

### `BaseMigrator`
Abstract base class that all migrators must inherit from. Defines the interface:

- `from_version`: Source version this migrator handles
- `to_version`: Target version this migrator produces  
- `migrate()`: Perform the migration with backup and verification
- `verify_migration()`: Verify that migration was successful

### `MigrationResult`
Data class containing the result of a migration operation:

- `success`: Whether the migration succeeded
- `version_from`/`version_to`: Version transition
- `changes_made`: Number of changes applied
- `error_message`: Error details if migration failed
- `backup_path`: Path to backup file created

## Automatic Discovery

The migration system automatically discovers and registers migrators using `__all__` exports for safety and predictability:

### Discovery Rules
- **File naming**: Must follow pattern `vXYZ_to_vABC.py` (e.g., `v030_to_v032.py`)
- **Explicit exports**: Must define `__all__` with migrator class names
- **Class inheritance**: Exported classes must inherit from `BaseMigrator`
- **Location**: Must be in the `migrators/` package

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

### `V032ToV040Migrator`  
**Purpose**: Migrates to domain-based schema format
**Changes**:
- `owa.env.desktop.msg.KeyboardEvent` → `desktop/KeyboardEvent`
- `owa.env.desktop.msg.MouseEvent` → `desktop/MouseEvent`
- And other module-based → domain-based schema name conversions

## Adding New Migrators

To add a new migrator for version X.Y.Z → A.B.C:

1. **Create the migrator file**: `vXYZ_to_vABC.py`

2. **Implement the migrator class**:
```python
from .base import BaseMigrator, MigrationResult

class VXYZToVABCMigrator(BaseMigrator):
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
__all__ = ["VXYZToVABCMigrator"]
```

4. **That's it!** The migrator will be automatically discovered and registered.

The system uses automatic discovery based on:
- File naming pattern: `vXYZ_to_vABC.py`
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
