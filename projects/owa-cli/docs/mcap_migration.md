# MCAP Migration System

The MCAP migration system provides automatic version detection and sequential migration for MCAP files, bringing them up to the latest format version with verification and rollback capabilities.

## Features

- **Automatic Version Detection**: Detects the version of MCAP files using embedded metadata
- **Sequential Migration**: Applies migrations in the correct order (e.g., 0.3.0 → 0.3.2 → 0.4.0)
- **Verification & Rollback**: Each migration step is verified and can be rolled back on failure
- **Glob Pattern Support**: Process multiple files using glob patterns
- **Backup Management**: Automatic backup creation with optional cleanup
- **Dry Run Mode**: Preview changes without modifying files

## Usage

### Basic Migration

Migrate all MCAP files in the current directory (shell expands the glob):
```bash
owl mcap migrate *.mcap
```

### Recursive Migration

Migrate all MCAP files recursively (shell expands the glob):
```bash
owl mcap migrate data/**/*.mcap
```

### Single File Migration

Migrate a specific file:
```bash
owl mcap migrate recording.mcap
```

### Multiple Specific Files

Migrate multiple specific files:
```bash
owl mcap migrate file1.mcap file2.mcap file3.mcap
```

### Preview Changes

Use dry-run mode to see what would be migrated:
```bash
owl mcap migrate *.mcap --dry-run
```

### Target Specific Version

Migrate to a specific version instead of the latest:
```bash
owl mcap migrate *.mcap --target 0.3.2
```

### Verbose Output

Show detailed migration information:
```bash
owl mcap migrate *.mcap --verbose
```

### Backup Management

Control backup file handling:
```bash
# Keep backups (default)
owl mcap migrate *.mcap --keep-backups

# Remove backups after successful migration
owl mcap migrate *.mcap --no-backups
```

## Migration Paths

The system supports the following migration paths:

### Sequential Migrations
- **0.3.0 → 0.3.2**: Updates keyboard state field names (`pressed_vk_list` → `buttons`)
- **0.3.2 → 0.4.0**: Migrates to domain-based schema format (e.g., `owa.env.desktop.msg.KeyboardEvent` → `desktop/KeyboardEvent`)

### Example Migration Sequences
- `0.3.0` → `0.3.2` → `0.4.0` (2 steps)
- `0.3.2` → `0.4.0` (1 step)
- `0.4.0` → `0.4.0` (no migration needed)

**Note**: Files with legacy schemas (e.g., `owa.env.*`) are detected as version `0.3.2` and will go through the schema migration step to reach `0.4.0`.

## Version Detection

The system detects file versions using multiple methods:

1. **OWAMcapReader metadata** (most reliable): Reads version from file headers using the high-level API
2. **Schema analysis**: Detects legacy schemas that need migration using exposed schema information
3. **Field analysis**: Checks for specific field patterns (e.g., `pressed_vk_list`)
4. **Fallback**: Assumes current mcap-owa-support library version if detection fails

The migration system automatically uses the current mcap-owa-support library version as the target version, ensuring compatibility with the installed library.

## Safety Features

### Automatic Backups
- Creates backup files before each migration step
- Backup files are stored in `.mcap_migration_backups` directories
- Backup naming: `{filename}_backup_{from_version}_to_{to_version}.mcap`

### Verification
- Each migration step is verified after completion
- Verification ensures the migration was successful
- Failed verification triggers automatic rollback

### Rollback
- Automatic rollback on migration failure
- Restores original file from backup
- Preserves data integrity

### Error Handling
- Graceful handling of corrupted files
- Detailed error messages
- Continues processing other files on individual failures

## Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `files` | MCAP files to migrate (supports shell glob expansion) | Required |
| `--target`, `-t` | Target version | Latest (current mcap-owa-support version) |
| `--dry-run` | Preview changes without modifying files | False |
| `--force` | Force migration even if files appear up-to-date | False |
| `--verbose`, `-v` | Show detailed migration information | False |
| `--keep-backups` / `--no-backups` | Keep backup files after migration | True |

## Examples

### Single File Migration
```bash
owl mcap migrate recording.mcap
```

### Multiple Files with Preview
```bash
owl mcap migrate recordings/*.mcap --dry-run --verbose
```

### Force Migration
```bash
owl mcap migrate *.mcap --force
```

### Clean Migration (no backups)
```bash
owl mcap migrate data/**/*.mcap --no-backups
```

## Troubleshooting

### Common Issues

**"No migration path found"**
- The source version is not supported
- Check file integrity and version detection

**"Migration verification failed"**
- The migration didn't complete successfully
- File is automatically restored from backup
- Check file permissions and disk space

**"No MCAP files found"**
- Check the glob pattern syntax
- Ensure files have `.mcap` extension
- Use absolute paths if needed

### Debug Information

Use `--verbose` to see detailed information:
- Version detection results
- Migration steps being performed
- Schema transformations
- Verification results

### Manual Recovery

If automatic rollback fails, backup files are available in `.mcap_migration_backups` directories:
```bash
# List backup files
ls .mcap_migration_backups/

# Manually restore a file
cp .mcap_migration_backups/recording_backup_0.3.0_to_0.3.2.mcap recording.mcap
```

## Integration

The migration system can be used programmatically:

```python
from owa.cli.mcap.migrate import MigrationOrchestrator
from rich.console import Console

orchestrator = MigrationOrchestrator()
console = Console()

# Migrate a single file
results = orchestrator.migrate_file(
    file_path=Path("recording.mcap"),
    target_version="0.4.0",
    console=console,
    verbose=True
)

# Check results
for result in results:
    if result.success:
        print(f"Migration successful: {result.changes_made} changes")
    else:
        print(f"Migration failed: {result.error_message}")
```
