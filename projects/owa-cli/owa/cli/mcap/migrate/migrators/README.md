# MCAP Migration Framework Specification

**Version:** 1.0
**Status:** Active
**Last Updated:** 2025-06-25

## Overview

This specification defines the implementation requirements for MCAP migration scripts within the Open World Agents ecosystem. MCAP migrators are version-specific transformation utilities that enable seamless data migration between different MCAP format versions.

## Architecture

### Design Principles

- **Isolation**: Each migrator operates as a standalone PEP 723 script with embedded dependencies
- **Reproducibility**: Exact dependency pinning ensures consistent behavior across environments
- **Composability**: Standardized interfaces enable programmatic chaining and validation
- **Separation of Concerns**: Migrators focus solely on transformation; orchestration is handled externally

### Execution Model

Migrators are executed via `uv run` to provide:
- Dependency isolation from host environment
- Reproducible execution contexts
- Automatic dependency resolution

## Technical Specification

### 1. File Naming Convention

**Pattern:** `v{major}_{minor}_{patch}_to_v{major}_{minor}_{patch}.py`

**Examples:**
- `v0_4_2_to_v0_5_0.py` - Minor version upgrade
- `v1_0_0_to_v1_1_0.py` - Feature release migration
- `v2_3_1_to_v2_3_2.py` - Patch-level migration

### 2. Script Structure

All migrators **MUST** implement the PEP 723 script format:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich>=13.0.0",
#     "mcap>=1.0.0",
#     "typer>=0.9.0",
#     # Additional dependencies as required
# ]
# [tool.uv]
# exclude-newer = "YYYY-MM-DDTHH:MM:SSZ"
# ///
```

**Requirements:**
- Shebang line **MUST** specify `uv run --script`
- Python version **MUST** be `>=3.11`
- Dependencies **SHOULD** use exact version pinning for reproducibility
- `exclude-newer` **SHOULD** be set to prevent dependency drift

### 3. Command Interface

Each migrator **MUST** implement exactly two commands with the following standardized signatures:

#### 3.1 Migration Command

```python
@app.command()
def migrate(
    input_file: Path = typer.Argument(
        ...,
        help="Input MCAP file path"
    ),
    output_file: Optional[Path] = typer.Argument(
        None,
        help="Output MCAP file path (defaults to in-place modification)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging output"
    ),
    output_format: str = typer.Option(
        "text",
        "--output-format",
        help="Output format: 'text' or 'json'"
    ),
) -> None:
    """
    Migrate MCAP file from source version to target version.

    Transforms the input MCAP file according to the version-specific
    migration rules. If output_file is not specified, performs in-place
    modification of the input file.
    """
```

#### 3.2 Verification Command

```python
@app.command()
def verify(
    file_path: Path = typer.Argument(
        ...,
        help="MCAP file path to verify"
    ),
    backup_path: Optional[Path] = typer.Option(
        None,
        help="Reference backup file path (optional)"
    ),
    output_format: str = typer.Option(
        "text",
        "--output-format",
        help="Output format: 'text' or 'json'"
    ),
) -> None:
    """
    Verify migration completeness and data integrity.

    Validates that all legacy structures have been properly migrated
    and no data corruption has occurred during the transformation process.
    """
```

### 4. Output Format Specification

#### 4.1 JSON Schema Compliance

When `--output-format json` is specified, all output **MUST** conform to the schema defined in [`../schemas.json`](../schemas.json).

#### 4.2 Migration Command Output

**Successful Migration:**
```json
{
    "success": true,
    "changes_made": 42,
    "from_version": "0.4.2",
    "to_version": "0.5.0"
}
```

**Failed Migration:**
```json
{
    "success": false,
    "changes_made": 0,
    "error": "Detailed error description",
    "from_version": "0.4.2",
    "to_version": "0.5.0"
}
```

#### 4.3 Verification Command Output

**Successful Verification:**
```json
{
    "success": true,
    "message": "Migration verification completed successfully"
}
```

**Failed Verification:**
```json
{
    "success": false,
    "error": "Legacy structures detected: [structure_list]"
}
```

#### 4.4 Text Output Format

For `--output-format text`, migrators **SHOULD** provide human-readable output with:
- Clear success/failure indicators
- Progress information (when `--verbose` is enabled)
- Error details with actionable guidance

## Implementation Requirements

### 5. Mandatory Requirements (MUST)

The following requirements are **mandatory** for framework compatibility:

| Requirement | Description | Rationale |
|-------------|-------------|-----------|
| **File Naming** | Pattern: `v{from}_to_v{to}.py` | Enables automatic discovery and version mapping |
| **PEP 723 Header** | Complete script metadata with dependencies | Ensures reproducible execution environment |
| **Command Signatures** | Exact parameter signatures as specified | Maintains interface compatibility for orchestration |
| **JSON Schema** | Output conforming to `../schemas.json` | Enables programmatic validation and chaining |
| **No Backup Logic** | Migrators do not handle backup creation | Separation of concerns with orchestrator |

### 6. Recommended Practices (SHOULD)

The following practices are **strongly recommended** for production quality:

| Practice | Description | Benefit |
|----------|-------------|---------|
| **Exact Versioning** | Pin dependencies to specific versions | Prevents version drift and ensures reproducibility |
| **Target Compatibility** | Use target version for `owa-*` packages | Ensures compatibility with destination format |
| **Pre-flight Checks** | Detect legacy structures before transformation | Enables early failure and better error reporting |
| **Data Preservation** | Preserve all non-migrated data exactly | Prevents data loss during transformation |
| **Comprehensive Logging** | Detailed logging with `--verbose` flag | Facilitates debugging and audit trails |

## Design Rationale

### 7. Architectural Decisions

#### 7.1 PEP 723 Script Format
- **Benefit**: Dependency isolation and reproducibility
- **Trade-off**: Slightly more complex than simple Python scripts
- **Justification**: Critical for production stability across environments

#### 7.2 uv Execution Environment
- **Benefit**: Consistent, isolated execution context
- **Trade-off**: Requires uv installation
- **Justification**: Modern Python tooling standard with superior dependency resolution

#### 7.3 JSON Schema Validation
- **Benefit**: Programmatic validation and tool chaining
- **Trade-off**: Additional complexity for simple use cases
- **Justification**: Enables automated testing and CI/CD integration

#### 7.4 Orchestrator-Managed Backups
- **Benefit**: Clear separation of concerns
- **Trade-off**: Migrators cannot handle their own backup logic
- **Justification**: Centralized backup strategy with consistent policies

## Compliance and Testing

### 8. Validation Checklist

Before deployment, migrators **MUST** pass the following validation:

- [ ] Filename follows naming convention
- [ ] PEP 723 header is complete and valid
- [ ] Both `migrate` and `verify` commands are implemented
- [ ] Command signatures match specification exactly
- [ ] JSON output validates against schema
- [ ] Text output is human-readable
- [ ] No backup logic is implemented
- [ ] Dependencies are properly pinned
- [ ] Script executes successfully with `uv run`

### 9. Integration Testing

Migrators **SHOULD** be tested with:
- Sample MCAP files from source version
- Edge cases and malformed data
- Large file performance validation
- JSON schema compliance verification
- Round-trip migration testing (where applicable)

## Testing Tips

You can test your migrator works within virtual environment without uploading package to pypi:

```
# dependencies = [
#   ...
#   "mcap-owa-support==0.5.0a3",
#   "owa-core==0.5.0a3",
#   "owa-msgs==0.5.0a3",
# ]
# [tool.uv.sources]
# mcap-owa-support = { path = "../../../../../../mcap-owa-support", editable = true }
# owa-core = { path = "../../../../../../owa-core", editable = true }
# owa-msgs = { path = "../../../../../../owa-msgs", editable = true }
```