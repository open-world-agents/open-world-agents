# OWA-CLI Tests

Organized test structure for the owa-cli package.

## Structure

```
tests/
├── conftest.py                    # Global fixtures and configurations
├── test_no_api_calls.py          # Global API call prevention tests
├── mcap/                          # MCAP-related tests
│   ├── conftest.py                # MCAP-specific fixtures
│   ├── data/                      # Test MCAP files
│   ├── test_sanitize.py           # Sanitization command tests
│   ├── test_upgrade.py            # Upgrade command tests
│   ├── backup/
│   │   └── test_backup_context.py # BackupContext tests
│   └── migrate/
│       ├── conftest.py            # Migration-specific fixtures
│       ├── test_cli.py            # CLI interface tests
│       ├── test_core.py           # Core migration functionality
│       └── test_system.py         # Migration system tests
├── env/                           # Environment plugin tests
│   └── test_validate.py          # Environment validation tests
├── messages/                      # Message management tests (placeholder)
├── video/                         # Video processing tests (placeholder)
└── utils/                         # Utility function tests
    ├── test_version.py            # Version checking utilities
    └── test_verification.py       # Verification utilities
```

## Test Organization Principles

### Modular Structure
- Tests are organized to mirror the source code structure
- Each major CLI component has its own test directory
- Shared fixtures are provided at appropriate levels

### Fixture Hierarchy
- **Global fixtures** (`tests/conftest.py`): CLI runner, temp directories, API mocking
- **MCAP fixtures** (`tests/mcap/conftest.py`): MCAP-specific utilities, warning suppression
- **Migration fixtures** (`tests/mcap/migrate/conftest.py`): Migration-specific mocks

### Test Categories

#### MCAP Tests (`mcap/`)
- **CLI tests** (`migrate/test_cli.py`): End-to-end CLI command testing
- **Core tests** (`migrate/test_core.py`): Core migration functionality
- **System tests** (`migrate/test_system.py`): Migration system integration
- **Backup tests** (`backup/test_backup_context.py`): Backup and rollback functionality
- **Feature tests** (`test_sanitize.py`, `test_upgrade.py`): Individual command features

#### Environment Tests (`env/`)
- Environment plugin validation and management

#### Utility Tests (`utils/`)
- Version checking and verification utilities
- Shared utility functions

## Running Tests

### All Tests
```bash
# From project root
python -m pytest projects/owa-cli/tests/ -v
```

### Specific Test Categories
```bash
# MCAP tests only
python -m pytest projects/owa-cli/tests/mcap/ -v

# Migration tests only
python -m pytest projects/owa-cli/tests/mcap/migrate/ -v

# Backup tests only
python -m pytest projects/owa-cli/tests/mcap/backup/ -v

# Environment tests only
python -m pytest projects/owa-cli/tests/env/ -v

# Utility tests only
python -m pytest projects/owa-cli/tests/utils/ -v
```

### Individual Test Files
```bash
# CLI migration tests
python -m pytest projects/owa-cli/tests/mcap/migrate/test_cli.py -v

# Backup context tests
python -m pytest projects/owa-cli/tests/mcap/backup/test_backup_context.py -v
```

## Key Features

### Fixture Reuse
- Common test utilities are provided as fixtures
- MCAP warning suppression is centralized
- Mock objects are reusable across tests

### Test Data Management
- Test MCAP files are located in `mcap/data/`
- Test data is shared across migration tests
- File copying utilities are provided as fixtures

### API Call Prevention
- Global mocking prevents GitHub API calls during testing
- Environment variables disable version checking
- Comprehensive coverage of all API endpoints

## Migration from Old Structure

The tests have been refactored from a flat structure to an organized hierarchy:

### Old Structure (Flat)
```
tests/
├── test_backup_utils.py
├── test_mcap_migrate_cli.py
├── test_migrate.py
├── test_migration_system.py
├── test_sanitize.py
├── test_mcap_upgrade.py
├── test_validate_command.py
├── test_version_utils.py
├── test_verification_utilities.py
└── test_no_api_calls.py
```

### New Structure (Organized)
- Tests are grouped by functionality
- Shared utilities are extracted to fixtures
- Test data is properly organized
- Import paths are updated for new structure

## Best Practices

1. **Use appropriate fixtures**: Leverage the fixture hierarchy for common functionality
2. **Organize by feature**: Keep related tests together in appropriate directories
3. **Minimize duplication**: Use shared fixtures instead of duplicating test utilities
4. **Clear naming**: Test files and functions should clearly indicate what they test
5. **Proper isolation**: Tests should not depend on each other or external state
