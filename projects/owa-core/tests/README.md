# OWA-Core Tests

Organized test structure for the owa-core package.

## Test Categories

- **Unit tests** (root level): Test individual modules in isolation
- **Integration tests** (`integration/`): Test cross-module functionality
- **I/O tests** (`io/`): Test external interfaces and file operations

## Shared Fixtures

The `conftest.py` file provides shared fixtures:
- `isolated_registries`: Clean registry instances for testing
- `create_mock_entry_point`: Mock entry point creation
- `mock_entry_points_factory`: Entry point factory for testing
