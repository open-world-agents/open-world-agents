# OWA-Core Tests

Organized test structure for the owa-core package.

## Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── test_component_access.py       # Component access API tests
├── test_message.py                # Message base classes and verification
├── test_messages.py               # MessageRegistry tests
├── test_plugin_spec.py            # Plugin specification tests
├── test_registry.py               # Registry system tests
├── test_runnable.py               # Runnable classes tests
├── integration/
│   ├── test_message_system.py     # Message system integration tests
│   └── test_std_plugin.py         # Built-in std plugin tests
└── io/
    └── test_video.py               # Video I/O functionality tests
```

## Running Tests

```bash
# Run all tests
pytest projects/owa-core/tests/

# Run specific test categories
pytest projects/owa-core/tests/integration/  # Integration tests
pytest projects/owa-core/tests/io/           # I/O tests

# Run specific test file
pytest projects/owa-core/tests/test_registry.py

# Run with verbose output
pytest projects/owa-core/tests/ -v
```

## Test Categories

- **Unit tests** (root level): Test individual modules in isolation
- **Integration tests** (`integration/`): Test cross-module functionality
- **I/O tests** (`io/`): Test external interfaces and file operations

## Shared Fixtures

The `conftest.py` file provides shared fixtures:
- `isolated_registries`: Clean registry instances for testing
- `create_mock_entry_point`: Mock entry point creation
- `mock_entry_points_factory`: Entry point factory for testing
