# OCAP Tests

Minimal, focused tests for the `ocap` module.

## Test Structure

```
tests/
├── test_utilities.py              # Tests for utility functions only
├── test_integration.py            # Integration tests for ocap command
└── README.md                      # This file
```

## What We Test

- **Utility functions** with business logic (`parse_additional_properties`, `ensure_output_files_ready`)
- **Integration tests** for the `ocap` command startup and initialization

## Running Tests

```bash
# Run all tests
python -m pytest projects/ocap/tests/ -v

# Run specific test files
python -m pytest projects/ocap/tests/test_utilities.py -v
python -m pytest projects/ocap/tests/test_integration.py -v
```

## Test Coverage

### Utility Functions (`test_utilities.py`)
- `parse_additional_properties()` - Argument parsing logic
- `ensure_output_files_ready()` - File handling and directory creation

### Integration Tests (`test_integration.py`)
- `ocap --help` command functionality
- `ocap` command startup without crashing
- Graceful error handling with invalid arguments
- Fallback to `owa.cli` if standalone command unavailable