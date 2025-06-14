# OCAP Tests

Minimal, focused tests for the `ocap` module.

## Test Structure

```
tests/
├── test_integration.py            # Integration tests for ocap command
└── README.md                      # This file
```

## What We Test

- **Integration tests** for the `ocap` command startup and initialization

## Running Tests

```bash
# Run all tests
python -m pytest projects/ocap/tests/ -v

# Run integration tests
python -m pytest projects/ocap/tests/test_integration.py -v
```

## Test Coverage

### Integration Tests (`test_integration.py`)
- `ocap --help` command functionality
- `ocap` command startup without crashing
- Graceful error handling with invalid arguments
- Fallback to `owa.cli` if standalone command unavailable