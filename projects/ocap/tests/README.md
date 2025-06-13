# ocap Tests

Comprehensive tests for the ocap (Omnimodal CAPture) recording functionality.

## Test Coverage

### Core Functionality
- **Event Queue Management**: Tests for `enqueue_event` function and global event queue
- **Callback Functions**: Tests for keyboard, mouse, and screen capture callbacks
- **Plugin System**: Tests for plugin discovery and validation
- **Utility Functions**: Tests for argument parsing and file management
- **Resource Management**: Tests for the `setup_resources` context manager
- **Integration**: End-to-end tests for the main recording workflow

### Test Classes

#### `TestEventQueue`
- Event enqueueing with timestamps
- Multiple event handling
- FIFO queue behavior

#### `TestCallbacks`
- Keyboard event callbacks (regular keys and F1-F12 keys)
- Screen capture callbacks with path handling
- Event topic routing

#### `TestPluginChecking`
- Successful plugin discovery
- Plugin failure handling
- Error reporting

#### `TestUtilityFunctions`
- Additional properties parsing
- Output file preparation
- Directory creation
- File conflict resolution

#### `TestResourceSetup`
- Context manager lifecycle
- Resource startup and teardown
- Exception handling during cleanup

#### `TestIntegration`
- Main record function workflow
- Window capture warnings
- CLI argument handling

## Running Tests

### Using the Test Runner (Recommended)
```bash
# Run all tests with verbose output
python projects/ocap/tests/run_tests.py --verbose

# Run tests with coverage reporting
python projects/ocap/tests/run_tests.py --coverage

# Run a specific test file
python projects/ocap/tests/run_tests.py --test test_record.py

# Combine options
python projects/ocap/tests/run_tests.py --verbose --coverage
```

### Using pytest directly

#### Run all ocap tests
```bash
python -m pytest projects/ocap/tests/ -v
```

#### Run specific test file
```bash
python -m pytest projects/ocap/tests/test_record.py -v
```

#### Run specific test class
```bash
python -m pytest projects/ocap/tests/test_record.py::TestEventQueue -v
```

#### Run specific test method
```bash
python -m pytest projects/ocap/tests/test_record.py::TestEventQueue::test_enqueue_event -v
```

#### Run with coverage
```bash
python -m pytest projects/ocap/tests/ --cov=owa.ocap --cov-report=html
```

## Test Design

### Mocking Strategy
- **External Dependencies**: All external dependencies (GStreamer, desktop APIs) are mocked
- **File System**: Uses temporary directories for file operations
- **Event Queue**: Cleared before and after each test to ensure isolation
- **Logging**: Mocked to verify correct log messages without output

### Fixtures
- `temp_output_dir`: Provides temporary directory for test files
- `mock_event`: Creates mock event objects for testing
- `clear_event_queue`: Ensures event queue is clean for each test

### Test Isolation
- Each test is independent and can run in any order
- Global state (event queue, MCAP_LOCATION) is properly managed
- No actual recording or hardware interaction occurs during tests

## Notes

- Tests are designed to run without requiring actual GStreamer plugins or desktop recording capabilities
- All hardware-dependent functionality is mocked to ensure tests can run in CI environments
- Tests focus on the core logic and error handling rather than actual recording functionality
- Integration tests verify the overall workflow without performing actual recording operations
