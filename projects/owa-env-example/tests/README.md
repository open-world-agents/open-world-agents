# OWA Example Plugin Tests

Simple tests for the `owa-env-example` plugin.

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_plugin.py -v
```

## Test Coverage

- Plugin activation and component registration
- Callable components: `example/callable`, `example/print`, `example/add`
- Listener components: `example/listener`, `example/timer`
- Runnable components: `example/runnable`, `example/counter`
