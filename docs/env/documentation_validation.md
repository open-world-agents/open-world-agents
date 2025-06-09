# Documentation Validation (OEP-0004)

OEP-0004 introduces a comprehensive documentation validation system for OWA plugins, providing both CI/CD integration and automatic documentation generation capabilities.

## Overview

The documentation validation system consists of two main components:

1. **Documentation Validator**: A command-line tool for validating plugin documentation quality
2. **mkdocstrings Handler**: A custom handler for automatic documentation generation

## Documentation Validation

### Basic Usage

Validate documentation for all plugins:

```bash
$ owl env validate-docs
✅ std plugin: 2/2 components documented (100%)
✅ example plugin: 7/7 components documented (100%)  
⚠️  desktop plugin: 23/25 components documented (92%)
❌ custom plugin: 1/5 components documented (20%)

📊 Overall: 33/39 components documented (85%)
❌ FAIL: Documentation coverage 85% below minimum 90%
```

Validate a specific plugin:

```bash
$ owl env validate-docs example
✅ example plugin: 7/7 components documented (100%)
✅ PASS: All components properly documented
```

### CI/CD Integration

The validation command provides proper exit codes for automated testing:

- **Exit Code 0**: All validations passed
- **Exit Code 1**: Documentation issues found
- **Exit Code 2**: Command error (invalid arguments, plugin not found, etc.)

#### GitHub Actions Example

```yaml
name: Documentation Validation
on: [push, pull_request]

jobs:
  validate-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e .
      - name: Validate Documentation
        run: owl env validate-docs --strict
```

### Advanced Options

#### Strict Mode

In strict mode, any missing documentation is treated as a failure:

```bash
$ owl env validate-docs --strict
❌ FAIL: Documentation coverage 88% below minimum 100% (strict mode)
```

#### JSON Output

For tooling integration, use JSON output format:

```bash
$ owl env validate-docs --format json
{
  "overall_coverage": 0.88,
  "plugins": {
    "example": {
      "coverage": 1.0,
      "documented": 7,
      "total": 7,
      "status": "pass"
    },
    "desktop": {
      "coverage": 0.92,
      "documented": 23,
      "total": 25,
      "status": "warning"
    }
  },
  "exit_code": 1
}
```

#### Quality Checks

Enable detailed quality validation:

```bash
$ owl env validate-docs --check-quality --min-examples 1
❌ desktop/window.get_active: Missing usage examples
❌ desktop/screen.capture: Missing parameter documentation
⚠️  Found 2 quality issues across 2 plugins
```

### Validation Criteria

The validator checks for:

1. **Docstring Presence**: Every component must have a non-empty docstring
2. **Docstring Quality**: Docstrings should include summary and parameter documentation
3. **Type Hints**: Functions should have type hints for parameters and return values
4. **Examples**: Components should include usage examples in docstrings
5. **Component Loading**: Components must be loadable without errors

## Automatic Documentation Generation

### mkdocstrings Integration

OEP-0004 provides a custom mkdocstrings handler that understands OWA's plugin structure:

#### Plugin Overview

```markdown
# Example Plugin Documentation

::: example
    handler: owa
```

This generates a complete plugin overview with all components.

#### Individual Components

```markdown
# Mouse Click Function

::: example/mouse.click
    handler: owa
    options:
      show_signature: true
      show_examples: true
```

This generates detailed documentation for a specific component.

### Configuration

The owa handler is available through the `mkdocstrings_handlers` package structure. Add it to your `mkdocs.yml`:

```yaml
plugins:
  - mkdocstrings:
      handlers:
        owa:
          options:
            show_plugin_metadata: true
            include_source_links: true
```

**Note**: The handler is located in `mkdocstrings_handlers/owa.py` following the mkdocstrings custom handler convention. It automatically detects and imports OWA core modules when available.

## Documentation Statistics

View documentation statistics for development:

```bash
$ owl env docs-stats
┏━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┓
┃ Plugin  ┃ Coverage ┃ Documented  ┃ Total ┃ Status ┃
┡━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━┩
│ std     │    100%  │          2  │     2 │   ✅   │
│ example │    100%  │          7  │     7 │   ✅   │
│ desktop │     92%  │         23  │    25 │   ⚠️   │
└─────────┴──────────┴─────────────┴───────┴────────┘

📊 Overall Coverage: 82.1% (32/39)
```

Group by component type:

```bash
$ owl env docs-stats --by-type
```

## Best Practices

### Writing Good Documentation

1. **Clear Summaries**: Start with a concise one-line summary
2. **Parameter Documentation**: Document all parameters with types and descriptions
3. **Return Values**: Clearly describe what the function returns
4. **Examples**: Include practical usage examples
5. **Type Hints**: Use proper type annotations

### Example: Well-Documented Component

```python
def screen_capture(region: Optional[Tuple[int, int, int, int]] = None) -> Image:
    """
    Capture a screenshot of the desktop or a specific region.
    
    This function captures the current state of the desktop and returns
    it as a PIL Image object. Optionally, you can specify a region to
    capture only a portion of the screen.
    
    Args:
        region: Optional tuple (x, y, width, height) defining the capture area.
               If None, captures the entire screen.
    
    Returns:
        PIL Image object containing the captured screenshot.
    
    Raises:
        ScreenCaptureError: If the screen capture fails.
    
    Examples:
        Capture the entire screen:
        
        >>> image = screen_capture()
        >>> image.save("screenshot.png")
        
        Capture a specific region:
        
        >>> region_image = screen_capture((100, 100, 800, 600))
        >>> region_image.show()
    """
    # Implementation here...
```

## Integration with Development Workflow

### Pre-commit Hooks

Add documentation validation to your pre-commit hooks:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-docs
        name: Validate Plugin Documentation
        entry: owl env validate-docs
        language: system
        pass_filenames: false
```

### pytest Integration

Include documentation validation in your test suite:

```python
import subprocess
import json

def test_plugin_documentation():
    """Test that all plugins have adequate documentation."""
    result = subprocess.run(
        ["owl", "env", "validate-docs", "--format", "json"],
        capture_output=True, text=True
    )
    
    data = json.loads(result.stdout)
    assert data["overall_coverage"] >= 0.9, f"Documentation coverage {data['overall_coverage']} below 90%"
    assert result.returncode == 0, "Documentation validation failed"
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all plugin dependencies are installed
2. **Missing Type Hints**: Add type annotations to function parameters and return values
3. **Empty Docstrings**: Add meaningful documentation to all components
4. **Loading Failures**: Check that components can be imported without errors

### Getting Help

- Check the validation output for specific issues
- Use `--check-quality` for detailed quality analysis
- Review the [Plugin Development Guide](custom_plugins.md) for best practices
