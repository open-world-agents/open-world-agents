# Documentation Quick Reference

Quick reference for OEP-0004 documentation validation and generation features.

## 🚀 Quick Start

### Validate All Plugins
```bash
owl env validate-docs
```

### Validate Specific Plugin
```bash
owl env validate-docs example
```

### CI/CD Mode
```bash
owl env validate-docs --strict
```

## 📊 Commands Reference

| Command | Description | Example |
|---------|-------------|---------|
| `owl env validate-docs` | Validate all plugins | `owl env validate-docs` |
| `owl env validate-docs PLUGIN` | Validate specific plugin | `owl env validate-docs example` |
| `owl env validate-docs --strict` | Enforce 100% coverage | `owl env validate-docs --strict` |
| `owl env validate-docs --format json` | JSON output | `owl env validate-docs --format json` |
| `owl env validate-docs --check-quality` | Detailed quality checks | `owl env validate-docs --check-quality` |
| `owl env docs-stats` | Show statistics | `owl env docs-stats` |
| `owl env docs-stats --by-type` | Group by component type | `owl env docs-stats --by-type` |

## 🎯 Exit Codes

| Code | Meaning | Usage |
|------|---------|-------|
| 0 | All validations passed | Success in CI/CD |
| 1 | Documentation issues found | Fail in CI/CD |
| 2 | Command error | Fix command usage |

## 📝 mkdocstrings Usage

### Plugin Overview
```markdown
::: example
    handler: owa
```

### Individual Component
```markdown
::: example/mouse.click
    handler: owa
    options:
      show_signature: true
      show_examples: true
```

### Configuration
```yaml
# mkdocs.yml
plugins:
  - mkdocstrings:
      handlers:
        owa:
          options:
            show_plugin_metadata: true
            include_source_links: true
```

**Note**: Requires `pip install -e projects/owa-core` for entry point registration.

## ✅ Documentation Checklist

### Required Elements
- [ ] Non-empty docstring
- [ ] Clear summary (first line)
- [ ] Parameter documentation (if applicable)
- [ ] Return value documentation
- [ ] Usage examples
- [ ] Type hints

### Example: Well-Documented Function
```python
def screen_capture(region: Optional[Tuple[int, int, int, int]] = None) -> Image:
    """
    Capture a screenshot of the desktop or a specific region.
    
    This function captures the current state of the desktop and returns
    it as a PIL Image object.
    
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
```

## 🔧 CI/CD Integration

### GitHub Actions
```yaml
- name: Validate Documentation
  run: owl env validate-docs --strict
```

### Pre-commit Hook
```yaml
- repo: local
  hooks:
    - id: validate-docs
      name: Validate Plugin Documentation
      entry: owl env validate-docs
      language: system
      pass_filenames: false
```

### pytest Integration
```python
def test_plugin_documentation():
    result = subprocess.run(
        ["owl", "env", "validate-docs", "--format", "json"],
        capture_output=True, text=True
    )
    data = json.loads(result.stdout)
    assert data["overall_coverage"] >= 0.9
    assert result.returncode == 0
```

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| "Missing docstring" | Add docstring to component |
| "Missing type hints" | Add type annotations |
| "Missing usage examples" | Add Examples section to docstring |
| "Component failed to load" | Check imports and dependencies |
| "Summary too short" | Write descriptive summary (>10 chars) |

## 📈 Output Examples

### Text Output
```
✅ std plugin: 2/2 components documented (100%)
⚠️  desktop plugin: 23/25 components documented (92%)
❌ custom plugin: 1/5 components documented (20%)

📊 Overall: 26/32 components documented (81%)
❌ FAIL: Documentation coverage 81% below minimum 90%
```

### JSON Output
```json
{
  "overall_coverage": 0.81,
  "plugins": {
    "std": {
      "coverage": 1.0,
      "documented": 2,
      "total": 2,
      "status": "pass"
    }
  },
  "exit_code": 1
}
```

## 🎯 Best Practices

1. **Write Clear Summaries**: First line should be descriptive
2. **Document Parameters**: Use Args: section for all parameters
3. **Include Examples**: Show practical usage
4. **Use Type Hints**: Add proper type annotations
5. **Test Documentation**: Run validation regularly
6. **Automate Checks**: Add to CI/CD pipeline

## 🔗 Related Documentation

- [Full Documentation Guide](documentation_validation.md)
- [Custom Plugin Development](custom_plugins.md)
- [OEP-0004 Specification](../../oeps/oep-0004.md)
