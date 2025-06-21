# Coverage Reporting

This document explains how to use and configure coverage reporting for the Open World Agents project.

## Overview

We use `pytest-cov` for coverage reporting, which provides:
- Line coverage tracking
- HTML reports with detailed analysis
- XML reports for CI integration
- Terminal output with missing line information

## Running Coverage Locally

### Quick Start

```bash
# Run tests with coverage and generate reports
python scripts/run_coverage.py

# Or manually with pytest
pytest --cov --cov-report=html --cov-report=term-missing
```

### Coverage Reports

After running coverage, you'll find:
- **HTML Report**: `htmlcov/index.html` - Interactive, detailed coverage report
- **XML Report**: `coverage.xml` - Machine-readable format for CI
- **Terminal Output**: Summary with missing lines

### Opening HTML Report

The HTML report provides the most detailed view:
```bash
# On Windows
start htmlcov/index.html

# On macOS
open htmlcov/index.html

# On Linux
xdg-open htmlcov/index.html
```

## CI Integration

### GitHub Actions

Coverage is automatically generated in our CI pipeline:
- Runs on every push and pull request
- Uploads reports as GitHub artifacts
- Shows coverage percentage in PR comments
- Displays coverage summary in workflow output

### GitHub-Only Coverage Setup

Our coverage setup uses only GitHub's built-in features:

1. **Automatic Coverage Reports**:
   - Coverage runs automatically on every push and PR
   - Reports are uploaded as GitHub artifacts
   - Coverage summary appears in workflow summaries

2. **Viewing Coverage**:
   - Go to Actions → "Pytest on Windows (common)" workflow
   - Download the coverage artifact from any workflow run
   - Open `htmlcov/index.html` for detailed coverage

3. **PR Comments**:
   - Coverage percentage is automatically commented on PRs
   - No external services or tokens required

## Configuration

### Coverage Settings

Coverage configuration is in `pyproject.toml`:
```toml
[tool.coverage.run]
source = ["projects"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "*/venv/*",
    "*/.venv/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "raise NotImplementedError",
    # ... more exclusions
]
show_missing = true
skip_covered = false
```

### Excluding Code from Coverage

Use `# pragma: no cover` to exclude specific lines:
```python
def debug_function():  # pragma: no cover
    print("Debug information")
```

## Coverage Goals

- **Target**: Maintain >80% overall coverage
- **New Code**: Aim for >90% coverage on new features
- **Critical Paths**: 100% coverage for core functionality

## Troubleshooting

### Common Issues

1. **Missing Coverage Data**:
   - Ensure tests are running with `--cov` flag
   - Check that source paths are correct

2. **Low Coverage**:
   - Review HTML report to identify untested code
   - Add tests for missing functionality
   - Consider if code should be excluded

3. **CI Failures**:
   - Check that `CODECOV_TOKEN` is set correctly
   - Verify coverage.xml is being generated

### Getting Help

- Check the [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- Review [Codecov documentation](https://docs.codecov.com/)
- Ask in project discussions for project-specific questions
