# Contributing to Open World Agents

This guide covers OWA-specific development practices and testing requirements.

## Issues and Pull Requests

Questions, feature requests and bug reports are welcome as [discussions or issues](https://github.com/open-world-agents/open-world-agents/issues/new/choose).

!!! warning "Security Vulnerabilities"
    For security vulnerabilities, see our [security policy](https://github.com/open-world-agents/open-world-agents/security/policy).

## Development Setup

!!! info "Complete Setup Instructions"
    See the [Installation Guide](install.md#development-installation-editable) for complete development setup instructions.

Quick setup for contributors:

```bash
git clone https://github.com/open-world-agents/open-world-agents.git
cd open-world-agents
conda create -n owa python=3.11 open-world-agents::gstreamer-bundle -y && conda activate owa
pip install uv virtual-uv
vuv install --dev
vuv pip install -e projects/owa-env-example  # For testing
```

## Testing Requirements

Before submitting a PR, run these OWA-specific checks:

=== "Code Quality"

    ```bash
    ruff check --fix      # Fix linting issues
    ruff format --check   # Check formatting
    ```

=== "Plugin Documentation"

    OWA validates environment plugin documentation automatically:

    ```bash
    owl env docs --validate --strict
    ```

=== "Test Suite"

    ```bash
    coverage run -m pytest  # Run all tests with coverage
    ```

## Environment Plugin Development

!!! info "Custom Plugin Development"
    For creating custom environment plugins, see the [Custom EnvPlugin Development Guide](env/custom_plugins.md) which covers:

    - Plugin structure and requirements
    - Entry point registration
    - Component types (Callables, Listeners, Runnables)
    - Complete examples and troubleshooting

## Documentation

OWA uses MkDocs with Material theme for documentation. The site includes auto-generated plugin documentation and manual content.

To work with documentation:

```bash
vuv install --extra docs  # Install MkDocs and dependencies
vuv run mkdocs serve       # Serve locally at http://localhost:8000
```

Documentation validation happens automatically in CI via `owl env docs --validate --strict`.

## Monorepo Development

OWA uses `virtual-uv` for dependency management. For complete setup instructions, see [Installation Guide](install.md#development-installation-editable).

Quick commands:
```bash
vuv install --dev              # Install all dev dependencies
vuv pip install -e projects/X  # Install specific project
```

## Release Management

For maintainers, OWA includes release automation scripts:

!!! info "Release Scripts"
    See [`scripts/release/README.md`](https://github.com/open-world-agents/open-world-agents/tree/main/scripts/release) for lockstep versioning and PyPI publishing tools.

    ```bash
    # Update all packages to version 1.0.0
    vuv run scripts/release/main.py version 1.0.0

    # Publish to PyPI
    vuv run scripts/release/main.py publish
    ```