# OWA mkdocstrings Handler

Custom mkdocstrings handler for OWA EnvPlugin components.

This package provides a specialized mkdocstrings handler that understands OWA's plugin structure and can generate documentation automatically for OWA plugins and their components.

## Installation

This package is automatically included when installing OWA with documentation dependencies:

```bash
pip install owa[docs]
```

Or install directly:

```bash
pip install owa-mkdocstrings-handler
```

## Usage

Add the handler to your `mkdocs.yml`:

```yaml
plugins:
  - mkdocstrings:
      handlers:
        owa:
          options:
            show_plugin_metadata: true
            include_source_links: true
```

Then use it in your documentation:

```markdown
# Plugin overview
::: example
    handler: owa

# Individual component
::: example/mouse.click
    handler: owa
    options:
      show_signature: true
      show_examples: true
```

## Features

- Automatic plugin discovery and documentation generation
- Component-level documentation with signatures and source code
- Integration with OWA's registry system
- Support for callables, listeners, and runnables
- Graceful fallback when dependencies are not available
