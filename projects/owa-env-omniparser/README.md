# OmniParser Environment for Open World Agents

This module integrates [Microsoft OmniParser](https://github.com/microsoft/OmniParser) into the Open World Agents framework, enabling UI screenshot parsing capabilities.

## Features

- Parse UI screenshots to identify interactive elements
- Generate descriptions of detected UI elements
- Support for both embedded mode and optional API server mode
- Fallback mechanisms for reliable operation

## Installation

### Prerequisites

- Python 3.11 or higher
- OmniParser dependencies (automatically installed)

### Install using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is the recommended package manager for Open World Agents projects. It offers faster, more reliable dependency resolution.

```bash
# Clone the Open World Agents repository
git clone https://github.com/your-org/open-world-agents.git
cd open-world-agents

# Install uv if you don't have it already
pip install uv

# Install the OmniParser environment module using uv
uv pip install -e projects/owa-env-omniparser
```

### Install from source with pip

```bash
# Clone the Open World Agents repository
git clone https://github.com/your-org/open-world-agents.git
cd open-world-agents

# Install the OmniParser environment module
pip install -e projects/owa-env-omniparser
```

## Usage

```python
from owa.registry import CALLABLES, activate_module

# Activate the OmniParser environment
activate_module("owa_env_omniparser")

# Capture screen and parse UI elements
screen_image = CALLABLES["screen.capture"]()
parsed_result = CALLABLES["screen.parse_omniparser"](screen_image)

# Access parsed elements
for element in parsed_result["parsed_content_list"]:
    print(f"Element: {element['element_type']}, Description: {element['description']}")
    print(f"Position: {element['center_coordinates']}")
```

## Configuration

The module behavior can be configured through environment variables:

- `OMNIPARSER_MODE`: Set to "embedded" (default) or "api"
- `OMNIPARSER_MODEL_PATH`: Custom path to OmniParser model weights
- `OMNIPARSER_API_URL`: URL for OmniParser API server (when using API mode)
- `OMNIPARSER_DEVICE`: Device to use for model inference ("cuda" or "cpu")
- `UV_PROJECT_ENVIRONMENT`: Path to your virtual environment (when using uv)

## Development with uv

For development purposes, you can use uv to manage dependencies:

```bash
# Sync development dependencies
uv sync --dev

# Run tests with uv
uv run pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
