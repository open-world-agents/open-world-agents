#!/bin/bash
set -e

PROJECT_DIR=${1:-/workspace}

# Clone the project
git clone --depth=1 https://github.com/open-world-agents/open-world-agents "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Set up conda/mamba environment
mamba create -n owa python=3.11 -y
conda config --set auto_activate_base false
. activate owa

# Install uv package manager and dependencies
pip install uv virtual-uv
# NOTE: `uv pip install --no-sources` ignores both path and editable, so I avoided it.
# Waiting for `--no-editable`. related issue: https://github.com/astral-sh/uv/issues/13087
find . -name "pyproject.toml" -exec cp {} {}.bak \;
find . -name "pyproject.toml" -exec sed -i 's/editable = true/editable = false/g' {} +
uv pip install projects/ocap
find . -name "pyproject.toml.bak" -exec sh -c 'mv -f "$1" "${1%.bak}"' _ {} \;


echo "Runtime environment setup complete!"
echo "Virtual environment: $(which python)"
echo "Python version: $(python --version)"
echo "uv version: $(uv --version)"
