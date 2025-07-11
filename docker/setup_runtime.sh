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
vuv pip install . --group dev

echo "Runtime environment setup complete!"
echo "Virtual environment: $(which python)"
echo "Python version: $(python --version)"
echo "uv version: $(uv --version)"
