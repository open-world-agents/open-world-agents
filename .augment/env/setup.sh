#!/bin/bash
set -e

echo "Setting up Open World Agents development environment..."

# Step 1: Setup Python 3.11
echo "Setting up Python 3.11..."
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

# Step 2: Create and activate virtual environment (replicating setup-venv action)
echo "Creating virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate
echo "VIRTUAL_ENV=$PWD/.venv" >> /etc/profile
echo "export PATH=\"$PWD/.venv/bin:\$PATH\"" >> /etc/profile
echo "Virtual environment activated: $VIRTUAL_ENV"
which python

# Step 3: Install uv (replicating setup-uv-env action)
echo "Installing uv..."
pip install uv
echo "Installing virtual-uv..."
pip install virtual-uv

# Step 4: Install dependencies (replicating the workflow)
echo "Installing development dependencies..."
vuv install --dev

# Step 5: Install example plugin (replicating the workflow)
echo "Installing owa-env-example plugin..."
vuv pip install -e projects/owa-env-example

echo "Development environment setup complete!"
echo "Virtual environment activated at: $(which python)"
echo "Python version: $(python --version)"
echo "uv version: $(uv --version)"
echo "pytest version: $(python -m pytest --version)"