#!/bin/bash
set -e

MINIFORGE_VERSION=${1:-latest}
CONDA_INSTALL_PATH=${2:-/opt/conda}

echo "Installing Miniforge version: $MINIFORGE_VERSION"
echo "Installation path: $CONDA_INSTALL_PATH"

# Set up environment
export DEBIAN_FRONTEND=noninteractive

# Install basic system dependencies
apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates && rm -rf /var/lib/apt/lists/*

# Create installation directory
mkdir -p "$(dirname "$CONDA_INSTALL_PATH")"

# Download and install Miniforge
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
[ "$MINIFORGE_VERSION" != "latest" ] && \
    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-x86_64.sh"

wget -O miniforge.sh "$MINIFORGE_URL"
bash miniforge.sh -b -p "$CONDA_INSTALL_PATH"
rm miniforge.sh

# Initialize and configure conda
"$CONDA_INSTALL_PATH/bin/conda" init --all
"$CONDA_INSTALL_PATH/bin/conda" config --set auto_activate_base false --set channel_priority strict
"$CONDA_INSTALL_PATH/bin/conda" clean -afy

echo "Miniforge installation completed at: $CONDA_INSTALL_PATH"